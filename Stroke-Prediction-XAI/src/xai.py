"""
xai.py
======
Explainable AI ด้วย SHAP และ LIME เพื่อตอบ **Q4** และ **Q5**

แนวคิดสำคัญ:
    อธิบายที่ระดับ **classifier บนฟีเจอร์หลัง transform** (one-hot แล้ว)
    ไม่ใช่ที่ตัว Pipeline ทั้งก้อน เพราะ:
        * SHAP/LIME ต้องการ predict function ที่รับเมทริกซ์ตัวเลข
        * ทำให้ SHAP และ LIME เห็นฟีเจอร์ชุดเดียวกัน -> เปรียบเทียบกันได้ตรง ๆ
    แล้วค่อยแปลงชื่อคอลัมน์ one-hot กลับเป็น "base feature" เพื่อให้อ่านง่าย
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .preprocessing import base_feature_of, get_feature_names, group_onehot_columns


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def split_pipeline(model):
    """แยก preprocessing transformer กับ classifier ออกจาก Pipeline"""
    steps = getattr(model, "named_steps", {})
    prep = steps.get("prep")
    clf = steps.get("clf")
    if prep is None or clf is None:
        raise ValueError("โมเดลต้องเป็น Pipeline ที่มี step ชื่อ 'prep' และ 'clf'")
    return prep, clf


def transform_train_test(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """คืน (X_train_transformed, X_test_transformed, feature_names)"""
    prep, _ = split_pipeline(model)
    names = get_feature_names(prep)
    Xtr = pd.DataFrame(prep.transform(X_train), columns=names, index=X_train.index)
    Xte = pd.DataFrame(prep.transform(X_test), columns=names, index=X_test.index)
    return Xtr, Xte, names


def positive_class_values(shap_values, expected_value=None):
    """
    จัดการรูปร่างผลลัพธ์ของ SHAP ที่ต่างกันตามเวอร์ชัน/ชนิดโมเดล
    ให้เหลือเฉพาะ attributions ของ "คลาส stroke (1)"

    คืน (values_2d, expected_value_scalar)
    """
    # กรณี list ของ 2 คลาส (shap เดิม ๆ สำหรับ sklearn RF)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])
        ev = np.asarray(expected_value)[1] if expected_value is not None and np.ndim(expected_value) else expected_value
        return sv, (float(ev) if ev is not None else None)

    sv = np.asarray(shap_values)
    if sv.ndim == 3:  # (n_samples, n_features, n_classes)
        sv = sv[:, :, 1]
        ev = expected_value
        if ev is not None and np.ndim(ev):
            ev = np.asarray(ev)[1]
        return sv, (float(ev) if ev is not None else None)
    return sv, (float(expected_value) if expected_value is not None and not np.ndim(expected_value) else None)


def shap_space(base_value: float | None) -> str:
    """
    บอกว่า SHAP values อยู่ใน "หน่วย" ใด

    * TreeExplainer ของ sklearn (RandomForest) -> probability space (0-1)
    * TreeExplainer ของ XGBoost/LightGBM และ LinearExplainer -> margin space (log-odds)

    สำคัญเพราะสมการ additivity ต้องเทียบในหน่วยเดียวกัน:
        probability space : p = base + Σ φ
        margin space      : logit(p) = base + Σ φ
    """
    if base_value is None:
        return "unknown"
    return "probability" if 0.0 <= float(base_value) <= 1.0 else "log-odds"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_explainer(clf, X_background: pd.DataFrame, max_background: int = 200):
    """เลือก explainer ให้เหมาะกับชนิดของ classifier (มี fallback ครบ)"""
    import shap

    name = type(clf).__name__.lower()
    tree_like = any(k in name for k in ("randomforest", "xgb", "lgbm", "gradientboosting", "decisiontree", "extratrees"))
    linear_like = any(k in name for k in ("logistic", "linear", "ridge", "lasso", "svc"))

    if tree_like:
        return shap.TreeExplainer(clf)
    if linear_like:
        # LinearExplainer ใช้ masker ที่จำกัด max_samples=100 อยู่แล้ว
        # ถ้าส่ง background เกินไปจะโดน subsample + เตือนโดยไม่จำเป็น
        bg = X_background.sample(min(100, len(X_background)), random_state=cfg.RANDOM_STATE)
        try:
            return shap.LinearExplainer(clf, bg)
        except Exception:  # noqa: BLE001
            pass
    # fallback: permutation explainer (ช้ากว่าแต่ใช้ได้กับทุกโมเดล)
    bg = X_background.sample(min(60, len(X_background)), random_state=cfg.RANDOM_STATE)
    return shap.Explainer(clf.predict_proba, bg, algorithm="permutation")


def compute_shap(model, X_explain: pd.DataFrame, X_background: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float | None, object]:
    """
    คำนวณ SHAP values ของคลาส stroke

    Returns
    -------
    X_transformed : DataFrame ฟีเจอร์หลัง transform
    shap_values   : ndarray (n_samples, n_features)
    base_value    : ค่า expected value (prob ฐาน)
    explainer     : shap explainer ที่ใช้ (นำไปวาด force/waterfall ต่อได้)
    """
    prep, clf = split_pipeline(model)
    names = get_feature_names(prep)

    Xtr = pd.DataFrame(prep.transform(X_explain), columns=names, index=X_explain.index)
    Xbg = pd.DataFrame(prep.transform(X_background), columns=names, index=X_background.index)

    explainer = make_explainer(clf, Xbg)
    raw = explainer(Xtr) if hasattr(explainer, "__call__") and not hasattr(explainer, "shap_values") else explainer.shap_values(Xtr)

    if hasattr(raw, "values"):
        sv = raw.values
        ev = getattr(raw, "base_values", None)
        sv, base = positive_class_values(sv, ev)
        expl_obj = raw
    else:
        sv, base = positive_class_values(raw, getattr(explainer, "expected_value", None))
        expl_obj = None

    return Xtr, np.asarray(sv), base, explainer


# ---------------------------------------------------------------------------
# Global importance (Q4)
# ---------------------------------------------------------------------------
def global_importance(
    X_transformed: pd.DataFrame,
    shap_values: np.ndarray,
    group_to_base: bool = True,
) -> pd.DataFrame:
    """
    จัดอันดับความสำคัญของฟีเจอร์จาก mean(|SHAP|)

    ⚠️ ข้อควรระวังที่สำคัญ: สำหรับฟีเจอร์ไบนารีที่ **หายาก** (เช่น heart_disease
    ซึ่งพบเพียง ~5% ของ records) ค่า mean(|SHAP|) จะต่ำโดยโครงสร้าง
    เพราะ records ส่วนใหญ่มีค่าเป็น 0 → contribution เป็น 0
    เราจึงรายงานเพิ่ม 2 คอลัมน์ **สำหรับฟีเจอร์ไบนารีเท่านั้น**:
        * ``pct_records_active``        : % ของ records ที่มีฟีเจอร์นั้น "เปิด" (= 1)
        * ``mean_abs_shap_when_active`` : อิทธิพลเฉลี่ย **เฉพาะเมื่อฟีเจอร์เปิด**

    หมายเหตุ: แนวคิด "เปิด/ปิด" ใช้ได้เฉพาะตัวแปรไบนารี
    ตัวแปรต่อเนื่อง (หลัง StandardScaler) ไม่มีความหมายว่า "เปิด" จึงปล่อยเป็น NaN
    และตัวแปรเชิงกลุ่มที่ถูก one-hot จะถูกอ่านผ่าน ``mean_shap``/beeswarm แทน
    """
    cols = list(X_transformed.columns)
    abs_sv = np.abs(shap_values)

    if group_to_base:
        g_abs = group_onehot_columns(
            pd.Series(abs_sv.mean(axis=0), index=cols).to_frame().T
        ).T.iloc[:, 0]
        g_signed = group_onehot_columns(
            pd.Series(shap_values.mean(axis=0), index=cols).to_frame().T
        ).T.iloc[:, 0]

        df = pd.DataFrame(
            {
                "feature": g_abs.index,
                "mean_abs_shap": g_abs.values,
                "mean_shap": g_signed.values,
            }
        )

        active_pct, active_val = [], []
        for feat in df["feature"]:
            if feat in cfg.BINARY_FEATURES and feat in cols:
                active = X_transformed[feat].to_numpy() > 0
                active_pct.append(round(float(active.mean() * 100), 2))
                active_val.append(
                    round(float(abs_sv[active, cols.index(feat)].mean()), 4) if active.any() else np.nan
                )
            else:
                active_pct.append(np.nan)
                active_val.append(np.nan)
        df["pct_records_active"] = active_pct
        df["mean_abs_shap_when_active"] = active_val
    else:
        df = pd.DataFrame(
            {
                "feature": cols,
                "mean_abs_shap": abs_sv.mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
            }
        )
        df["pct_records_active"] = [
            round(float((X_transformed[c].to_numpy() > 0).mean() * 100), 2)
            if c in cfg.BINARY_FEATURES
            else np.nan
            for c in cols
        ]
        df["mean_abs_shap_when_active"] = np.nan

    df["label_th"] = df["feature"].map(cfg.FEATURE_LABELS_TH).fillna(df["feature"])
    total = df["mean_abs_shap"].sum()
    df["importance_pct"] = (df["mean_abs_shap"] / total * 100).round(2) if total else 0.0
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank_overall"] = np.arange(1, len(df) + 1)
    return df


def shap_values_to_base(X_transformed: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """แปลง SHAP matrix ให้เป็นระดับ base feature (รวม one-hot เข้าด้วยกัน)"""
    sv_df = pd.DataFrame(shap_values, columns=X_transformed.columns, index=X_transformed.index)
    return group_onehot_columns(sv_df)


# ---------------------------------------------------------------------------
# Local explanation (Q4)
# ---------------------------------------------------------------------------
def explain_records(
    model,
    X_records: pd.DataFrame,
    y_true: pd.Series | None = None,
    y_prob: np.ndarray | None = None,
    top_k: int = 8,
    X_background: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    อธิบายทีละ record: ฟีเจอร์ใดดันความน่าจะเป็นขึ้น/ลง มากที่สุด

    คืน DataFrame แบบ long: 1 แถว = 1 (record, feature) สำหรับ top_k ของ record นั้น
    """
    if X_background is None:
        X_background = X_records
    Xtr, sv, base, _ = compute_shap(model, X_records, X_background)
    sv_base = shap_values_to_base(Xtr, sv)

    records = []
    for i, idx in enumerate(X_records.index):
        contrib = sv_base.iloc[i]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_k]
        for feat, val in top.items():
            records.append(
                {
                    "record_index": idx,
                    "feature": feat,
                    "shap_value": round(float(val), 4),
                    "direction": "ดันให้เป็น stroke" if val > 0 else "ดันให้ไม่เป็น stroke",
                    "raw_value": X_records.loc[idx, feat] if feat in X_records.columns else np.nan,
                    "predicted_prob": round(float(y_prob[i]), 4) if y_prob is not None else np.nan,
                    "actual": int(y_true.loc[idx]) if y_true is not None else np.nan,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LIME (Q4 + Q5)
# ---------------------------------------------------------------------------
@dataclass
class LimeResult:
    record_index: object
    explanation: object
    top_features: pd.DataFrame


def run_lime(
    model,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    num_features: int = 8,
    num_samples: int = 5000,
) -> list[LimeResult]:
    """สร้าง LIME explanation (tabular) สำหรับแต่ละ record ใน ``X_explain``"""
    from lime.lime_tabular import LimeTabularExplainer

    prep, clf = split_pipeline(model)
    names = get_feature_names(prep)
    Xtr = pd.DataFrame(prep.transform(X_train), columns=names)
    Xex = pd.DataFrame(prep.transform(X_explain), columns=names)

    explainer = LimeTabularExplainer(
        training_data=Xtr.to_numpy(),
        feature_names=names,
        class_names=["No Stroke", "Stroke"],
        mode="classification",
        discretize_continuous=True,
        random_state=cfg.RANDOM_STATE,
    )

    results: list[LimeResult] = []
    for i, idx in enumerate(X_explain.index):
        exp = explainer.explain_instance(
            Xex.iloc[i].to_numpy(),
            clf.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),
        )
        rows = []
        for feat_cond, weight in exp.as_list(label=1):
            # LIME คืนชื่อแบบมีเงื่อนไข เช่น "age <= 45.00" -> ดึงชื่อคอลัมน์หลัง transform
            matched = _match_feature_name(feat_cond, names)
            # แล้วแปลงต่อเป็น "base feature" ให้ตรงกับที่เรา group SHAP ไว้
            # -> รับประกันว่า SHAP และ LIME อ้างอิงฟีเจอร์ชุดเดียวกันจริง
            base_feat = base_feature_of(matched)
            rows.append(
                {
                    "record_index": idx,
                    "feature_condition": feat_cond,
                    "feature_matched": matched,
                    "feature": base_feat,
                    "lime_weight": round(float(weight), 4),
                    "direction": "ดันให้เป็น stroke" if weight > 0 else "ดันให้ไม่เป็น stroke",
                }
            )
        top = pd.DataFrame(rows).sort_values("lime_weight", key=abs, ascending=False).reset_index(drop=True)
        results.append(LimeResult(record_index=idx, explanation=exp, top_features=top))
    return results


def _match_feature_name(condition: str, feature_names: list[str]) -> str:
    """จับคู่ feature จากข้อความเงื่อนไขของ LIME กลับเป็นชื่อคอลัมน์จริง"""
    candidates = [f for f in feature_names if condition.startswith(f)]
    if candidates:
        return max(candidates, key=len)
    # กันกรณี LIME ตัดชื่อ: เทียบแบบ substring
    for f in sorted(feature_names, key=len, reverse=True):
        if f in condition:
            return f
    return condition.split()[0]


def lime_results_to_frame(results: list[LimeResult]) -> pd.DataFrame:
    """รวมผล LIME ทุก record เป็น DataFrame เดียว"""
    return pd.concat([r.top_features for r in results], ignore_index=True)


# ---------------------------------------------------------------------------
# Q5: SHAP vs LIME agreement
# ---------------------------------------------------------------------------
def compare_shap_lime(
    model,
    X_records: pd.DataFrame,
    lime_results: list[LimeResult],
    X_background: pd.DataFrame | None = None,
    top_k: int = 8,
) -> pd.DataFrame:
    """
    วัดความสอดคล้องของ SHAP และ LIME ทีละ record

    ตัวชี้วัด:
        * jaccard_topk  : สัดส่วนฟีเจอร์ (base) ที่ติด top-k ของทั้งสองวิธี
        * overlap_count : จำนวนฟีเจอร์ที่ตรงกันใน top-k
        * spearman_rho  : สหสัมพันธ์ของขนาดอิทธิพล (เฉพาะฟีเจอร์ที่ทั้งคู่ให้ค่า)
        * sign_agreement: สัดส่วนทิศทาง (+/-) ที่ตรงกันในฟีเจอร์ที่ทับกัน
    """
    if X_background is None:
        X_background = X_records

    from scipy.stats import spearmanr

    Xtr, sv, _, _ = compute_shap(model, X_records, X_background)
    sv_base = shap_values_to_base(Xtr, sv)

    rows = []
    for r in lime_results:
        i = list(X_records.index).index(r.record_index)
        shap_series = sv_base.iloc[i]

        # รวม LIME ระดับ base feature: ถ้าหนึ่งฟีเจอร์มีหลายเงื่อนไข ให้รวมแบบ "บวก"
        # เพื่อให้สอดคล้องกับวิธีที่เรารวม one-hot ของ SHAP (group_onehot_columns.sum)
        lime_series = r.top_features.groupby("feature")["lime_weight"].sum()

        shap_top = set(shap_series.abs().sort_values(ascending=False).head(top_k).index)
        lime_top = set(lime_series.abs().sort_values(ascending=False).head(top_k).index)
        inter, union = shap_top & lime_top, shap_top | lime_top

        # คำนวณสหสัมพันธ์บน "ฟีเจอร์ทั้งหมดที่ทั้งสองวิธีให้ค่า" (ไม่จำกัดแค่ top-k)
        # เพราะการจำกัดเฉพาะ top-k ทำให้เหลือจุดน้อยเกินไปและ ρ ไม่เสถียร
        common = sorted(set(lime_series.index) & set(shap_series.index))
        if len(common) >= 3:
            rho, p = spearmanr(shap_series[common], lime_series[common])
        else:
            rho, p = np.nan, np.nan

        signs = [(np.sign(shap_series[f]) == np.sign(lime_series[f])) for f in common]

        rows.append(
            {
                "record_index": r.record_index,
                "overlap_count": len(inter),
                "jaccard_topk": round(len(inter) / len(union), 4) if union else np.nan,
                "n_common_features": len(common),
                "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else np.nan,
                "spearman_p": round(float(p), 4) if not np.isnan(p) else np.nan,
                "sign_agreement": round(float(np.mean(signs)), 4) if signs else np.nan,
                "shap_top": ", ".join(sorted(shap_top)),
                "lime_top": ", ".join(sorted(lime_top)),
            }
        )
    return pd.DataFrame(rows)
