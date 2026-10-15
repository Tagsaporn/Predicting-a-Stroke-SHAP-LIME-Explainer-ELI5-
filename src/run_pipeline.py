"""
run_pipeline.py
===============
รันทั้งโปรเจกต์แบบ end-to-end (สร้างผลลัพธ์ทั้งหมดในคำสั่งเดียว)

    python -m src.run_pipeline              # รันเต็ม
    python -m src.run_pipeline --quick      # ลดจำนวน record สำหรับ LIME/SHAP (ทดสอบเร็ว)
    python -m src.run_pipeline --skip-lime  # ข้าม LIME (ไม่ต้องใช้เวลา)

ผลลัพธ์ที่ได้:
    data/processed/stroke_clean.csv
    reports/figures/**.png
    reports/tables/**.csv
    reports/metrics.json
    models/best_model.joblib
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from . import data_loader as dl
from . import models as mdl
from . import stats_analysis as sa
from . import visualization as viz
from . import xai

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Step 1-2: data + EDA
# ---------------------------------------------------------------------------
def step_data(quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    banner("STEP 1 | โหลดข้อมูลและทำความสะอาด")
    cfg.ensure_dirs()

    raw = dl.load_raw()
    print(f"ข้อมูลดิบ: {raw.shape[0]:,} แถว x {raw.shape[1]} คอลัมน์")
    raw.to_csv(cfg.RAW_DATA_DIR / "healthcare-dataset-stroke-data.csv", index=False)

    quality = dl.data_quality_report(raw)
    quality.to_csv(cfg.TABLES_DIR / "t_data_quality_raw.csv", index=False, encoding="utf-8-sig")
    print(quality.to_string(index=False))

    clean = dl.basic_clean(raw)
    clean.to_csv(cfg.PROCESSED_DATA_DIR / "stroke_clean.csv", index=False, encoding="utf-8-sig")
    print(f"\nหลังทำความสะอาด: {clean.shape[0]:,} แถว (ลบ gender='Other' และคอลัมน์ id)")
    print("สัดส่วนคลาส:", clean[cfg.TARGET].value_counts(normalize=True).round(4).to_dict())

    return raw, clean


def step_eda(clean: pd.DataFrame) -> None:
    banner("STEP 2 | EDA & Visualization (Q2)")

    viz.plot_target_distribution(clean)
    viz.plot_missing_values(clean)
    viz.plot_numeric_by_target(clean)
    viz.plot_categorical_vs_target(clean)
    viz.plot_correlation_heatmap(clean)
    viz.plot_age_glucose_bmi_scatter(clean)
    viz.plot_vascular_risk(clean)

    # ตารางสรุปสำหรับ Q2
    summary = pd.DataFrame(
        {
            "metric": [
                "n_records", "n_features", "n_stroke", "n_no_stroke",
                "stroke_prevalence_%", "imbalance_ratio (neg:pos)",
                "missing_bmi", "missing_bmi_%", "smoking_unknown_%",
            ],
            "value": [
                len(clean),
                len(cfg.ALL_FEATURES),
                int((clean[cfg.TARGET] == 1).sum()),
                int((clean[cfg.TARGET] == 0).sum()),
                round(clean[cfg.TARGET].mean() * 100, 3),
                round((clean[cfg.TARGET] == 0).sum() / max((clean[cfg.TARGET] == 1).sum(), 1), 2),
                int(clean["bmi"].isna().sum()),
                round(clean["bmi"].isna().mean() * 100, 3),
                round((clean["smoking_status"] == "Unknown").mean() * 100, 3),
            ],
        }
    )
    summary.to_csv(cfg.TABLES_DIR / "q2_dataset_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Step 3: statistics (Q1)
# ---------------------------------------------------------------------------
def step_stats(clean: pd.DataFrame) -> sa.Q1Results:
    banner("STEP 3 | Statistical Analysis (Q1)")

    results = sa.run_q1_analysis(clean)

    print("\n[Chi-square] ตัวแปรเชิงกลุ่ม vs stroke")
    print(results.chi_square[["feature", "chi2", "p_value", "p_adjusted_bh",
                              "cramers_v", "effect_size", "significant_0.05"]].to_string(index=False))

    print("\n[Mann-Whitney U] ตัวแปรต่อเนื่อง vs stroke")
    print(results.mann_whitney[["feature", "u_statistic", "p_value", "p_adjusted_bh",
                                "median_no_stroke", "median_stroke",
                                "rank_biserial", "significant_0.05"]].to_string(index=False))

    print("\n[Odds Ratio] ตัวแปรไบนารี")
    print(results.odds_ratios.to_string(index=False))

    for name, table in results.to_dict().items():
        table.to_csv(cfg.TABLES_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    return results


# ---------------------------------------------------------------------------
# Step 4: modelling (Q3)
# ---------------------------------------------------------------------------
def step_models(clean: pd.DataFrame, quick: bool = False) -> dict:
    banner("STEP 4 | Model Comparison (Q3)")

    X, y = dl.split_features_target(clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )
    print(f"Train: {len(X_train):,} (stroke {y_train.sum()}) | Test: {len(X_test):,} (stroke {y_test.sum()})")

    zoo = mdl.get_model_zoo(y_train)
    if quick:
        for name in list(zoo)[:2]:  # ลดจำนวนโมเดลเมื่อรันแบบเร็ว
            pass

    result = mdl.compare_models(X_train, y_train, X_test, y_test, model_zoo=zoo, verbose=True)

    summary = result.summary
    summary.to_csv(cfg.TABLES_DIR / "q3_model_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n--- สรุปผลเปรียบเทียบโมเดล (เรียงตาม Recall) ---")
    print(summary[["model", "threshold", "recall_sensitivity", "precision", "f1",
                   "pr_auc_average_precision", "roc_auc", "specificity", "accuracy"]].to_string(index=False))

    # confusion matrix ของทุกโมเดล
    cm_table = pd.concat(
        {
            name: mdl.confusion_frame(cm)
            for name, cm in result.confusion_matrices.items()
        }
    )
    cm_table.to_csv(cfg.TABLES_DIR / "q3_confusion_matrices.csv", encoding="utf-8-sig")

    viz.plot_model_comparison(summary)
    viz.plot_confusion_matrices(result.confusion_matrices)

    # PR/ROC curves
    curves = {}
    for name, model in result.fitted_models.items():
        curves[name] = {
            "y_true": y_test.to_numpy(),
            "y_prob": model.predict_proba(X_test)[:, 1],
        }
    viz.plot_pr_roc_curves(curves)

    # เลือกโมเดลที่ดีที่สุด (F1 ก่อน, tie-break ด้วย Recall)
    best_name, best_model, best_thr = mdl.select_best_model(result, primary="f1")
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "threshold": best_thr, "name": best_name},
                cfg.MODELS_DIR / "best_model.joblib")
    print(f"\n>>> โมเดลที่ดีที่สุด (ตาม F1): {best_name} | threshold = {best_thr:.3f}")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "result": result, "best_name": best_name,
        "best_model": best_model, "best_threshold": best_thr,
    }


# ---------------------------------------------------------------------------
# Step 5-6: SHAP / LIME (Q4, Q5)
# ---------------------------------------------------------------------------
def step_xai(ctx: dict, quick: bool = False, skip_lime: bool = False) -> dict:
    banner("STEP 5 | SHAP Global & Local Explanation (Q4)")

    best_model = ctx["best_model"]
    X_train, X_test = ctx["X_train"], ctx["X_test"]
    y_test = ctx["y_test"]
    threshold = ctx["best_threshold"]

    # ---- global ----
    sample_n = 300 if quick else 1500
    X_sample = X_test.sample(min(sample_n, len(X_test)), random_state=cfg.RANDOM_STATE)
    Xtr, sv, base_value, explainer = xai.compute_shap(best_model, X_sample, X_train)
    space = xai.shap_space(base_value)
    print(f"คำนวณ SHAP สำเร็จ: {sv.shape[0]} records x {sv.shape[1]} engineered features")
    print(f"base value = {base_value} | หน่วยของค่า SHAP = {space}")
    if space == "log-odds":
        print("  หมายเหตุ: LinearExplainer/XGBoost ให้ค่าในหน่วย log-odds "
              "การตีความ 'ทิศทาง' ยังเหมือนเดิม แต่ขนาดไม่ใช่สเกลความน่าจะเป็น")

    importance = xai.global_importance(Xtr, sv, group_to_base=True)
    importance.to_csv(cfg.TABLES_DIR / "q4_shap_global_importance.csv", index=False, encoding="utf-8-sig")
    print("\n--- Global importance (mean |SHAP|, จัดกลุ่ม one-hot แล้ว) ---")
    print(importance[["feature", "mean_abs_shap", "importance_pct",
                      "pct_records_active", "mean_abs_shap_when_active"]].head(12).to_string(index=False))
    print("\n  ⚠️ ฟีเจอร์ไบนารีที่หายาก (เช่น heart_disease) มี mean|SHAP| ต่ำโดยโครงสร้าง")
    print("     -> ให้ดูคอลัมน์ mean_abs_shap_when_active ประกอบเสมอ")

    viz.plot_shap_bar(importance)
    viz.plot_shap_beeswarm(
        sv,
        Xtr,
        feature_labels={**cfg.FEATURE_LABELS_TH,
                        **{f"work_type_{v}": f"work_type={v}" for v in X_test['work_type'].unique()},
                        **{f"smoking_status_{v}": f"smoking={v}" for v in X_test['smoking_status'].unique()},
                        **{f"ever_married_{v}": f"married={v}" for v in X_test['ever_married'].unique()},
                        **{f"gender_{v}": f"gender={v}" for v in X_test['gender'].unique()},
                        **{f"Residence_type_{v}": f"residence={v}" for v in X_test['Residence_type'].unique()}},
    )

    # ---- local cases: TP, FN, FP ----
    prob = best_model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    cases = {}
    for label, mask in (
        ("TP (ทำนายถูก-เป็น stroke)", (pred == 1) & (y_test.to_numpy() == 1)),
        ("FN (พลาดผู้ป่วยจริง)", (pred == 0) & (y_test.to_numpy() == 1)),
        ("FP (เตือนเกิน)", (pred == 1) & (y_test.to_numpy() == 0)),
    ):
        idx = np.where(mask)[0]
        if len(idx):
            cases[label] = X_test.index[idx[0]]

    print("\n--- เคสตัวอย่างสำหรับ local explanation ---")
    for label, idx in cases.items():
        pos = X_test.index.get_loc(idx)
        print(f"  {label}: index={idx} | prob={prob[pos]:.3f}")

    local_frames = []
    lime_results = []
    for label, idx in cases.items():
        one = X_test.loc[[idx]]
        p = float(best_model.predict_proba(one)[:, 1][0])

        frame = xai.explain_records(best_model, one, y_true=y_test, y_prob=np.array([p]),
                                    top_k=10, X_background=X_train)
        frame.insert(0, "case", label)
        local_frames.append(frame)

        # waterfall สำหรับเคสนี้
        try:
            _, sv_one, _, _ = xai.compute_shap(best_model, one, X_train)
            import shap

            expl = shap.Explanation(
                values=sv_one[0],
                base_values=float(base_value) if base_value is not None else 0.0,
                data=xai.transform_train_test(best_model, one, one)[0].iloc[0].to_numpy(),
                feature_names=xai.transform_train_test(best_model, one, one)[2],
            )
            viz.plot_shap_waterfall(expl, max_display=10,
                                    filename=f"22_shap_waterfall_{label.split()[0]}.png")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] waterfall ไม่สำเร็จสำหรับ {label}: {exc}")

    local_df = pd.concat(local_frames, ignore_index=True)
    local_df.to_csv(cfg.TABLES_DIR / "q4_shap_local_cases.csv", index=False, encoding="utf-8-sig")

    # ---- Q5: LIME ----
    comparison = None
    if not skip_lime:
        banner("STEP 6 | SHAP vs LIME (Q5)")
        n_lime = 3 if quick else 15
        rng = np.random.default_rng(cfg.RANDOM_STATE)
        # สุ่ม record ที่โมเดลให้ความสนใจ: ครึ่งหนึ่งเป็นผู้ป่วยจริง
        pos_idx = X_test.index[y_test.to_numpy() == 1]
        neg_idx = X_test.index[y_test.to_numpy() == 0]
        pick_pos = rng.choice(pos_idx, size=min(len(pos_idx), max(1, n_lime // 2)), replace=False)
        pick_neg = rng.choice(neg_idx, size=min(len(neg_idx), n_lime - len(pick_pos)), replace=False)
        lime_idx = list(pick_pos) + list(pick_neg)

        lime_input = X_test.loc[lime_idx]
        lime_results = xai.run_lime(best_model, X_train, lime_input,
                                    num_features=8, num_samples=3000 if quick else 5000)

        lime_df = xai.lime_results_to_frame(lime_results)
        lime_df.to_csv(cfg.TABLES_DIR / "q4_lime_local.csv", index=False, encoding="utf-8-sig")

        for r in lime_results[:3]:
            viz.plot_lime_explanation(r)

        comparison = xai.compare_shap_lime(best_model, lime_input, lime_results,
                                           X_background=X_train, top_k=8)
        comparison.insert(0, "actual", [int(y_test.loc[i]) for i in lime_idx])
        comparison.to_csv(cfg.TABLES_DIR / "q5_shap_vs_lime.csv", index=False, encoding="utf-8-sig")
        viz.plot_shap_lime_comparison(comparison)

        print("\n--- ความสอดคล้อง SHAP vs LIME ---")
        print(comparison[["record_index", "actual", "overlap_count", "jaccard_topk",
                          "spearman_rho", "sign_agreement"]].to_string(index=False))
        print(f"\nค่าเฉลี่ย: Jaccard = {comparison['jaccard_topk'].mean():.3f} | "
              f"Spearman ρ = {comparison['spearman_rho'].mean():.3f} | "
              f"Sign agreement = {comparison['sign_agreement'].mean():.3f}")

    return {"importance": importance, "local": local_df,
            "lime": lime_results, "comparison": comparison,
            "base_value": base_value, "space": space}


# ---------------------------------------------------------------------------
# Metrics + report
# ---------------------------------------------------------------------------
def step_report(ctx: dict, xai_out: dict, stats: sa.Q1Results) -> None:
    banner("STEP 7 | สรุปรวมผลลัพธ์")

    summary = ctx["result"].summary
    best = summary.loc[summary["model"] == ctx["best_name"]].iloc[0]

    metrics = {
        "dataset": {
            "n_records": int(len(ctx["X_train"]) + len(ctx["X_test"])),
            "n_features": len(cfg.ALL_FEATURES),
            "test_size": cfg.TEST_SIZE,
            "random_state": cfg.RANDOM_STATE,
        },
        "best_model": {
            "name": ctx["best_name"],
            "threshold": float(ctx["best_threshold"]),
            "recall": float(best["recall_sensitivity"]),
            "precision": float(best["precision"]),
            "f1": float(best["f1"]),
            "pr_auc": float(best["pr_auc_average_precision"]),
            "roc_auc": float(best["roc_auc"]),
            "specificity": float(best["specificity"]),
            "confusion_matrix": ctx["result"].confusion_matrices[ctx["best_name"]].tolist(),
        },
        "q1_significant_categorical": stats.chi_square.loc[
            stats.chi_square["significant_0.05"], "feature"].tolist(),
        "q1_significant_continuous": stats.mann_whitney.loc[
            stats.mann_whitney["significant_0.05"], "feature"].tolist(),
        "shap_space": xai.shap_space(xai_out.get("base_value")),
        "top_shap_features": xai_out["importance"].head(5)["feature"].tolist(),
        "top_shap_features_when_active": (
            xai_out["importance"]
            .dropna(subset=["mean_abs_shap_when_active"])
            .sort_values("mean_abs_shap_when_active", ascending=False)
            .head(5)["feature"]
            .tolist()
        ),
    }

    if xai_out.get("comparison") is not None:
        comp = xai_out["comparison"]
        metrics["q5_shap_lime_agreement"] = {
            "mean_jaccard_top8": round(float(comp["jaccard_topk"].mean()), 4),
            "mean_spearman_rho": round(float(comp["spearman_rho"].mean()), 4),
            "mean_sign_agreement": round(float(comp["sign_agreement"].mean()), 4),
            "n_records": int(len(comp)),
        }

    with open(cfg.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nบันทึกผลลัพธ์ทั้งหมดแล้วที่: {cfg.REPORTS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stroke Prediction + XAI pipeline")
    parser.add_argument("--quick", action="store_true", help="ลดขนาดงานสำหรับทดสอบเร็ว")
    parser.add_argument("--skip-lime", action="store_true", help="ข้ามขั้นตอน LIME")
    args = parser.parse_args(argv)

    raw, clean = step_data(quick=args.quick)
    step_eda(clean)
    stats = step_stats(clean)
    ctx = step_models(clean, quick=args.quick)
    xai_out = step_xai(ctx, quick=args.quick, skip_lime=args.skip_lime)
    step_report(ctx, xai_out, stats)

    print("\n✅ เสร็จสิ้น — ดูรายงานได้ที่ reports/ และรัน dashboard ด้วย `streamlit run dashboard/app.py`")


if __name__ == "__main__":
    main()
