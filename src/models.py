"""
models.py
=========
เทรนและเปรียบเทียบโมเดลจำแนก stroke เพื่อตอบ **Q3**

Design decisions:
    * ใช้ ``Pipeline`` รวม preprocessing + classifier เสมอ -> fit ใน CV ถูกต้อง
    * จัดการ class imbalance ด้วย 2 กลยุทธ์ที่เทียบกันได้:
        1) class weighting (class_weight / scale_pos_weight)
        2) oversampling ด้วย SMOTE (imbalanced-learn) *ภายใน* train fold เท่านั้น
    * เลือก threshold จาก Out-of-Fold (OOF) predictions ไม่ใช่จาก test set
      -> test set ยังคงบริสุทธิ์สำหรับการประเมินขั้นสุดท้าย
    * metric ที่เน้น: Recall, F1, PR-AUC (Average Precision), Confusion Matrix
      เพราะข้อมูลไม่สมดุล (stroke ≈ 4.9%) ทำให้ Accuracy หลอกตา
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as cfg
from .preprocessing import build_column_transformer


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def _scale_pos_weight(y: pd.Series) -> float:
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))


def get_model_zoo(y_train: pd.Series | None = None) -> dict[str, BaseEstimator]:
    """
    คืน dict ของ pipeline ที่ยังไม่ fit: {ชื่อโมเดล: estimator}

    ทุกตัวใช้ preprocessing ชุดเดียวกันเพื่อให้เปรียบเทียบได้อย่างเป็นธรรม
    """
    spw = _scale_pos_weight(y_train) if y_train is not None else 1.0

    def pipe(clf: BaseEstimator) -> ImbPipeline:
        return ImbPipeline(
            steps=[
                ("prep", build_column_transformer(scale_numeric=True)),
                ("clf", clf),
            ]
        )

    models: dict[str, BaseEstimator] = {
        "LogReg (class_weight=balanced)": pipe(
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=cfg.RANDOM_STATE,
            )
        ),
        "LogReg (SMOTE)": ImbPipeline(
            steps=[
                ("prep", build_column_transformer(scale_numeric=True)),
                (
                    "smote",
                    SMOTE(random_state=cfg.RANDOM_STATE, k_neighbors=5),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        solver="liblinear",
                        random_state=cfg.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest (class_weight=balanced_subsample)": pipe(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=cfg.RANDOM_STATE,
            )
        ),
        "RandomForest (SMOTE)": ImbPipeline(
            steps=[
                ("prep", build_column_transformer(scale_numeric=True)),
                ("smote", SMOTE(random_state=cfg.RANDOM_STATE, k_neighbors=5)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=cfg.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost (scale_pos_weight)": pipe(
            _make_xgboost(spw)
        ),
        "LightGBM (class_weight=balanced)": pipe(
            _make_lightgbm()
        ),
    }
    return models


def _make_xgboost(scale_pos_weight: float):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
    )


def _make_lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        verbose=-1,
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Threshold selection & evaluation
# ---------------------------------------------------------------------------
def metrics_at_threshold(y_true, y_prob, threshold: float = 0.5) -> dict:
    """คำนวณ metric ทั้งชุด ณ threshold ที่กำหนด"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "threshold": round(float(threshold), 3),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc_average_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        "negative_predictive_value": round(float(npv), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # base rate เพื่อเทียบกับ PR-AUC (baseline = prevalence)
        "prevalence": round(float(y_true.mean()), 4),
    }


def best_threshold(y_true, y_prob, metric: str = "f1") -> tuple[float, float]:
    """
    หา threshold ที่ให้ metric ดีที่สุดบน **OOF predictions**
    metric: 'f1' | 'recall' (เลือก recall สูงสุดโดยที่ precision >= 0.10) | 'balanced_accuracy'
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]

    if metric == "f1":
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
        idx = int(np.nanargmax(f1))
        return float(thresholds[idx]), float(f1[idx])

    if metric == "recall":
        # ต้องการ recall สูงสุด แต่ต้องไม่ใช่โมเดลที่ทายบวกทุกคน -> คุม precision ขั้นต่ำ
        feasible = np.where(precision >= 0.10)[0]
        idx = int(feasible[np.argmax(recall[feasible])]) if len(feasible) else int(np.argmax(recall))
        return float(thresholds[idx]), float(recall[idx])

    if metric == "balanced_accuracy":
        from sklearn.metrics import balanced_accuracy_score

        grid = np.unique(np.round(thresholds, 4))
        scores = [balanced_accuracy_score(y_true, (y_prob >= t).astype(int)) for t in grid]
        idx = int(np.argmax(scores))
        return float(grid[idx]), float(scores[idx])

    raise ValueError(f"ไม่รู้จัก metric: {metric}")


def oof_probabilities(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = cfg.CV_FOLDS,
) -> np.ndarray:
    """คำนวณ Out-of-Fold predicted probabilities สำหรับเลือก threshold"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


# ---------------------------------------------------------------------------
# Main comparison routine
# ---------------------------------------------------------------------------
@dataclass
class ModelComparisonResult:
    summary: pd.DataFrame
    confusion_matrices: dict[str, np.ndarray] = field(default_factory=dict)
    fitted_models: dict[str, BaseEstimator] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    oof: dict[str, np.ndarray] = field(default_factory=dict)


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_metric: str = "f1",
    model_zoo: dict[str, BaseEstimator] | None = None,
    verbose: bool = True,
) -> ModelComparisonResult:
    """
    เทรนทุกโมเดลใน zoo แล้วเปรียบเทียบกัน

    ขั้นตอน:
        1. หา OOF probabilities บน train (Stratified 5-fold)
        2. เลือก threshold ที่ดีที่สุดจาก OOF
        3. fit โมเดลเต็มบน train ทั้งชุด
        4. ประเมินบน test ด้วย threshold นั้น (และรายงานที่ 0.5 เทียบกันด้วย)
    """
    if model_zoo is None:
        model_zoo = get_model_zoo(y_train)

    rows, cms, fitted, thresholds, oof_store = [], {}, {}, {}, {}

    for name, model in model_zoo.items():
        if verbose:
            print(f"  -> กำลังเทรน: {name}")

        prob_oof = oof_probabilities(model, X_train, y_train)
        thr, thr_score = best_threshold(y_train, prob_oof, metric=threshold_metric)

        model.fit(X_train, y_train)
        prob_test = model.predict_proba(X_test)[:, 1]
        y_pred = (prob_test >= thr).astype(int)

        row = {
            "model": name,
            "threshold_rule": f"{threshold_metric} on OOF",
            **metrics_at_threshold(y_test, prob_test, thr),
            "oof_best_score": round(float(thr_score), 4),
        }
        # เทียบกับ default 0.5 เพื่อโชว์ผลของการ tune threshold
        row["f1@0.5"] = round(float(f1_score(y_test, (prob_test >= 0.5).astype(int), zero_division=0)), 4)
        row["recall@0.5"] = round(float(recall_score(y_test, (prob_test >= 0.5).astype(int), zero_division=0)), 4)

        rows.append(row)
        cms[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fitted[name] = model
        thresholds[name] = thr
        oof_store[name] = prob_oof

    summary = pd.DataFrame(rows).sort_values(
        ["recall_sensitivity", "f1"], ascending=False
    ).reset_index(drop=True)

    return ModelComparisonResult(
        summary=summary,
        confusion_matrices=cms,
        fitted_models=fitted,
        thresholds=thresholds,
        oof=oof_store,
    )


def select_best_model(
    result: ModelComparisonResult,
    primary: str = "f1",
    secondary: str = "recall_sensitivity",
) -> tuple[str, BaseEstimator, float]:
    """
    เลือกโมเดลที่ดีที่สุดตามกลยุทธ์: F1 สูงสุดก่อน แล้ว tie-break ด้วย Recall
    (ปรับได้ตามต้นทุนทางคลินิก: ถ้าเน้นไม่พลาดผู้ป่วย ให้ใช้ recall เป็น primary)
    """
    ranked = result.summary.sort_values(
        [primary, secondary], ascending=False
    ).reset_index(drop=True)
    best_name = ranked.loc[0, "model"]
    return best_name, result.fitted_models[best_name], result.thresholds[best_name]


def confusion_frame(cm: np.ndarray) -> pd.DataFrame:
    """แปลง confusion matrix เป็น DataFrame พร้อมป้ายอ่านง่าย"""
    return pd.DataFrame(
        cm,
        index=["Actual: No Stroke", "Actual: Stroke"],
        columns=["Predicted: No Stroke", "Predicted: Stroke"],
    )
