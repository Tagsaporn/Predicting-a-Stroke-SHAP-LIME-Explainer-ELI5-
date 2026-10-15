"""
visualization.py
================
ฟังก์ชันสร้างกราฟสำหรับ EDA / Model evaluation / XAI
ทุกฟังก์ชันบันทึกไฟล์ลง ``reports/figures`` และคืน ``Figure`` กลับไป
(เพื่อให้ฝังใน notebook หรือส่งต่อไปใช้ใน dashboard ได้)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ให้รันแบบ headless / pipeline ได้

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config as cfg  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = cfg.FIGURE_DPI
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False


def setup_thai_font() -> str:
    """พยายามเลือกฟอนต์ที่รองรับภาษาไทย ถ้าไม่มีก็ใช้ค่าเริ่มต้น"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Leelawadee UI", "Tahoma", "Sarabun", "TH Sarabun New", "Noto Sans Thai"):
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return plt.rcParams["font.family"][0] if isinstance(plt.rcParams["font.family"], list) else "default"


FALLBACK_FONT = setup_thai_font()


def _save(fig, filename: str, subdir: str = "") -> Path:
    cfg.ensure_dirs()
    out_dir = cfg.FIGURES_DIR / subdir if subdir else cfg.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# EDA (Q2)
# ---------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, target: str = cfg.TARGET):
    """สัดส่วนคลาส target + ป้ายกำกับความไม่สมดุล"""
    counts = df[target].value_counts().sort_index()
    pct = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = axes[0].bar(["No Stroke (0)", "Stroke (1)"], counts.values,
                       color=[cfg.PALETTE_STROKE[0], cfg.PALETTE_STROKE[1]])
    axes[0].set_title("Class Distribution (Counts)")
    axes[0].set_ylabel("Number of records")
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")

    axes[1].pie(
        pct.values,
        labels=[f"No Stroke\n{pct.iloc[0]:.2f}%", f"Stroke\n{pct.iloc[1]:.2f}%"],
        colors=[cfg.PALETTE_STROKE[0], cfg.PALETTE_STROKE[1]],
        autopct="%1.2f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class Imbalance")
    fig.suptitle("Target Variable: stroke", fontsize=14, fontweight="bold")
    return _save(fig, "01_target_distribution.png", "eda")


def plot_missing_values(df: pd.DataFrame):
    """แผนภูมิแท่งเปอร์เซ็นต์ค่าที่หายรายคอลัมน์"""
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(miss) + 1.5)))
    if miss.empty:
        ax.text(0.5, 0.5, "ไม่พบค่าที่หายเลย", ha="center", va="center")
        ax.axis("off")
    else:
        sns.barplot(x=miss.values, y=miss.index, ax=ax, color=cfg.COLORS["accent"])
        ax.set_xlabel("% missing")
        ax.set_title("Missing Values by Column")
        for i, v in enumerate(miss.values):
            ax.text(v, i, f" {v:.2f}%", va="center")
    return _save(fig, "02_missing_values.png", "eda")


def plot_numeric_by_target(df: pd.DataFrame, cols=None, target: str = cfg.TARGET):
    """Histogram + KDE ของตัวแปรต่อเนื่อง แยกตามกลุ่ม stroke"""
    cols = cols or list(cfg.NUMERIC_FEATURES)
    fig, axes = plt.subplots(1, len(cols), figsize=(5.2 * len(cols), 4))
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, cols):
        sns.histplot(
            data=df, x=col, hue=target, bins=35, kde=True,
            palette=cfg.PALETTE_STROKE, alpha=0.55, ax=ax, common_norm=False,
        )
        ax.set_title(f"{col} by stroke")
        ax.set_xlabel(cfg.FEATURE_LABELS_TH.get(col, col))
        ax.legend(title="stroke", labels=["No Stroke", "Stroke"])
    fig.suptitle("Distribution of Continuous Features by Stroke Status", fontweight="bold")
    return _save(fig, "03_numeric_distributions.png", "eda")


def plot_categorical_vs_target(df: pd.DataFrame, cols=None, target: str = cfg.TARGET):
    """แผนภูมิแท่ง: stroke rate (%) ในแต่ละกลุ่มของตัวแปรเชิงกลุ่ม"""
    cols = cols or list(cfg.CATEGORICAL_FEATURES)
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        rate = (
            df.groupby(col)[target].agg(["mean", "count"])
            .assign(stroke_rate=lambda d: d["mean"] * 100)
            .sort_values("stroke_rate", ascending=False)
        )
        sns.barplot(x=rate.index, y=rate["stroke_rate"], ax=ax,
                    hue=rate.index, palette="rocket_r", legend=False)
        overall = df[target].mean() * 100
        ax.axhline(overall, color=cfg.COLORS["primary"], ls="--", lw=1.2,
                   label=f"overall {overall:.2f}%")
        ax.set_title(f"Stroke rate by {col}")
        ax.set_ylabel("Stroke rate (%)")
        ax.tick_params(axis="x", rotation=25)
        for i, (v, c) in enumerate(zip(rate["stroke_rate"], rate["count"])):
            ax.text(i, v, f"n={c}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Stroke Rate Across Categorical Features", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return _save(fig, "04_categorical_vs_target.png", "eda")


def plot_correlation_heatmap(df: pd.DataFrame, target: str = cfg.TARGET):
    """Heatmap สหสัมพันธ์ Spearman ของตัวแปรตัวเลขทั้งหมด"""
    num_df = df[[c for c in list(cfg.NUMERIC_FEATURES) + list(cfg.BINARY_FEATURES) + [target]]]
    corr = num_df.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Spearman Correlation Matrix")
    return _save(fig, "05_correlation_heatmap.png", "eda")


def plot_age_glucose_bmi_scatter(df: pd.DataFrame, target: str = cfg.TARGET):
    """Scatter 2 มิติ (age × glucose, age × bmi) ไล่สีตามกลุ่ม stroke"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for grp, sub in df.groupby(target):
        label = "Stroke" if grp == 1 else "No Stroke"
        axes[0].scatter(sub["age"], sub["avg_glucose_level"], s=14 if grp else 8,
                        alpha=0.55 if grp else 0.25, c=cfg.PALETTE_STROKE[grp], label=label)
        axes[1].scatter(sub["age"], sub["bmi"], s=14 if grp else 8,
                        alpha=0.55 if grp else 0.25, c=cfg.PALETTE_STROKE[grp], label=label)

    axes[0].set(xlabel="Age (years)", ylabel="Average glucose level", title="Age vs Glucose")
    axes[1].set(xlabel="Age (years)", ylabel="BMI", title="Age vs BMI")

    # stroke rate ตามช่วงอายุ
    age_bins = pd.cut(df["age"], bins=np.arange(0, 91, 10))
    rate = df.groupby(age_bins, observed=True)[target].agg(["mean", "count"])
    axes[2].bar([str(i) for i in rate.index], rate["mean"] * 100, color=cfg.COLORS["secondary"])
    axes[2].set(xlabel="Age band", ylabel="Stroke rate (%)", title="Stroke Rate by Age Band")
    axes[2].tick_params(axis="x", rotation=45)
    for i, (v, c) in enumerate(zip(rate["mean"] * 100, rate["count"])):
        axes[2].text(i, v, f"n={c}", ha="center", va="bottom", fontsize=8)

    for ax in axes[:2]:
        ax.legend()
    fig.suptitle("Bivariate Exploration of Stroke Risk Factors", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "06_scatter_age_glucose_bmi.png", "eda")


def plot_vascular_risk(df: pd.DataFrame, target: str = cfg.TARGET):
    """Stroke rate เมื่อมีจำนวนปัจจัยเสี่ยงหัวใจและหลอดเลือด (HT + HD) ต่างกัน"""
    tmp = df.copy()
    tmp["vascular_risk"] = tmp["hypertension"].astype(int) + tmp["heart_disease"].astype(int)
    rate = tmp.groupby("vascular_risk")[target].agg(["mean", "count"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.barplot(x=rate.index.astype(str), y=rate["mean"] * 100, ax=axes[0],
                hue=rate.index.astype(str), palette="rocket_r", legend=False)
    for i, (v, c) in enumerate(zip(rate["mean"] * 100, rate["count"])):
        axes[0].text(i, v, f"n={c}", ha="center", va="bottom", fontsize=9)
    axes[0].set(xlabel="Number of vascular risk factors (HT + HD)",
                ylabel="Stroke rate (%)", title="Stroke Rate vs Vascular Risk Burden")

    sns.heatmap(pd.crosstab(tmp["hypertension"], tmp["heart_disease"],
                            values=tmp[target], aggfunc="mean") * 100,
                annot=True, fmt=".2f", cmap="Reds", ax=axes[1], cbar_kws={"label": "Stroke rate (%)"})
    axes[1].set(title="Stroke Rate by HT x HD (%)", xlabel="Heart disease", ylabel="Hypertension")
    fig.tight_layout()
    return _save(fig, "07_vascular_risk.png", "eda")


# ---------------------------------------------------------------------------
# Model evaluation (Q3)
# ---------------------------------------------------------------------------
def plot_model_comparison(summary: pd.DataFrame):
    """เปรียบเทียบโมเดลด้วย Recall / Precision / F1 / PR-AUC"""
    metrics = ["recall_sensitivity", "precision", "f1", "pr_auc_average_precision", "roc_auc"]
    plot_df = summary.set_index("model")[metrics].sort_values("f1")
    short = {m: m.replace("_average_precision", "").replace("recall_sensitivity", "recall") for m in metrics}

    fig, ax = plt.subplots(figsize=(11, max(4, 0.55 * len(plot_df) + 2)))
    plot_df.rename(columns=short).plot(kind="barh", ax=ax, width=0.82,
                                       color=["#D64550", "#2E5A88", "#F0A202", "#4C9F70", "#9B6AA8"])
    ax.set_xlabel("Score")
    ax.set_title("Model Comparison (threshold tuned on OOF)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1.02)
    fig.tight_layout()
    return _save(fig, "10_model_comparison.png", "model")


def plot_confusion_matrices(confusion_matrices: dict, cols: int = 3):
    """วาด confusion matrix ของทุกโมเดลพร้อมตัวเลข TN/FP/FN/TP"""
    names = list(confusion_matrices)
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        cm = confusion_matrices[name]
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred: No", "Pred: Stroke"],
                    yticklabels=["True: No", "True: Stroke"])
        tn, fp, fn, tp = cm.ravel()
        ax.set_title(f"{name}\nRecall={tp/(tp+fn):.3f}  Precision={tp/(tp+fp) if (tp+fp) else 0:.3f}", fontsize=10)

    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices at OOF-tuned Threshold", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "11_confusion_matrices.png", "model")


def plot_pr_roc_curves(curves: dict[str, dict], filename: str = "12_pr_roc_curves.png"):
    """
    วาด PR curve และ ROC curve ของหลายโมเดล
    curves = {model_name: {"y_true":..., "y_prob":...}}
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    palette = sns.color_palette("husl", len(curves))

    for (name, data), color in zip(curves.items(), palette):
        y_true, y_prob = np.asarray(data["y_true"]), np.asarray(data["y_prob"])
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[0].plot(r, p, color=color, lw=1.8, label=f"{name} (AP={ap:.3f})")

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[1].plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")

    base = np.mean(list(curves.values())[0]["y_true"])
    axes[0].axhline(base, ls=":", color="gray", label=f"baseline prevalence={base:.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve (focus of imbalanced learning)")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    axes[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    return _save(fig, filename, "model")


# ---------------------------------------------------------------------------
# XAI plots (Q4 / Q5)
# ---------------------------------------------------------------------------
def plot_shap_bar(importance: pd.DataFrame, top_n: int = 12, filename: str = "20_shap_bar.png"):
    """แผนภูมิแท่งความสำคัญรวม (mean |SHAP|)"""
    top = importance.head(top_n).iloc[::-1]
    labels = top["label_th"].where(top["label_th"].notna(), top["feature"])

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(top) + 1.8)))
    colors = [cfg.COLORS["secondary"] if v > 0 else cfg.COLORS["positive"] for v in top["mean_shap"]]
    ax.barh(labels, top["mean_abs_shap"], color=colors, alpha=0.85)
    ax.set_xlabel("mean(|SHAP value|)  ->  average impact on model output")
    ax.set_title(f"Global Feature Importance (Top {top_n})")
    for i, (v, pct) in enumerate(zip(top["mean_abs_shap"], top["importance_pct"])):
        ax.text(v, i, f" {v:.4f} ({pct:.1f}%)", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, filename, "xai")


def plot_shap_beeswarm(shap_values, X_transformed, feature_labels=None, top_n: int = 12,
                       filename: str = "21_shap_beeswarm.png"):
    """SHAP beeswarm (summary plot) — เห็นทั้งขนาดและทิศทางพร้อมค่าฟีเจอร์"""
    import shap

    labels = [feature_labels.get(c, c) if feature_labels else c for c in X_transformed.columns]
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=labels,
        max_display=top_n,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(10, max(4.5, 0.45 * top_n + 2))
    plt.title("SHAP Beeswarm — direction & magnitude of each feature", fontweight="bold")
    plt.tight_layout()
    return _save(fig, filename, "xai")


def plot_shap_waterfall(explanation, max_display: int = 12, filename: str = "22_shap_waterfall_case.png"):
    """SHAP waterfall สำหรับ 1 record (local explanation)"""
    import shap

    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, max(4.5, 0.42 * max_display + 2))
    plt.tight_layout()
    return _save(fig, filename, "xai")


def plot_lime_explanation(lime_result, filename: str | None = None, top_n: int = 10):
    """แปลง LIME explanation เป็นกราฟแท่งที่อ่านง่าย"""
    top = lime_result.top_features.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(top) + 2)))
    colors = [cfg.COLORS["secondary"] if w > 0 else cfg.COLORS["positive"] for w in top["lime_weight"]]
    ax.barh(top["feature_condition"], top["lime_weight"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LIME weight (local surrogate)")
    ax.set_title(f"LIME Local Explanation — record #{lime_result.record_index}")
    fig.tight_layout()
    fname = filename or f"30_lime_case_{lime_result.record_index}.png"
    return _save(fig, fname, "xai")


def plot_shap_lime_comparison(comparison: pd.DataFrame, filename: str = "31_shap_vs_lime.png"):
    """เปรียบเทียบระดับความสอดคล้อง SHAP vs LIME ราย record"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    x = np.arange(len(comparison))
    axes[0].bar(x, comparison["overlap_count"], color=cfg.COLORS["primary"])
    axes[0].axhline(8, ls="--", color="gray", label="max = top-k (k=8)")
    axes[0].set(xlabel="Record", ylabel="Overlapping features",
                title="Top-k Feature Overlap (SHAP & LIME)")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, comparison["jaccard_topk"], color=cfg.COLORS["accent"])
    axes[1].set(xlabel="Record", ylabel="Jaccard index", title="Jaccard Similarity of Top-k Sets")
    axes[1].set_ylim(0, 1)

    valid = comparison["spearman_rho"].dropna()
    axes[2].hist(valid, bins=10, color=cfg.COLORS["positive"], edgecolor="white")
    axes[2].axvline(valid.mean() if len(valid) else 0, color=cfg.COLORS["secondary"], ls="--",
                    label=f"mean rho = {valid.mean():.2f}" if len(valid) else "n/a")
    axes[2].set(xlabel="Spearman rho", ylabel="Frequency", title="Agreement of Attribution Magnitude")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.suptitle("Q5: Do SHAP and LIME agree?", fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename, "xai")
