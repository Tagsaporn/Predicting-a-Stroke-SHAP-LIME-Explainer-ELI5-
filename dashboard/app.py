"""
app.py
======
Streamlit dashboard สำหรับนำเสนอผลการวิเคราะห์ Stroke Prediction + XAI

รันด้วย:
    streamlit run dashboard/app.py

หมายเหตุ: dashboard นี้ "อ่านผลลัพธ์ที่ pipeline สร้างไว้" (ไม่เทรนโมเดลใหม่)
จึงต้องรัน `python -m src.run_pipeline` ก่อนอย่างน้อยหนึ่งครั้ง
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- ให้ import src ได้เมื่อรันจาก root หรือจาก dashboard/ ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

st.set_page_config(
    page_title="Stroke Prediction + XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_STROKE = {"No Stroke": "#4C9F70", "Stroke": "#D64550"}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    path = cfg.PROCESSED_DATA_DIR / "stroke_clean.csv"
    if path.exists():
        return pd.read_csv(path)
    from src.data_loader import basic_clean, load_raw

    return basic_clean(load_raw())


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame | None:
    path = cfg.TABLES_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    path = cfg.REPORTS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def has_artifacts() -> bool:
    return (cfg.TABLES_DIR / "q3_model_comparison.csv").exists()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar(df: pd.DataFrame | None) -> str:
    st.sidebar.title("🧠 Stroke Prediction + XAI")
    st.sidebar.caption("Research Questions Q1–Q5")

    pages = [
        "🏠 ภาพรวมโปรเจกต์",
        "📊 EDA (Q2)",
        "🔬 สถิติ (Q1)",
        "🤖 โมเดล (Q3)",
        "🔍 SHAP (Q4)",
        "⚖️ SHAP vs LIME (Q5)",
        "🩺 ทดลองทำนาย",
    ]
    page = st.sidebar.radio("เลือกหน้า", pages, label_visibility="collapsed")

    st.sidebar.divider()
    if df is not None:
        st.sidebar.metric("จำนวน records", f"{len(df):,}")
        st.sidebar.metric("Stroke prevalence", f"{df[cfg.TARGET].mean() * 100:.2f}%")
    if not has_artifacts():
        st.sidebar.warning(
            "ยังไม่พบผลลัพธ์จาก pipeline\n\n"
            "รันก่อน:\n```\npython -m src.run_pipeline\n```"
        )
    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(df: pd.DataFrame) -> None:
    st.title("🧠 ทำนายโรคหลอดเลือดสมอง + อธิบายผลด้วย SHAP / LIME")
    st.markdown(
        """
        โปรเจกต์นี้ศึกษาว่า **ปัจจัยใดสัมพันธ์กับ Stroke** สร้าง **โมเดลจำแนก**
        ที่ทำงานได้ดีภายใต้ข้อมูลไม่สมดุล และ **อธิบายผลการทำนาย** ด้วยเทคนิค XAI
        """
    )

    metrics = load_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Stroke cases", f"{int(df[cfg.TARGET].sum())}")
    c3.metric("Class imbalance", f"{(df[cfg.TARGET] == 0).sum() / max((df[cfg.TARGET] == 1).sum(), 1):.1f} : 1")
    if metrics:
        c4.metric("โมเดลที่ดีที่สุด", metrics["best_model"]["name"].split(" (")[0],
                  f"F1 = {metrics['best_model']['f1']:.3f}")
    else:
        c4.metric("โมเดลที่ดีที่สุด", "—")

    st.divider()

    st.subheader("คำถามวิจัย (Research Questions)")
    rq = pd.DataFrame(
        [
            ("Q1", "ปัจจัยใดสัมพันธ์กับ stroke อย่างมีนัยสำคัญ?", "Chi-square, Mann-Whitney U, Odds Ratio"),
            ("Q2", "ข้อมูลกระจายและไม่สมดุลอย่างไร?", "EDA + Visualization"),
            ("Q3", "โมเดลใดจำแนกได้ดีที่สุดภายใต้ imbalance?", "Recall / F1 / PR-AUC / Confusion Matrix"),
            ("Q4", "ฟีเจอร์ใดขับเคลื่อน และทำไม record นี้ถูกทำนายเช่นนั้น?", "SHAP (global + local)"),
            ("Q5", "SHAP สอดคล้องกับ LIME หรือไม่?", "Jaccard, Spearman, sign agreement"),
        ],
        columns=["#", "คำถามวิจัย", "เครื่องมือ"],
    )
    st.dataframe(rq, hide_index=True, width="stretch")

    if metrics:
        st.divider()
        st.subheader("ผลลัพธ์สำคัญ (จาก pipeline)")
        c1, c2 = st.columns([1, 1])
        with c1:
            best = metrics["best_model"]
            st.plotly_chart(
                go.Figure(
                    go.Bar(
                        x=[best["recall"], best["precision"], best["f1"], best["pr_auc"], best["roc_auc"]],
                        y=["Recall", "Precision", "F1", "PR-AUC", "ROC-AUC"],
                        orientation="h",
                        marker_color=["#D64550", "#2E5A88", "#F0A202", "#4C9F70", "#9B6AA8"],
                        text=[f"{v:.3f}" for v in [best["recall"], best["precision"], best["f1"], best["pr_auc"], best["roc_auc"]]],
                        textposition="outside",
                    )
                ).update_layout(
                    title=f"ประสิทธิภาพ {best['name']} (threshold={best['threshold']:.2f})",
                    xaxis_range=[0, 1.05], height=340, margin=dict(l=10, r=10, t=50, b=10),
                ),
                width="stretch",
            )
        with c2:
            cm = np.array(best["confusion_matrix"])
            st.plotly_chart(
                go.Figure(
                    go.Heatmap(
                        z=cm, x=["Pred: No", "Pred: Stroke"], y=["True: No", "True: Stroke"],
                        text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
                    )
                ).update_layout(title="Confusion Matrix", height=340,
                                margin=dict(l=10, r=10, t=50, b=10)),
                width="stretch",
            )

        st.info(
            f"**ฟีเจอร์ที่มีอิทธิพลสูงสุด (SHAP):** "
            + ", ".join(metrics.get("top_shap_features", []))
        )


def page_eda(df: pd.DataFrame) -> None:
    st.title("📊 การสำรวจข้อมูล (EDA) — Q2")
    st.caption("ข้อมูลกระจายอย่างไร และความไม่สมดุลของคลาสส่งผลต่อการเลือก metric อย่างไร")

    tab1, tab2, tab3, tab4 = st.tabs(["ภาพรวมข้อมูล", "การกระจายตัว", "ตัวแปรเชิงกลุ่ม", "ความสัมพันธ์"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            counts = df[cfg.TARGET].value_counts().sort_index()
            labels = ["No Stroke", "Stroke"]
            fig = px.pie(
                names=labels, values=counts.values, hole=0.45,
                color=labels, color_discrete_map=COLOR_STROKE,
                title="สัดส่วนผู้ป่วย Stroke",
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("#### คุณภาพข้อมูล")
            quality = pd.DataFrame(
                {
                    "คอลัมน์": df.columns,
                    "ชนิดข้อมูล": [str(t) for t in df.dtypes],
                    "ค่าที่หาย": [int(df[c].isna().sum()) for c in df.columns],
                    "ค่าที่หาย (%)": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
                    "ค่าที่ไม่ซ้ำ": [int(df[c].nunique()) for c in df.columns],
                }
            )
            st.dataframe(quality, hide_index=True, height=420, width="stretch")

        st.markdown(
            """
            > **Insight:** คลาสบวกมีเพียง **{:.2f}%** ทำให้ *Accuracy* ไม่ใช่ตัวชี้วัดที่เชื่อถือได้
            > (ทาย "ไม่เป็น stroke" ทั้งหมดก็ได้ accuracy ≈ {:.1f}%) จึงต้องใช้
            > **Recall / F1 / PR-AUC / Confusion Matrix** แทน
            """.format(df[cfg.TARGET].mean() * 100, (1 - df[cfg.TARGET].mean()) * 100)
        )

    with tab2:
        col = st.selectbox("เลือกตัวแปรต่อเนื่อง", list(cfg.NUMERIC_FEATURES))
        tmp = df.copy()
        tmp["Stroke status"] = tmp[cfg.TARGET].map({0: "No Stroke", 1: "Stroke"})
        fig = px.histogram(
            tmp.dropna(subset=[col]), x=col, color="Stroke status", barmode="overlay",
            marginal="box", opacity=0.7, color_discrete_map=COLOR_STROKE,
            title=f"การกระจายของ {col} แยกตามสถานะ Stroke",
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")

        st.plotly_chart(
            px.box(
                tmp.dropna(subset=[col]), x="Stroke status", y=col, color="Stroke status",
                color_discrete_map=COLOR_STROKE, points="outliers",
                title=f"Box plot ของ {col}",
            ).update_layout(showlegend=False, height=400),
            width="stretch",
        )

    with tab3:
        col = st.selectbox("เลือกตัวแปรเชิงกลุ่ม", list(cfg.CATEGORICAL_FEATURES))
        rate = (
            df.groupby(col)[cfg.TARGET].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "stroke_rate"})
        )
        rate["stroke_rate_%"] = rate["stroke_rate"] * 100
        fig = px.bar(
            rate.sort_values("stroke_rate_%", ascending=False),
            x=col, y="stroke_rate_%", text="count",
            color="stroke_rate_%", color_continuous_scale="Reds",
            title=f"อัตราการเกิด Stroke (%) แยกตาม {col}",
        )
        fig.add_hline(y=df[cfg.TARGET].mean() * 100, line_dash="dash",
                      annotation_text="overall rate", line_color="#2E5A88")
        fig.update_layout(height=430, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(rate, hide_index=True, width="stretch")

    with tab4:
        num_cols = list(cfg.NUMERIC_FEATURES) + list(cfg.BINARY_FEATURES) + [cfg.TARGET]
        corr = df[num_cols].corr(method="spearman")
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, title="Spearman correlation matrix", aspect="auto",
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")

        sub = df.dropna(subset=["bmi", "avg_glucose_level"]).copy()
        sub["Stroke status"] = sub[cfg.TARGET].map({0: "No Stroke", 1: "Stroke"})

        fig2 = px.scatter(
            sub,
            x="age", y="avg_glucose_level", color="Stroke status",
            color_discrete_map=COLOR_STROKE, opacity=0.6, size="bmi",
            title="Age × Glucose × BMI แยกตามสถานะ Stroke",
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, width="stretch")


def page_stats() -> None:
    st.title("🔬 การวิเคราะห์ทางสถิติ — Q1")
    st.caption("ทดสอบนัยสำคัญพร้อม effect size และปรับ p-value ด้วย Benjamini-Hochberg (FDR)")

    chi = load_table("q1_chi_square.csv")
    mw = load_table("q1_mann_whitney.csv")
    odds = load_table("q1_odds_ratios.csv")

    if chi is None and mw is None:
        st.warning("ไม่พบตารางผลลัพธ์ — รัน `python -m src.run_pipeline` ก่อน")
        return

    tab1, tab2, tab3 = st.tabs(["ตัวแปรเชิงกลุ่ม (Chi-square)", "ตัวแปรต่อเนื่อง (Mann-Whitney)", "Odds Ratio"])

    with tab1:
        st.markdown(
            "**Chi-square test of independence** ตรวจว่าการกระจายของ stroke ต่างกันตามกลุ่มหรือไม่ "
            "พร้อม **Cramér's V** เป็นขนาดความสัมพันธ์"
        )
        if chi is not None:
            show = chi[["feature", "chi2", "p_value", "p_adjusted_bh", "cramers_v",
                        "effect_size", "significant_0.05", "stroke_rate_by_group_%"]]
            st.dataframe(
                show.style.apply(
                    lambda r: ["background-color: #e8f5e9" if r["significant_0.05"] else "background-color: #fdecea"] * len(r),
                    axis=1,
                ),
                hide_index=True, width="stretch",
            )
            sig = chi[chi["significant_0.05"]]["feature"].tolist()
            st.success(f"มีนัยสำคัญทางสถิติ (FDR < 0.05): **{', '.join(sig) if sig else 'ไม่มี'}**")

    with tab2:
        st.markdown(
            "**Mann-Whitney U test** เปรียบเทียบมัธยฐานระหว่าง 2 กลุ่ม "
            "(เลือกใช้เพราะข้อมูลไม่เป็น normal — ดูคอลัมน์ `shapiro_p_*` ประกอบ)"
        )
        if mw is not None:
            st.dataframe(
                mw[["feature", "median_no_stroke", "median_stroke", "u_statistic", "p_value",
                    "p_adjusted_bh", "rank_biserial", "normality_rejected", "significant_0.05"]],
                hide_index=True, width="stretch",
            )
            st.plotly_chart(
                px.bar(
                    mw, x="feature", y="rank_biserial", color="significant_0.05",
                    color_discrete_map={True: "#D64550", False: "#B0B0B0"},
                    title="Effect size (rank-biserial correlation) — ยิ่ง |ค่า| สูง ยิ่งต่างกันชัด",
                ).update_layout(height=400),
                width="stretch",
            )

    with tab3:
        st.markdown("**Odds Ratio** ของปัจจัยเสี่ยงไบนารี (ค่ามากกว่า 1 = เพิ่มความเสี่ยง)")
        if odds is not None:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=odds["odds_ratio"], y=odds["feature"], mode="markers",
                    marker=dict(size=14, color="#D64550"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=odds["ci_95_high"] - odds["odds_ratio"],
                        arrayminus=odds["odds_ratio"] - odds["ci_95_low"],
                    ),
                )
            )
            fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="OR = 1 (no effect)")
            fig.update_layout(title="Odds Ratio พร้อมช่วงเชื่อมั่น 95%", height=350,
                              xaxis_title="Odds Ratio", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(odds, hide_index=True, width="stretch")


def page_models() -> None:
    st.title("🤖 การเปรียบเทียบโมเดล — Q3")
    summary = load_table("q3_model_comparison.csv")
    if summary is None:
        st.warning("ไม่พบผลลัพธ์ — รัน `python -m src.run_pipeline` ก่อน")
        return

    st.markdown(
        "ทุกโมเดลใช้ preprocessing เดียวกัน และเลือก threshold จาก **Out-of-Fold predictions** "
        "เพื่อไม่ให้ test set รั่วไหล"
    )

    metric_labels = {
        "recall_sensitivity": "Recall",
        "precision": "Precision",
        "f1": "F1",
        "pr_auc_average_precision": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "specificity": "Specificity",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Acc.",
    }
    chosen = st.multiselect("เลือก metric ที่จะแสดง", list(metric_labels),
                            default=["recall_sensitivity", "precision", "f1", "pr_auc_average_precision"])

    if chosen:
        plot_df = summary.melt(id_vars="model", value_vars=chosen, var_name="metric", value_name="score")
        plot_df["metric"] = plot_df["metric"].map(metric_labels)
        fig = px.bar(plot_df, x="model", y="score", color="metric", barmode="group",
                     title="เปรียบเทียบประสิทธิภาพโมเดล", text_auto=".3f")
        fig.update_layout(height=460, xaxis_tickangle=-20, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")

    st.subheader("ตารางผลลัพธ์เต็ม")
    st.dataframe(summary, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Confusion Matrix")
    picked = st.selectbox("เลือกโมเดล", summary["model"].tolist())

    row = summary[summary["model"] == picked].iloc[0]
    cm = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])

    st.plotly_chart(
        go.Figure(go.Heatmap(z=cm, x=["Pred: No", "Pred: Stroke"], y=["True: No", "True: Stroke"],
                             text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        .update_layout(height=380, title=f"{picked}  (threshold = {row['threshold']:.3f})"),
        width="stretch",
    )
    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positive", tp, help="ผู้ป่วยที่ตรวจพบ — ยิ่งมากยิ่งดี")
    c2.metric("False Negative", fn, help="ผู้ป่วยที่ถูกมองข้าม — ต้นทุนสูงสุดทางคลินิก")
    c3.metric("False Positive", fp, help="เตือนเกิน — ต้นทุนต่ำกว่า")
    c4.metric("True Negative", tn)
    st.caption(
        f"Recall = {row['recall_sensitivity']:.3f} | Precision = {row['precision']:.3f} | "
        f"F1 = {row['f1']:.3f} | PR-AUC = {row['pr_auc_average_precision']:.3f} | "
        f"ROC-AUC = {row['roc_auc']:.3f}"
    )

    st.divider()
    fig_path = cfg.FIGURES_DIR / "model" / "12_pr_roc_curves.png"
    if fig_path.exists():
        st.subheader("Precision-Recall และ ROC Curve")
        st.image(str(fig_path))


def page_shap() -> None:
    st.title("🔍 อธิบายผลด้วย SHAP — Q4")
    importance = load_table("q4_shap_global_importance.csv")
    local = load_table("q4_shap_local_cases.csv")

    if importance is None:
        st.warning("ไม่พบผลลัพธ์ SHAP — รัน `python -m src.run_pipeline` ก่อน")
        return

    st.markdown(
        "**SHAP** แตกการทำนายออกเป็นผลบวกของ contributions ของแต่ละฟีเจอร์ "
        "ค่า + ดันไปทาง *เป็น stroke* ค่า − ดันไปทาง *ไม่เป็น stroke*"
    )

    top_n = st.slider("แสดงกี่อันดับแรก", 3, max(4, len(importance)), min(12, len(importance)))
    imp = importance.head(top_n).iloc[::-1]
    fig = px.bar(
        imp, x="mean_abs_shap", y="label_th", orientation="h",
        color="mean_shap", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        title=f"Global importance — ค่าเฉลี่ย |SHAP| (Top {top_n})",
        labels={"mean_abs_shap": "mean(|SHAP|)", "label_th": "", "mean_shap": "ทิศทางเฉลี่ย"},
    )
    fig.update_layout(height=max(380, 30 * top_n))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns([1, 1])
    with col1:
        p = cfg.FIGURES_DIR / "xai" / "21_shap_beeswarm.png"
        if p.exists():
            st.image(str(p), caption="SHAP beeswarm — การกระจายของอิทธิพลต่อ record แต่ละราย")
    with col2:
        st.dataframe(importance, hide_index=True, height=430, width="stretch")

    st.divider()
    st.subheader("Local explanation — ทำไม record นี้จึงถูกทำนายเช่นนั้น")
    if local is not None:
        case = st.selectbox("เลือกเคส", local["case"].unique().tolist())
        sub = local[local["case"] == case]
        rec = st.selectbox("เลือก record", sub["record_index"].unique().tolist())
        rec_df = sub[sub["record_index"] == rec].sort_values("shap_value")
        fig = px.bar(
            rec_df, x="shap_value", y="feature", orientation="h",
            color="direction", color_discrete_map={
                "ดันให้เป็น stroke": "#D64550",
                "ดันให้ไม่เป็น stroke": "#4C9F70",
            },
            title=f"SHAP contribution — record #{rec} ({case})",
            hover_data=["raw_value", "predicted_prob", "actual"],
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(rec_df[["feature", "raw_value", "shap_value", "direction",
                             "predicted_prob", "actual"]], hide_index=True, width="stretch")

    wf = sorted((cfg.FIGURES_DIR / "xai").glob("22_shap_waterfall_*.png"))
    if wf:
        st.subheader("SHAP Waterfall")
        cols = st.columns(min(3, len(wf)))
        for i, p in enumerate(wf):
            cols[i % len(cols)].image(str(p), caption=p.stem.replace("22_shap_waterfall_", ""))


def page_shap_lime() -> None:
    st.title("⚖️ SHAP vs LIME — Q5")
    comp = load_table("q5_shap_vs_lime.csv")
    lime = load_table("q4_lime_local.csv")

    if comp is None:
        st.warning("ไม่พบผลลัพธ์การเปรียบเทียบ — รัน pipeline โดยไม่ใส่ `--skip-lime`")
        return

    st.markdown(
        "SHAP และ LIME ต่างก็อธิบายโมเดลเดียวกัน แต่ใช้ทฤษฎีต่างกัน "
        "จึงควรถามว่า **คำอธิบายทั้งสองวิธีสอดคล้องกันหรือไม่** ถ้าไม่สอดคล้อง "
        "แปลว่าเราไม่ควรเชื่อคำอธิบายนั้นมากเกินไป"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("ค่าเฉลี่ย Jaccard (top-8)", f"{comp['jaccard_topk'].mean():.3f}")
    c2.metric("ค่าเฉลี่ย Spearman ρ", f"{comp['spearman_rho'].mean():.3f}")
    c3.metric("ทิศทางตรงกัน", f"{comp['sign_agreement'].mean() * 100:.1f}%")

    st.dataframe(comp, hide_index=True, width="stretch")

    fig_path = cfg.FIGURES_DIR / "xai" / "31_shap_vs_lime.png"
    if fig_path.exists():
        st.image(str(fig_path))

    if lime is not None:
        st.divider()
        st.subheader("รายละเอียด LIME ต่อ record")
        rec = st.selectbox("เลือก record", lime["record_index"].unique().tolist())
        sub = lime[lime["record_index"] == rec].sort_values("lime_weight")
        st.plotly_chart(
            px.bar(
                sub, x="lime_weight", y="feature_condition", orientation="h",
                color="direction",
                color_discrete_map={"ดันให้เป็น stroke": "#D64550", "ดันให้ไม่เป็น stroke": "#4C9F70"},
                title=f"LIME weights — record #{rec}",
            ).update_layout(height=420),
            width="stretch",
        )

        st.markdown("#### เปรียบเทียบฟีเจอร์ที่ทั้งสองวิธีจัดอันดับ")
        st.dataframe(
            comp[["record_index", "overlap_count", "jaccard_topk", "shap_top", "lime_top"]],
            hide_index=True, width="stretch",
        )


def page_predict(df: pd.DataFrame) -> None:
    st.title("🩺 ทดลองทำนายรายบุคคล")
    model_path = cfg.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        st.warning("ยังไม่พบโมเดล — รัน `python -m src.run_pipeline` ก่อน")
        return

    import joblib

    bundle = joblib.load(model_path)
    model, threshold, name = bundle["model"], bundle["threshold"], bundle["name"]
    st.caption(f"โมเดล: **{name}** | threshold ที่ tune แล้ว = **{threshold:.3f}**")

    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("อายุ (ปี)", 0.0, 100.0, 55.0, 1.0)
            gender = st.selectbox("เพศ", ["Male", "Female"])
            ever_married = st.selectbox("สถานะสมรส", ["Yes", "No"])
        with c2:
            glucose = st.slider("ระดับน้ำตาลในเลือดเฉลี่ย", 50.0, 300.0, 100.0, 1.0)
            bmi = st.slider("BMI", 10.0, 60.0, 28.0, 0.5)
            work_type = st.selectbox("ประเภทงาน", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        with c3:
            hypertension = st.selectbox("ความดันโลหิตสูง", [0, 1], format_func=lambda v: "มี" if v else "ไม่มี")
            heart_disease = st.selectbox("โรคหัวใจ", [0, 1], format_func=lambda v: "มี" if v else "ไม่มี")
            residence = st.selectbox("ที่อยู่อาศัย", ["Urban", "Rural"])
            smoking = st.selectbox("สถานะการสูบบุหรี่", ["never smoked", "formerly smoked", "smokes", "Unknown"])

        submitted = st.form_submit_button("ทำนาย", width="stretch")

    if submitted:
        record = pd.DataFrame(
            [{
                "gender": gender, "age": age, "hypertension": hypertension,
                "heart_disease": heart_disease, "ever_married": ever_married,
                "work_type": work_type, "Residence_type": residence,
                "avg_glucose_level": glucose, "bmi": bmi,
                "smoking_status": smoking,
            }]
        )[list(cfg.ALL_FEATURES)]

        prob = float(model.predict_proba(record)[:, 1][0])
        pred = int(prob >= threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("ความน่าจะเป็น (stroke)", f"{prob * 100:.1f}%")
        c2.metric("ผลทำนาย", "เป็น Stroke" if pred else "ไม่เป็น Stroke")
        c3.metric("เกณฑ์ (threshold)", f"{threshold:.3f}")

        st.progress(min(prob, 1.0))
        if pred:
            st.error("โมเดลประเมินว่ามีความเสี่ยง — ควรปรึกษาแพทย์ (นี่ไม่ใช่การวินิจฉัย)")
        else:
            st.success("โมเดลประเมินว่ามีความเสี่ยงต่ำ (นี่ไม่ใช่การวินิจฉัย)")

        st.divider()
        st.subheader("ทำไมโมเดลจึงทำนายเช่นนี้ (SHAP)")
        try:
            from src import xai as xai_mod
            from src.data_loader import split_features_target
            from src.preprocessing import group_onehot_columns

            X_all, _ = split_features_target(df)
            bg = X_all.sample(min(200, len(X_all)), random_state=cfg.RANDOM_STATE)
            Xtr, sv, base, _ = xai_mod.compute_shap(model, record, bg)
            contrib = pd.Series(sv[0], index=Xtr.columns)
            contrib_base = group_onehot_columns(contrib.to_frame().T).T.iloc[:, 0].sort_values()
            contrib_base = contrib_base.reindex(contrib_base.abs().sort_values(ascending=False).index)

            plot_df = contrib_base.reset_index()
            plot_df.columns = ["feature", "shap_value"]
            plot_df["label_th"] = plot_df["feature"].map(cfg.FEATURE_LABELS_TH).fillna(plot_df["feature"])
            plot_df["direction"] = np.where(plot_df["shap_value"] > 0,
                                            "ดันให้เป็น stroke", "ดันให้ไม่เป็น stroke")
            st.plotly_chart(
                px.bar(
                    plot_df.sort_values("shap_value"), x="shap_value", y="label_th", orientation="h",
                    color="direction",
                    color_discrete_map={"ดันให้เป็น stroke": "#D64550",
                                        "ดันให้ไม่เป็น stroke": "#4C9F70"},
                    title="SHAP contribution ของผู้ป่วยรายนี้",
                ).update_layout(height=420, yaxis_title=""),
                width="stretch",
            )
        except Exception as exc:  # noqa: BLE001
            st.info(f"ไม่สามารถคำนวณ SHAP สำหรับรายบุคคลได้: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        df = load_clean_data()
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ข้อมูล — ตรวจสอบว่า `data/raw/healthcare-dataset-stroke-data.csv` มีอยู่")
        st.stop()

    page = sidebar(df)

    if page.startswith("🏠"):
        page_overview(df)
    elif page.startswith("📊"):
        page_eda(df)
    elif page.startswith("🔬"):
        page_stats()
    elif page.startswith("🤖"):
        page_models()
    elif page.startswith("🔍"):
        page_shap()
    elif page.startswith("⚖️"):
        page_shap_lime()
    else:
        page_predict(df)


if __name__ == "__main__":
    main()
