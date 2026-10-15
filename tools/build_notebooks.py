"""
build_notebooks.py
==================
สร้าง notebook ทั้ง 5 เล่ม (01 → 05) ด้วย nbformat

รันครั้งเดียว:
    python tools/build_notebooks.py

ทำไมต้องมี script นี้: notebook เป็นไฟล์ JSON ที่แก้ด้วยมือยาก
การ generate จากโค้ดทำให้ปรับปรุงเนื้อหาแล้วสร้างใหม่ได้เสมอ
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
import sys
from pathlib import Path

# ให้ import src ได้ไม่ว่าจะเปิด notebook จากโฟลเดอร์ไหน
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
print("Project root:", ROOT)
"""

SHOW_FIG = """\
def show_figure(relative_path: str, width: int = 950):
    \"\"\"แสดงรูปที่ pipeline สร้างไว้ใน reports/figures\"\"\"
    path = ROOT / "reports" / "figures" / relative_path
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print(f"[ไม่พบรูป] {path} — รัน `python -m src.run_pipeline` ก่อน")
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(path: Path, cells: list) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("เขียนแล้ว:", path)


# ===========================================================================
# 01 — EDA
# ===========================================================================
def build_01() -> list:
    return [
        md("""
# 01 · Exploratory Data Analysis (Q2)

**คำถามวิจัย Q2** — ข้อมูลมีการกระจายอย่างไร และความไม่สมดุลของคลาสส่งผลต่อการเลือก metric อย่างไร?

โปรเจกต์: การทำนายโรคหลอดเลือดสมอง (Stroke) + อธิบายผลด้วย SHAP / LIME
ชุดข้อมูล: *Stroke Prediction Dataset* — 5,110 records, 12 คอลัมน์

> ในเล่มนี้เรายัง **ไม่แก้ไขข้อมูล** — ดูตามสภาพจริงก่อน แล้วค่อยทำความสะอาด
"""),
        md("## 1. เตรียมสภาพแวดล้อม"),
        code(SETUP),
        code(SHOW_FIG),
        md("## 2. โหลดข้อมูล"),
        code("""
from src import config as cfg
from src import data_loader as dl

raw = dl.load_raw()
print("ขนาดข้อมูลดิบ:", raw.shape)
raw.head()
"""),
        md("## 3. คุณภาพข้อมูล (Data Quality)"),
        code("""
quality = dl.data_quality_report(raw)
quality
"""),
        code("""
clean = dl.basic_clean(raw)
print("ก่อนทำความสะอาด:", raw.shape[0], "แถว")
print("หลังทำความสะอาด:", clean.shape[0], "แถว")
print()
print("แถวที่ถูกลบ:", raw.shape[0] - clean.shape[0], "(gender = 'Other' มีเพียง 1 แถว)")
print("คอลัมน์ที่ถูกลบ: id (ไม่มีข้อมูลเชิงทำนาย)")
print()
print("ค่าที่หายคงเหลือ (จะถูกเติมใน pipeline):")
print(clean.isna().sum()[clean.isna().sum() > 0])
"""),
        md("""
### 🧭 ข้อสังเกตจากคุณภาพข้อมูล

| ประเด็น | รายละเอียด | การจัดการ |
|---|---|---|
| `id` | ตัวระบุ ไม่มีความหมายเชิงทำนาย | ตัดออก |
| `gender = 'Other'` | มีเพียง 1 แถว | ตัดออก (ไม่มีอำนาจทางสถิติ) |
| `bmi` ขาด 3.93% | 201 records | เติมด้วย median **ภายใน** sklearn Pipeline (กัน leakage) |
| `smoking_status = 'Unknown'` | 30.22% | คงไว้เป็นอีก level หนึ่ง (การเดาค่าอาจสร้างอคติ) |
"""),
        md("## 4. ความไม่สมดุลของคลาส — หัวใจของ Q2"),
        code("""
counts = clean[cfg.TARGET].value_counts().sort_index()
summary = pd.DataFrame({
    "จำนวน": counts.values,
    "สัดส่วน (%)": (counts / counts.sum() * 100).round(3).values,
}, index=["No Stroke (0)", "Stroke (1)"])
summary.loc["รวม"] = [counts.sum(), 100.0]

imbalance = counts[0] / counts[1]
baseline_acc = counts[0] / counts.sum() * 100

display(summary)
print(f"อัตราส่วน class imbalance = {imbalance:.1f} : 1")
print(f"ถ้าทำนาย 'ไม่เป็น stroke' ทุกคน จะได้ Accuracy = {baseline_acc:.2f}%")
print("=> Accuracy ใช้ไม่ได้กับข้อมูลชุดนี้ ต้องใช้ Recall / F1 / PR-AUC / Confusion Matrix")
"""),
        code('show_figure("eda/01_target_distribution.png")'),
        md("""
> **Insight Q2.1** — เพราะคลาสบวกมีเพียง **4.87%** โมเดลที่ "ทายไม่เป็น stroke ทั้งหมด"
> จะได้ Accuracy สูงถึง ~95% แต่ **Recall = 0** ซึ่งไร้ประโยชน์ทางคลินิก
> เราจึงต้องเปลี่ยน metric เป้าหมายเป็น **Recall / F1 / PR-AUC** และดู **Confusion Matrix**
> เพื่อนับจำนวนผู้ป่วยที่ถูกมองข้าม (FN) เป็นรายคน
"""),
        md("## 5. การกระจายของตัวแปรต่อเนื่อง"),
        code("""
desc = clean[list(cfg.NUMERIC_FEATURES) + [cfg.TARGET]].groupby(cfg.TARGET).describe().T
desc
"""),
        code('show_figure("eda/03_numeric_distributions.png", 1100)'),
        md("""
**อ่านกราฟ**: อายุของกลุ่ม stroke กระจายไปทางค่ามากอย่างชัดเจน (มัธยฐาน 71 vs 43 ปี)
ส่วน `avg_glucose_level` และ `bmi` กลุ่ม stroke มีค่าสูงกว่าเล็กน้อยแต่ทับซ้อนกันมาก
→ บอกได้ว่าอายุคือตัวแปรที่แยกกลุ่มได้ชัดที่สุด (จะยืนยันด้วยสถิติใน notebook 02)
"""),
        md("## 6. ตัวแปรเชิงกลุ่ม vs สถานะ Stroke"),
        code("""
for col in cfg.CATEGORICAL_FEATURES:
    tbl = pd.crosstab(clean[col], clean[cfg.TARGET], normalize="index").mul(100).round(2)
    tbl.columns = ["No Stroke (%)", "Stroke (%)"]
    print(f"--- {col} ---")
    display(tbl)
    print()
"""),
        code('show_figure("eda/04_categorical_vs_target.png", 1150)'),
        md("""
> **Insight Q2.2** — `smoking_status = 'Unknown'` มีสัดส่วนสูงถึง 30% (1,544 ราย)
> หากสัดส่วนการเกิด stroke ในกลุ่มนี้ไม่ต่างจากกลุ่มอื่น อาจแปลว่าค่า "Unknown"
> เป็น **Missing Not At Random** ที่อาจกลบความสัมพันธ์ที่แท้จริงของการสูบบุหรี่
> เป็นข้อจำกัดที่ต้องรายงานในเล่มสรุป
"""),
        md("## 7. ความสัมพันธ์ระหว่างตัวแปร"),
        code("""
num_cols = list(cfg.NUMERIC_FEATURES) + list(cfg.BINARY_FEATURES) + [cfg.TARGET]
clean[num_cols].corr(method="spearman").round(3)
"""),
        code('show_figure("eda/05_correlation_heatmap.png", 800)'),
        code('show_figure("eda/06_scatter_age_glucose_bmi.png", 1250)'),
        md("## 8. ภาระความเสี่ยงหลอดเลือด (Vascular Risk Burden)"),
        code("""
tmp = clean.copy()
tmp["vascular_risk"] = tmp["hypertension"].astype(int) + tmp["heart_disease"].astype(int)
tbl = tmp.groupby("vascular_risk")[cfg.TARGET].agg(stroke_rate="mean", n="count")
tbl["stroke_rate_%"] = (tbl["stroke_rate"] * 100).round(2)
tbl[["n", "stroke_rate_%"]]
"""),
        code('show_figure("eda/07_vascular_risk.png", 1050)'),
        md("""
> **Insight Q2.3** — จำนวนปัจจัยเสี่ยง (ความดันโลหิตสูง + โรคหัวใจ) มีความสัมพันธ์แบบ
> **เพิ่มขึ้นตามจำนวน (dose-response)**: ยิ่งมีปัจจัยเสี่ยงมาก อัตราการเกิด stroke ยิ่งสูงขึ้นชัดเจน
> รูปแบบนี้สนับสนุนสมมติฐานเชิงสาเหตุได้ดีกว่าตัวแปรเดี่ยว ๆ
"""),
        md("""
## 9. สรุป Q2

1. **ข้อมูลไม่สมดุลรุนแรง** (19.5 : 1) → เลือก metric ที่เหมาะกับคลาสบวชนอกเสียก่อน
2. **อายุ** แยกกลุ่มได้ชัดที่สุด, รองลงมาคือ **avg_glucose_level** และ **bmi**
3. **ภาระความเสี่ยงหลอดเลือด** แสดงความสัมพันธ์แบบ dose-response
4. **คุณภาพข้อมูล** มีจุดอ่อนที่ `smoking_status = Unknown` (30%) และ `bmi` หาย 3.9%
5. ขั้นตอนถัดไป → ทดสอบนัยสำคัญทางสถิติอย่างเป็นทางการใน `02_statistical_analysis.ipynb`
"""),
    ]


# ===========================================================================
# 02 — Statistics
# ===========================================================================
def build_02() -> list:
    return [
        md("""
# 02 · Statistical Analysis (Q1)

**คำถามวิจัย Q1** — ปัจจัยใดสัมพันธ์กับ Stroke อย่างมีนัยสำคัญ?

เลือกการทดสอบตาม **ชนิดของตัวแปร** ไม่ใช่ใช้ t-test กับทุกอย่าง:

| ตัวแปร | การทดสอบ | Effect size |
|---|---|---|
| เชิงกลุ่ม / ไบนารี | Chi-square test of independence | Cramér's V |
| ต่อเนื่อง | Mann-Whitney U (ไม่ต้องสมมติ normal) | rank-biserial correlation |
| ไบนารี (ความเสี่ยง) | Odds Ratio + 95% CI | — |

**ปรับ p-value ด้วย Benjamini-Hochberg (FDR)** เพราะทดสอบพร้อมกันหลายตัวแปร
→ ลดปัญหา false positive inflation
"""),
        md("## 1. เตรียมข้อมูล"),
        code(SETUP),
        code("""
from src import config as cfg
from src import data_loader as dl
from src import stats_analysis as sa

clean = dl.basic_clean(dl.load_raw())
print("n =", len(clean), "| stroke =", int(clean[cfg.TARGET].sum()))
"""),
        md("""
## 2. ตัวแปรเชิงกลุ่ม vs Stroke — Chi-square

**สมมติฐาน**
- H₀ : สัดส่วนการเกิด stroke ไม่ขึ้นกับตัวแปรนั้น (independent)
- H₁ : มีความสัมพันธ์กัน

เกณฑ์ตัดสิน: ใช้ **p-value ที่ปรับ FDR แล้ว** < 0.05 **ร่วมกับ** ขนาด effect size
(เพราะข้อมูล 5,000 แถวทำให้ p ต่ำได้ง่ายแม้ความสัมพันธ์จะเล็กมาก)
"""),
        code("""
chi = sa.chi_square_vs_target(clean)
chi[["feature", "chi2", "dof", "p_value", "p_adjusted_bh",
     "cramers_v", "effect_size", "significant_0.05", "warning"]]
"""),
        code("""
(chi.assign(stroke_rate=chi["stroke_rate_by_group_%"])
    [["feature", "p_adjusted_bh", "cramers_v", "stroke_rate"]]
    .sort_values("cramers_v", ascending=False)
    .style.background_gradient(subset=["cramers_v"], cmap="Reds")
    .format({"p_adjusted_bh": "{:.2e}"}))
"""),
        md("""
**เกณฑ์ตีความ Cramér's V** (Cohen): < 0.10 น้อยมาก | 0.10–0.19 น้อย | 0.20–0.29 ปานกลาง | ≥ 0.30 มาก

> **Insight Q1.1** — `heart_disease` (V = 0.134) และ `hypertension` (V = 0.127)
> มีความสัมพันธ์แรงที่สุดในกลุ่มตัวแปรเชิงกลุ่ม แต่อยู่ในระดับ "น้อย"
> ส่วน `gender` และ `Residence_type` **ไม่มีนัยสำคัญ** (p > 0.05)
"""),
        md("""
## 3. ตัวแปรต่อเนื่อง vs Stroke — Mann-Whitney U

**ทำไมไม่ใช้ t-test?** เพราะ t-test สมมติว่าข้อมูลเป็น normal distribution
เราจึงตรวจ normality ด้วย **Shapiro-Wilk** ก่อน (คอลัมน์ `shapiro_p_*`)
ถ้า p < 0.05 = ปฏิเสธ normality → ใช้ Mann-Whitney U ซึ่งเป็น non-parametric
"""),
        code("""
mw = sa.mann_whitney_vs_target(clean)
mw[["feature", "median_no_stroke", "median_stroke", "mean_no_stroke", "mean_stroke",
    "u_statistic", "p_value", "p_adjusted_bh", "rank_biserial",
    "shapiro_p_no_stroke", "shapiro_p_stroke", "normality_rejected", "significant_0.05"]]
"""),
        code("""
mw[["feature", "rank_biserial", "direction"]].style \\
    .background_gradient(subset=["rank_biserial"], cmap="RdBu_r", vmin=-1, vmax=1)
"""),
        md("""
**อ่านค่า rank-biserial**: ค่าอยู่ในช่วง −1 ถึง +1
- ค่าบวก = กลุ่ม stroke มีค่าสูงกว่ากลุ่มไม่เป็น stroke
- |ค่า| ยิ่งมาก = ทั้งสองกลุ่มยิ่งแยกออกจากกันชัด

> **Insight Q1.2** — `age` มีค่า rank-biserial = **0.669** ซึ่งสูงมาก
> (มัธยฐาน 71 ปี vs 43 ปี) นี่คือหลักฐานเชิงสถิติที่แข็งแรงที่สุดในชุดข้อมูลนี้
> ขณะที่ `avg_glucose_level` (0.221) และ `bmi` (0.158) มีนัยสำคัญแต่ขนาดผลเล็กกว่ามาก
"""),
        md("## 4. Odds Ratio — ขนาดความเสี่ยง"),
        code("""
or_df = sa.odds_ratios(clean)
or_df
"""),
        md("""
**ตีความ**: OR = 4.71 ของ `heart_disease` หมายความว่า
ผู้ที่มีโรคหัวใจมี **odds** ของการเกิด stroke สูงเป็น **4.71 เท่า** ของผู้ที่ไม่มี

> **Insight Q1.3** — ปัจจัยเสี่ยงที่ "แก้ไขไม่ได้" (อายุ, โรคหัวใจ, ความดัน)
> มีพลังในการทำนายสูงกว่าปัจจัยด้านพฤติกรรมอย่างการสูบบุหรี่/สถานะสมรส
> ซึ่งอธิบายได้ว่าทำไมโมเดลในเล่ม 03 จึงให้ความสำคัญกับอายุเป็นอันดับหนึ่ง
"""),
        md("## 5. สรุป Q1 (สรุปเป็นตารางเดียว)"),
        code("""
rows = []
for _, r in chi.iterrows():
    rows.append({"feature": r["feature"], "type": "categorical", "test": "Chi-square",
                 "p_adjusted": r["p_adjusted_bh"], "effect_size": r["cramers_v"],
                 "metric": "Cramér's V", "significant": r["significant_0.05"]})
for _, r in mw.iterrows():
    rows.append({"feature": r["feature"], "type": "continuous", "test": "Mann-Whitney U",
                 "p_adjusted": r["p_adjusted_bh"], "effect_size": abs(r["rank_biserial"]),
                 "metric": "rank-biserial", "significant": r["significant_0.05"]})

final = pd.DataFrame(rows).sort_values(["significant", "effect_size"], ascending=False)
final.style.background_gradient(subset=["effect_size"], cmap="Blues").format({"p_adjusted": "{:.2e}"})
"""),
        code("""
final.to_csv(cfg.TABLES_DIR / "q1_summary_combined.csv", index=False, encoding="utf-8-sig")
print("บันทึกตารางสรุปแล้ว")
"""),
        md("""
## 6. คำตอบของ Q1

ปัจจัยที่มีความสัมพันธ์กับ stroke อย่างมีนัยสำคัญทางสถิติ (FDR < 0.05) เรียงตามขนาดผล:

| อันดับ | ปัจจัย | ขนาดผล | ทิศทาง |
|---|---|---|---|
| 1 | **age** | rank-biserial 0.669 | ยิ่งอายุมาก ยิ่งเสี่ยง |
| 2 | **avg_glucose_level** | rank-biserial 0.221 | น้ำตาลสูง → เสี่ยงขึ้น |
| 3 | **bmi** | rank-biserial 0.158 | BMI สูง → เสี่ยงขึ้นเล็กน้อย |
| 4 | **heart_disease** | Cramér's V 0.134, OR 4.71 | มีโรคหัวใจ → เสี่ยงสูงมาก |
| 5 | **hypertension** | Cramér's V 0.127, OR 3.70 | มีความดันสูง → เสี่ยงสูง |
| 6 | **ever_married** | Cramér's V 0.107 | สมรสแล้ว → เสี่ยงขึ้น (likely confounded by age) |
| 7 | **work_type** | Cramér's V 0.094 | ต่างกันตามกลุ่มอาชีพ |
| 8 | **smoking_status** | Cramér's V 0.072 | มีนัยสำคัญแต่ผลเล็ก (และมี Unknown 30%) |

**ไม่มีนัยสำคัญ**: `gender` (p = 0.56), `Residence_type` (p = 0.30)

> ⚠️ **ข้อควรระวัง**: `ever_married` และ `work_type` น่าจะเป็น *proxy* ของอายุ
> (คนอายุมากย่อมมีโอกาสสมรสและเกษียณมากกว่า) จึงควรตีความว่าเป็น
> **ความสัมพันธ์ (association)** ไม่ใช่ **สาเหตุ (causation)**
> และควรตรวจ confounding เพิ่มเติมด้วย multivariable model
"""),
    ]


# ===========================================================================
# 03 — Modeling
# ===========================================================================
def build_03() -> list:
    return [
        md("""
# 03 · Model Comparison (Q3)

**คำถามวิจัย Q3** — โมเดลใดจำแนกผู้ป่วย stroke ได้ดีที่สุดภายใต้ข้อมูลไม่สมดุล?

เราเปรียบเทียบ **6 โมเดล** จาก 4 ตระกูล 알고리즘 × 2 กลยุทธ์จัดการ imbalance:

| โมเดล | กลยุทธ์ imbalance |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Logistic Regression | SMOTE (oversampling) |
| Random Forest | `class_weight='balanced_subsample'` |
| Random Forest | SMOTE |
| XGBoost | `scale_pos_weight` |
| LightGBM | `class_weight='balanced'` |
"""),
        md("## 1. เตรียมข้อมูลและแยก train/test"),
        code(SETUP),
        code(SHOW_FIG),
        code("""
from sklearn.model_selection import train_test_split
from src import config as cfg, data_loader as dl, models as mdl

clean = dl.basic_clean(dl.load_raw())
X, y = dl.split_features_target(clean)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)

print(f"Train: {len(X_train):,} (stroke {int(y_train.sum())})")
print(f"Test : {len(X_test):,} (stroke {int(y_test.sum())})")
print("ใช้ stratify=y เพื่อรักษาสัดส่วนคลาสในทั้งสองชุด")
"""),
        md("""
## 2. ระเบียบวิธี: เลือก threshold อย่างถูกต้อง

ด้วยข้อมูลไม่สมดุล (4.87%) ค่า probability ที่โมเดลให้มักต่ำกว่า 0.5 เกือบทั้งหมด
→ การใช้ threshold = 0.5 แบบไม่คิด **ไม่ควรถือเป็นค่าเริ่มต้น**

ขั้นตอนที่ใช้:

1. **Out-of-Fold (OOF) predictions** — ใช้ `cross_val_predict` บน train เท่านั้น
2. หา threshold ที่ให้ **F1 สูงสุด** จาก OOF
3. `fit` โมเดลเต็มบน train ทั้งชุด
4. ประเมินบน **test set** ด้วย threshold นั้น (และรายงานผลที่ 0.5 เทียบกันด้วย)

> ทำไมต้อง OOF? ถ้าไปหา threshold บน test set โดยตรง = **data leakage**
> ผลที่ได้จะดูดีเกินจริงและเชื่อถือไม่ได้

⚠️ **ข้อควรเข้าใจให้ถูก**: threshold ที่ optimal สำหรับ F1 **ไม่ได้แปลว่า Recall
จะสูงขึ้นเสมอ** — มันคือการเลือก "จุด" บน Precision-Recall curve
ถ้าโมเดลถูกถ่วงน้ำหนักคลาสอยู่แล้ว (balanced / SMOTE) threshold 0.5 อาจให้
Recall สูงมากอยู่แล้ว การขยับขึ้นไปที่ F1-optimal จึงเป็นการ **แลก Recall
เพื่อเอา Precision** ไม่ใช่การเพิ่ม Recall เราจะดูตัวเลขจริงในหัวข้อ 7
"""),
        md("## 3. โหลดผลลัพธ์จากการเปรียบเทียบโมเดล"),
        code("""
RETRAIN = False  # ตั้งเป็น True ถ้าต้องการเทรนใหม่จริง ๆ (ใช้เวลาหลายนาที)

summary = pd.read_csv(cfg.TABLES_DIR / "q3_model_comparison.csv")
if RETRAIN or summary is None:
    result = mdl.compare_models(X_train, y_train, X_test, y_test, verbose=True)
    summary = result.summary

cols = ["model", "threshold", "recall_sensitivity", "precision", "f1",
        "pr_auc_average_precision", "roc_auc", "specificity", "accuracy"]
summary[cols]
"""),
        md("## 4. เปรียบเทียบด้วยกราฟ"),
        code('show_figure("model/10_model_comparison.png", 1000)'),
        md("""
**อ่านกราฟ** — เราให้ความสำคัญกับ **Recall** และ **F1** มาก่อน
เพราะการ "พลาดผู้ป่วย" (False Negative) มีต้นทุนสูงกว่าการ "เตือนเกิน" (False Positive) มาก
"""),
        md("## 5. Confusion Matrix — ดูเป็นจำนวนคน"),
        code('show_figure("model/11_confusion_matrices.png", 1150)'),
        code("""
cm_tbl = pd.read_csv(cfg.TABLES_DIR / "q3_confusion_matrices.csv", index_col=0)
cm_tbl
"""),
        code("""
# แสดง FN (ผู้ป่วยที่ถูกมองข้าม) ของแต่ละโมเดล — ยิ่งน้อยยิ่งดี
best_rows = summary.set_index("model")
fn_table = best_rows[["fn", "tp", "fp", "tn", "recall_sensitivity", "f1"]].copy()
fn_table["missing_rate_%"] = (fn_table["fn"] / (fn_table["fn"] + fn_table["tp"]) * 100).round(1)
fn_table.sort_values("fn")
"""),
        md("""
## 6. Precision-Recall และ ROC Curve
"""),
        code('show_figure("model/12_pr_roc_curves.png", 1250)'),
        md("""
**ทำไม PR Curve สำคัญกว่า ROC ที่นี่?**
ROC-AUC ใช้ทั้ง TPR และ FPR ซึ่ง FPR มีตัวหารเป็น TN ที่มีอยู่มหาศาล (4,860 ราย)
ทำให้ดูเหมือนได้เปรียบเสมอแม้โมเดลจะแย่ในคลาสบวก
ส่วน **PR-AUC (Average Precision)** โฟกัสที่คลาสบวกโดยตรง
จึงสะท้อนความสามารถจริงในการจับผู้ป่วยได้ดีกว่า

> เส้น baseline สีเทาในกราฟ PR = prevalence (0.049) ไม่ใช่ 0.5
> โมเดลที่ PR-AUC = prevalence ถือว่า "ไม่ดีไปกว่าการสุ่มเลย"
"""),
        md("## 7. โมเดลที่ดีที่สุด — ขึ้นกับเกณฑ์ที่เลือก"),
        code("""
def report(model_row, label):
    print(f"=== {label}: {model_row['model']} ===")
    print(f"  threshold = {model_row['threshold']:.3f}")
    for k in ["recall_sensitivity", "precision", "f1", "pr_auc_average_precision",
              "roc_auc", "specificity", "accuracy"]:
        print(f"  {k:28s} = {model_row[k]:.4f}")
    print(f"  เทียบกับ threshold 0.5: F1 {model_row['f1@0.5']:.4f} -> {model_row['f1']:.4f} "
          f"| Recall {model_row['recall@0.5']:.4f} -> {model_row['recall_sensitivity']:.4f}")
    print()

best_f1 = summary.sort_values(["f1", "recall_sensitivity"], ascending=False).iloc[0]
best_pr = summary.sort_values(["pr_auc_average_precision", "recall_sensitivity"], ascending=False).iloc[0]
best_recall = summary.sort_values(["recall_sensitivity", "f1"], ascending=False).iloc[0]

report(best_f1, "สมดุลที่สุด (F1 สูงสุด)")
report(best_pr, "โฟกัสคลาสบวก (PR-AUC สูงสุด)")
report(best_recall, "คัดกรองแบบเน้น Recall (Recall สูงสุด)")
"""),
        md("""
### 7.1 ผลของ threshold — ไม่ได้เป็นไปในทิศทางเดียวกันทุกโมเดล

ตารางนี้เทียบ Recall/F1 ที่ **threshold 0.5** กับ **threshold ที่ tune จาก OOF**
ดูให้ดีว่า **ทิศทางของ Recall ต่างกันตามโมเดล** — เป็นบทเรียนสำคัญ
"""),
        code("""
cmp = summary[["model", "threshold", "threshold_rule",
                "recall@0.5", "recall_sensitivity", "f1@0.5", "f1"]].copy()
cmp["recall_change"] = (cmp["recall_sensitivity"] - cmp["recall@0.5"]).round(4)
cmp["f1_change"] = (cmp["f1"] - cmp["f1@0.5"]).round(4)
display(cmp.sort_values("recall_change", ascending=False))

print("โมเดลที่ Recall 'เพิ่ม' หลัง tune:", list(cmp.loc[cmp["recall_change"] > 0, "model"]))
print("โมเดลที่ Recall 'ลด' หลัง tune :", list(cmp.loc[cmp["recall_change"] < 0, "model"]))
"""),
        md("""
> **Insight Q3.2 — threshold คือการเลือก "จุด" บน PR curve ไม่ใช่ "การปรับให้ดีขึ้น"**
>
> - **โมเดลที่ยังไม่จัดการ imbalance** (RandomForest, LightGBM): ที่ 0.5 Recall ต่ำมาก
>   (0.12–0.22) เพราะ probability ของคลาสบวกถูกกดต่ำ → การ tune **ยก Recall ขึ้นมหาศาล**
> - **โมเดลที่ถ่วงน้ำหนักคลาสแล้ว** (LogReg balanced / SMOTE และ XGBoost ที่ใช้
>   `scale_pos_weight`): ที่ 0.5 Recall **สูงอยู่แล้ว** เพราะน้ำหนักของคลาสบวก
>   ดัน probability ขึ้น → การเลือกจุด F1-optimal จึง **"ลด" Recall เพื่อแลก Precision**
>
> ทั้งสองกรณีถูกต้องทั้งคู่ — มันคือการเลือกจุดบนเส้นโค้ง ตามต้นทุน FN:FP ที่เราเชื่อ
>
> ⚠️ **ข้อควรระวัง**: ถ้าต้องการ Recall สูงสุด *ไม่จำเป็น* ต้อง tune threshold เลย —
> โมเดลที่ถ่วงน้ำหนักคลาสแล้วที่ threshold 0.5 ให้ Recall สูงถึง **0.80**
> (จับผู้ป่วยได้ 40 จาก 50 ราย) โดยแลกกับ false positive ที่เพิ่มขึ้นมาก
"""),
        md("""
> **ข้อสังเกตสำคัญ**: "โมเดลที่ดีที่สุด" **ไม่ใช่คำตอบเดียว**
> ขึ้นกับว่าเราให้น้ำหนักกับอะไร
>
> - ถ้าเน้น **F1** → เลือกตัวที่สมดุล Recall/Precision
> - ถ้าเน้น **PR-AUC** → เลือกตัวที่แยกคลาสบวกได้ดีที่สุดโดยรวม (เหมาะกับงานคัดกรอง)
> - ถ้าเน้น **Recall** → เลือกตัวที่พลาดผู้ป่วยน้อยที่สุด แต่จะเตือนเกินเยอะ
>
> การรายงาน **ทางเลือกที่แตกต่างกัน** อย่างตรงไปตรงมา เป็นส่วนสำคัญของงานวิเคราะห์
> ที่ดี เพราะผู้อ่านอาจมีต้นทุน FN/FP ไม่เหมือนเรา
"""),
        md("""
## 8. สรุป Q3

**ผลลัพธ์ที่ได้ (จากการรัน pipeline)**

- โมเดลเชิงเส้น (**Logistic Regression**) ให้ผลดีที่สุดหรือใกล้เคียงดีที่สุดในชุดข้อมูลนี้
  เมื่อดูทั้ง F1, PR-AUC และ ROC-AUC — เป็นไปได้เพราะความสัมพันธ์ระหว่างฟีเจอร์กับ
  log-odds ของ stroke ค่อนข้างเป็นแบบเชิงเส้น/มอนอโทนิก (โดยเฉพาะอายุ)
- **Gradient boosting (XGBoost/LightGBM)** ไม่ได้ให้ผลดีกว่าในชุดข้อมูลนี้ เพราะ
  ข้อมูลมีเพียง ~4,000 แถวและคลาสบวกแค่ ~200 ราย โมเดลที่ซับซ้อนจึงมีแนวโน้ม overfit
- **การ tune threshold มีผลมาก แต่ทิศทางขึ้นกับโมเดล** — ดูหัวข้อ 7.1
  ไม่ควรสรุปว่า "tune แล้ว Recall จะเพิ่มเสมอ"
- **Accuracy เกือบเท่ากันทุกโมเดล (~77–88%)** ซึ่งพิสูจน์ว่า accuracy ไม่ใช่ตัวชี้วัดที่มีประโยชน์
  เพราะทุกโมเดลAccuracy สูง แต่ Recall ต่างกันเกือบสองเท่า
- **Class weighting vs SMOTE** ให้ผลใกล้เคียงกัน โดย SMOTE มักได้ Recall สูงขึ้น
  แต่ Precision ลดลง (แลกกัน) และมีข้อเสียคือสร้าง record สังเคราะห์ที่ตีความทางคลินิกไม่ได้

**ข้อจำกัดที่ต้องรายงาน**

- test set มีผู้ป่วยจริงเพียง 50 ราย → metric มีความแปรปรวนสูง
  (ผู้ป่วย 1 ราย = 2% ของ Recall) ควรดูช่วงความเชื่อมั่นจาก repeated CV ประกอบ
- ยังไม่มี external validation → ผลอาจไม่ generalize กับประชากรอื่น
"""),
    ]


# ===========================================================================
# 04 — SHAP
# ===========================================================================
def build_04() -> list:
    return [
        md("""
# 04 · SHAP Explanations (Q4)

**คำถามวิจัย Q4** — ฟีเจอร์ใดขับเคลื่อนการทำนาย (global) และทำไมแต่ละ record จึงถูกทำนายเช่นนั้น (local)?

**SHAP (SHapley Additive exPlanations)** มาจากทฤษฎีเกม (Shapley values):
แตกการทำนายของ record หนึ่งออกเป็น "ผลบวกของ contributions" ของแต่ละฟีเจอร์

```
prediction = base_value + Σ (SHAP value ของแต่ละฟีเจอร์)
```

- ค่า **+** → ดันความน่าจะเป็นไปทาง "เป็น stroke"
- ค่า **−** → ดันไปทาง "ไม่เป็น stroke"
- |ค่า| มาก → อิทธิพลมาก
"""),
        md("""
## 0. หมายเหตุเชิงเทคนิค (สำคัญ)

เราอธิบาย **classifier บนฟีเจอร์หลัง transform** (one-hot แล้ว) ไม่ใช่ที่ตัว Pipeline ทั้งก้อน
เพราะ SHAP ต้องใช้ predict function ที่รับเมทริกซ์ตัวเลข

จากนั้นเราจึง **รวมคอลัมน์ one-hot กลับเป็น base feature**
เช่น `work_type_Private` + `work_type_Govt_job` + ... → `work_type`
เพื่อให้อ่านง่ายและเทียบกับ LIME ได้ตรงกัน (LIME ทำงานระดับ base feature)

เรายัง **ไม่ใช้ KernelExplainer** เป็นค่าเริ่มต้น เพราะช้ามากและมี variance สูง
เลือกใช้ `TreeExplainer` (สำหรับ XGBoost/LightGBM/RandomForest) และ `LinearExplainer`
(สำหรับ Logistic Regression) ซึ่งเป็น exact algorithm
"""),
        md("## 1. โหลดโมเดลและข้อมูล"),
        code(SETUP),
        code(SHOW_FIG),
        code("""
import joblib
from sklearn.model_selection import train_test_split
from src import config as cfg, data_loader as dl

bundle = joblib.load(cfg.MODELS_DIR / "best_model.joblib")
model, threshold, model_name = bundle["model"], bundle["threshold"], bundle["name"]
print("โมเดล:", model_name)
print("threshold:", round(threshold, 3))

clean = dl.basic_clean(dl.load_raw())
X, y = dl.split_features_target(clean)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)
"""),
        md("""
## 2. คำนวณ SHAP values

ตัวอย่างนี้ใช้ test set 300 records เพื่อให้คำนวณเร็ว
(ใน pipeline เต็มใช้ 1,500 records — ดู `reports/tables/q4_shap_global_importance.csv`)
"""),
        code("""
from src import xai

sample = X_test.sample(300, random_state=cfg.RANDOM_STATE)
X_transformed, shap_values, base_value, explainer = xai.compute_shap(model, sample, X_train)

print("รูปร่างของ SHAP matrix:", shap_values.shape)
print("  = (จำนวน records, จำนวนฟีเจอร์หลัง one-hot)")
print("base value:", base_value, "| หน่วยของค่า SHAP:", xai.shap_space(base_value))
print()
print("ตรวจสอบ additivity (prediction = base_value + ผลรวม SHAP):")
prob = model.predict_proba(sample)[:, 1]
reconstructed = base_value + shap_values.sum(axis=1)

if xai.shap_space(base_value) == "probability":
    target = prob
    print("  หน่วยเป็น probability -> เทียบกับ predicted probability โดยตรง")
else:
    target = np.log(prob / (1 - prob))
    print("  หน่วยเป็น log-odds -> แปลง predicted probability เป็น logit ก่อนเทียบ")

print("  ค่าสูงสุดของ |ส่วนต่าง| =", np.abs(target - reconstructed).max().round(6))
"""),
        md("""
### ⚠️ หน่วยของค่า SHAP (log-odds vs probability)

| Explainer / โมเดล | หน่วย |
|---|---|
| `TreeExplainer` + RandomForest (sklearn) | probability (0–1) |
| `TreeExplainer` + XGBoost / LightGBM | **log-odds (margin)** |
| `LinearExplainer` + LogisticRegression | **log-odds (margin)** |

งานวิเคราะห์จำนวนมากรายงานค่า SHAP โดยไม่ระบุหน่วย ทำให้ตีความผิด
เราจึงบันทึกหน่วยที่ใช้ไว้ใน `reports/metrics.json` (คีย์ `shap_space`) เสมอ
"""),
        md("## 3. Global importance — ฟีเจอร์ใดสำคัญที่สุด"),
        code("""
importance = xai.global_importance(X_transformed, shap_values, group_to_base=True)
importance
"""),
        code("""
importance.to_csv(cfg.TABLES_DIR / "q4_shap_global_importance_notebook.csv",
                  index=False, encoding="utf-8-sig")
"""),
        code('show_figure("xai/20_shap_bar.png", 950)'),
        md("""
**อ่านค่า**: `mean(|SHAP|)` คือขนาดอิทธิพลเฉลี่ย และ `importance_pct` คือสัดส่วน
เมื่อเทียบกับอิทธิพลรวมทั้งหมด → บอกได้ว่าแต่ละฟีเจอร์ "แบก" การทำนายไว้กี่เปอร์เซ็นต์

> **Insight Q4.1** — **อายุ (age)** มีอิทธิพลสูงสุดอย่างชัดเจน (มักครองสัดส่วนสูงที่สุด
> ของอิทธิพลรวมทั้งหมด) สอดคล้องกับการทดสอบทางสถิติในเล่ม 02 (rank-biserial สูงสุด)
> แสดงว่า **หลักฐานเชิงสถิติและพฤติกรรมของโมเดลสอดคล้องกัน** — เป็นสัญญาณที่ดี
> ว่าคำอธิบายที่ได้น่าเชื่อถือ
"""),
        md("""
### 🚨 กับดักสำคัญ: ฟีเจอร์ไบนารีที่หายาก

`mean(|SHAP|)` **เอนเอียงเข้าข้างฟีเจอร์ที่พบบ่อย**

ตัวอย่าง: `heart_disease` พบเพียง ~5% ของ records → อีก ~95% contribution เป็น 0
ค่า `mean(|SHAP|)` จึงต่ำมาก **ทั้งที่เมื่อพบผู้ป่วยโรคหัวใจจริง อิทธิพลต่อการทำนายสูง**
(สอดคล้องกับ Odds Ratio ≈ 4.7 จากเล่ม 02)

**วิธีแก้ที่ถูกต้อง** — รายงานอิทธิพลเฉพาะเมื่อฟีเจอร์ "เปิด" เพิ่มด้วย
"""),
        code("""
imp = importance.copy()
imp["rank_overall"] = imp["mean_abs_shap"].rank(ascending=False).astype(int)

# แนวคิด "เปิด/ปิด" ใช้ได้เฉพาะตัวแปรไบนารี -> เทียบเฉพาะกลุ่มนี้
binary = imp[imp["feature"].isin(cfg.BINARY_FEATURES)].copy()
binary["rank_when_active"] = binary["mean_abs_shap_when_active"].rank(ascending=False).astype(int)

print(f"=== ตัวแปรไบนารี {len(binary)} ตัว: อันดับโดยรวม vs อันดับเมื่อ 'เปิด' ===")
display(binary[["feature", "rank_overall", "mean_abs_shap", "pct_records_active",
                "rank_when_active", "mean_abs_shap_when_active"]]
        .sort_values("rank_overall"))

print(f"=== ตารางรวมทั้งหมด {len(imp)} ฟีเจอร์ (เรียงตาม mean|SHAP|) ===")
display(imp[["rank_overall", "feature", "mean_abs_shap", "importance_pct"]])
"""),
        md("""
> **Insight Q4.2** — `heart_disease` อยู่ **อันดับสุดท้าย** เมื่อดู `mean(|SHAP|)` โดยรวม
> (เพราะมีเพียง ~4% ของ records ที่ "เปิด") แต่ **ไต่ขึ้นมาเมื่อดูเฉพาะเมื่อเปิด**
> ในทางกลับกัน `hypertension` (เปิด ~11%) ก็ขยับขึ้นเช่นกัน
>
> 👉 **บทเรียน**: `mean(|SHAP|)` ตอบคำถามว่า *"ฟีเจอร์นี้มีอิทธิพลต่อการทำนายของ
> ประชากรทั้งหมดแค่ไหน"* แต่ไม่ตอบคำถามที่แพทย์สนใจว่า
> *"ถ้าผู้ป่วยรายนี้มีปัจจัยเสี่ยงนี้ อิทธิพลมันแรงแค่ไหน"*
> → ต้องรายงานทั้งสองมุมมองคู่กันเสมอ ไม่งั้นจะตีความผิดว่าปัจจัยเสี่ยงที่พบน้อย "ไม่สำคัญ"
>
> (ตัวแปรต่อเนื่องและตัวแปรเชิงกลุ่มไม่ใช้แนวคิดนี้ — ค่าที่เป็น 0 ยังมีความหมาย
> จึงอ่านผ่าน `mean(|SHAP|)` และ beeswarm แทน)
"""),
        md("## 4. Beeswarm — ทิศทางของอิทธิพล"),
        code('show_figure("xai/21_shap_beeswarm.png", 1000)'),
        md("""
**วิธีอ่าน beeswarm**: แต่ละจุด = 1 record
- แกน X = ค่า SHAP (ซ้าย = ดันไปทางไม่เป็น stroke, ขวา = ดันไปทางเป็น stroke)
- สี = ค่าจริงของฟีเจอร์นั้น (แดง = ค่าสูง, น้ำเงิน = ค่าต่ำ)

ถ้าจุดสีแดง (ค่าสูง) กระจุกอยู่ทางขวา → ฟีเจอร์นั้น "ยิ่งค่าสูง ยิ่งเพิ่มความเสี่ยง"
"""),
        md("""
### 🔎 ปรากฏการณ์น่าสนใจ: sign flip จาก confounding

ลองสังเกตฟีเจอร์ที่มีค่าเฉลี่ย SHAP (`mean_shap`) **ตรงข้าม** กับผลการทดสอบสถิติในเล่ม 02
เช่น `ever_married`:

- **การทดสอบทางสถิติ**: คนที่สมรสแล้วมีอัตรา stroke สูงกว่า (มีนัยสำคัญ)
- **SHAP ของโมเดล**: ค่าเฉลี่ยอาจกลับเป็น **ลบ** (ดันไปทาง "ไม่เป็น stroke")

สาเหตุคือ **confounding by age** — โมเดลเห็น `age` อยู่แล้ว จึงหักอิทธิพลของอายุออกไป
และพบว่าในกลุ่มคนอายุเท่ากัน การสมรสไม่ได้เพิ่มความเสี่ยง (อาจให้ผลป้องกันด้วยซ้ำ)

> 👉 นี่คือตัวอย่างของ **Simpson's paradox** และเป็นเหตุผลว่าทำไม
> **คำอธิบายของโมเดล ≠ ความสัมพันธ์ดิบในข้อมูล** — ทั้งสองถูกต้องแต่ตอบคนละคำถาม
"""),
        md("## 5. Local explanation — ทำไม record นี้จึงถูกทำนายเช่นนั้น"),
        code("""
local = pd.read_csv(cfg.TABLES_DIR / "q4_shap_local_cases.csv")
cases = local["case"].unique()
print("เคสที่มีให้อธิบาย:", list(cases))
local.head(12)
"""),
        code("""
# เลือกเคส False Negative — ผู้ป่วยจริงที่โมเดลมองข้าม (น่ากังวลที่สุดทางคลินิก)
case = "FN (พลาดผู้ป่วยจริง)" if "FN (พลาดผู้ป่วยจริง)" in cases else cases[0]
sub = local[local["case"] == case].sort_values("shap_value")
rec = sub["record_index"].iloc[0]
view = sub[sub["record_index"] == rec][["feature", "raw_value", "shap_value", "direction", "predicted_prob"]]
print(f"เคส: {case} | record #{rec}")
view
"""),
        md("""
**วิธีแนะนำให้อ่าน**: ไล่จาก SHAP ที่เป็นลบมากที่สุด (ดันให้ "ไม่เป็น stroke")
ไปหาบวกมากที่สุด แล้วถามว่า "ถ้าฟีเจอร์นี้เปลี่ยนไป คำตอบจะพลิกไหม"
"""),
        md("### SHAP Waterfall ของ 3 เคสตัวแทน"),
        md("""
- **TP (True Positive)** — โมเดลจับผู้ป่วยได้ → ดูว่าโมเดล "เห็น" อะไร
- **FN (False Negative)** — โมเดลพลาด → ดูว่าโมเดล "ตีความผิด" ตรงไหน
- **FP (False Positive)** — เตือนเกิน → ดูว่าโมเดลตื่นตระหนกกับอะไร
"""),
        code("""
from IPython.display import Image, display

for tag in ["TP", "FN", "FP"]:
    matches = sorted((cfg.FIGURES_DIR / "xai").glob(f"22_shap_waterfall_{tag}*.png"))
    for m in matches:
        print(f"--- {tag} ---")
        display(Image(filename=str(m), width=900))
"""),
        md("""
## 6. ข้อจำกัดของ SHAP ที่ต้องรายงาน

1. **SHAP อธิบายโมเดล ไม่ได้อธิบายโรค** — ค่า SHAP คือ "โมเดลใช้ฟีเจอร์นี้อย่างไร"
   ไม่ใช่ "ฟีเจอร์นี้ทำให้เกิดโรค"
2. **ฟีเจอร์ correlated** (เช่น age กับ ever_married) จะแชร์อิทธิพลกันแบบไม่แน่นอน
   ทำให้ค่า SHAP ของแต่ละตัวผันผวนได้
3. **ค่า SHAP ขึ้นกับ background dataset** — ถ้าเปลี่ยนชุดอ้างอิง ค่าที่ได้จะเปลี่ยน
4. **one-hot grouping** เป็นการประมาณ (ยกเว้น categorical ที่เป็น mutually exclusive จริง)

## 7. สรุป Q4

- **Global**: `age` คือปัจจัยที่ขับเคลื่อนการทำนายสูงสุดอย่างชัดเจน รองลงมาคือ
  กลุ่มฟีเจอร์เชิงกลุ่มที่ correlated กับอายุ (`work_type`, `ever_married`) และ
  `avg_glucose_level` — **สอดคล้องกับ Q1** ซึ่งเป็นหลักฐานว่าโมเดลเรียนรู้แบบแผนจริง
  ไม่ได้เรียนรู้ noise
- **Local**: โมเดลใช้ "การผสมกัน" ของหลายฟีเจอร์ ไม่ได้ตัดสินจากฟีเจอร์เดียว
  เคส FN มักเป็นผู้ป่วยที่มีความเสี่ยงไม่เด่นชัด — แบบแผนเดียวกับกลุ่มที่
  การทดสอบทางสถิติก็จับได้ยาก
- **กับดัก**: `mean(|SHAP|)` เอนเอียงเข้าข้างฟีเจอร์ที่พบบ่อย ต้องรายงาน
  `mean_abs_shap_when_active` คู่กันเสมอสำหรับฟีเจอร์ไบนารีที่หายาก
- **Confounding**: อิทธิพลของฟีเจอร์บางตัว **กลับเครื่องหมาย** เมื่อเทียบกับ
  ความสัมพันธ์ดิบ เพราะโมเดลควบคุมอิทธิพลของอายุแล้ว — เป็นจุดที่ต้องอธิบายให้ชัด
  ว่า "คำอธิบายของโมเดล" กับ "ความสัมพันธ์ทางสถิติ" ตอบคนละคำถาม
"""),
    ]


# ===========================================================================
# 05 — SHAP vs LIME
# ===========================================================================
def build_05() -> list:
    return [
        md("""
# 05 · SHAP vs LIME (Q5)

**คำถามวิจัย Q5** — SHAP และ LIME ให้คำอธิบายสอดคล้องกันหรือไม่ และเราเชื่อถือคำอธิบายได้แค่ไหน?

## ความต่างเชิงทฤษฎี

| ประเด็น | SHAP | LIME |
|---|---|---|
| หลักการ | ทฤษฎีเกม (Shapley values) — แบ่งเครดิตอย่างยุติธรรม | ประมาณโมเดลด้วย linear surrogate รอบ ๆ จุดที่สนใจ |
| Consistency | มีทฤษฎีรับประกัน (local accuracy, missingness, consistency) | ไม่มีหลักประกัน |
| ความเร็ว | เร็ว (exact สำหรับ tree/linear) | ต้องสุ่มตัวอย่างเยอะ |
| Stochasticity | deterministic | **stochastic** (สุ่มทุกครั้งที่รัน) |
| Output | ค่าต่อฟีเจอร์ (บวก/ลบ) | ค่าน้ำหนักต่อเงื่อนไข (เช่น `age <= 45`) |
"""),
        md("""
## 1. โหลดผลลัพธ์การเปรียบเทียบ
"""),
        code(SETUP),
        code(SHOW_FIG),
        code("""
from src import config as cfg

comparison = pd.read_csv(cfg.TABLES_DIR / "q5_shap_vs_lime.csv")
comparison
"""),
        md("""
## 2. ตัวชี้วัดที่ใช้ประเมินความสอดคล้อง

| ตัวชี้วัด | ความหมาย | ค่าที่ดี |
|---|---|---|
| `overlap_count` | จำนวนฟีเจอร์ที่ติด top-8 ของทั้งสองวิธี | มาก |
| `jaccard_topk` | |A ∩ B| / |A ∪ B| ของเซต top-k | > 0.3 |
| `spearman_rho` | ความสอดคล้องของ *ขนาด* อิทธิพล (อันดับ) | > 0.5 |
| `sign_agreement` | สัดส่วนที่ทั้งคู่บอกทิศทางเดียวกัน (บวก/ลบ) | > 0.7 |
"""),
        code("""
summary_stats = pd.Series({
    "ค่าเฉลี่ย overlap (จาก 8)": comparison["overlap_count"].mean(),
    "ค่าเฉลี่ย Jaccard": comparison["jaccard_topk"].mean(),
    "ค่าเฉลี่ย Spearman ρ": comparison["spearman_rho"].mean(),
    "ค่าเฉลี่ย sign agreement": comparison["sign_agreement"].mean(),
    "จำนวน records": len(comparison),
}).round(3)
summary_stats
"""),
        code('show_figure("xai/31_shap_vs_lime.png", 1250)'),
        md("""
## 3. ดูรายละเอียดรายตัวว่าเห็นตรงกันไหม
"""),
        code("""
view = comparison[["record_index", "actual", "overlap_count", "jaccard_topk",
                   "spearman_rho", "sign_agreement"]].sort_values("jaccard_topk", ascending=False)
view
"""),
        code("""
# เคสที่ตรงกันมากที่สุด และน้อยที่สุด
highest = comparison.loc[comparison["jaccard_topk"].idxmax()]
lowest = comparison.loc[comparison["jaccard_topk"].idxmin()]

for title, row in [("เห็นตรงกันมากที่สุด", highest), ("เห็นตรงกันน้อยที่สุด", lowest)]:
    print(f"=== {title} (record #{row['record_index']}) ===")
    print(f"  Jaccard = {row['jaccard_topk']:.3f} | overlap = {row['overlap_count']}/8")
    print(f"  SHAP เห็น: {row['shap_top']}")
    print(f"  LIME เห็น: {row['lime_top']}")
    print()
"""),
        md("""
## 4. LIME ระดับ record
"""),
        code("""
lime_df = pd.read_csv(cfg.TABLES_DIR / "q4_lime_local.csv")
records = lime_df["record_index"].unique()
print(f"มีคำอธิบาย LIME ทั้งหมด {len(records)} records")
print("ตัวอย่าง record:", [int(r) for r in records[:10]])
lime_df.head(12)
"""),
        code("""
from IPython.display import Image, display
for p in sorted((cfg.FIGURES_DIR / "xai").glob("30_lime_case_*.png"))[:3]:
    display(Image(filename=str(p), width=850))
"""),
        md("""
## 5. สรุป Q5 แบบอัตโนมัติ (อ่านค่าจริงจากตาราง)
"""),
        code("""
j = comparison["jaccard_topk"].mean()
r = comparison["spearman_rho"].mean()
s = comparison["sign_agreement"].mean()
ov = comparison["overlap_count"].mean()

def verdict(value, good, label):
    ok = value >= good
    return f"{'✅' if ok else '⚠️'} {label}: {value:.3f} (เกณฑ์ ≥ {good})"

print(verdict(ov, 4.0, "top-8 overlap เฉลี่ย (ฟีเจอร์)"))
print(verdict(j, 0.30, "Jaccard เฉลี่ย"))
print(verdict(r, 0.50, "Spearman rho เฉลี่ย"))
print(verdict(s, 0.70, "Sign agreement เฉลี่ย"))
print()
print("สรุป:", "ทั้งสองวิธี 'ชี้ทางเดียวกัน' ในภาพรวม (convergent)"
      if (j >= 0.30 and s >= 0.70) else
      "ทั้งสองวิธี 'เห็นต่างกันพอสมควร' (divergent) — ต้องตีความอย่างระวัง")
"""),
        md("""
## 6. อภิปรายผล

### คำตอบของ Q5

จากตัวเลขที่คำนวณได้ข้างต้น เราประเมินได้ 2 กรณี:

**กรณี A: สอดคล้องกัน (convergent)** — overlap/Jaccard/sign agreement สูง
→ ทั้งสองวิธีชี้ไปทางเดียวกัน เรามีความมั่นใจในคำอธิบายมากขึ้น

**กรณี B: เห็นต่างกัน (divergent)** — Jaccard ต่ำ และ/หรือ Spearman ต่ำ
→ **นี่คือผลลัพธ์ที่มีคุณค่าเช่นกัน** เพราะเตือนเราว่าไม่ควรเชื่อคำอธิบายจากวิธีเดียว

### ทำไม SHAP กับ LIME จึงมักไม่ตรงกันเต็มที่

1. **ต่างทฤษฎีโดยพื้นฐาน** — SHAP ใช้ Shapley values แบ่งเครดิตแบบ additive
   ส่วน LIME สร้าง linear surrogate เฉพาะจุด
2. **LIME discretize ตัวแปรต่อเนื่อง** (`age <= 45`) → สูญเสียความละเอียด
   และอาจให้ความสำคัญกับ "ขอบเขต" มากกว่าฟีเจอร์ทั้งตัว
3. **LIME เป็น stochastic** — รันใหม่ได้ผลต่างกัน ควรตั้ง seed และรายงานหลายรอบ
4. **ฟีเจอร์ correlated** — SHAP อาจกระจายเครดิตไปยังฟีเจอร์ที่ LIME ไม่ได้เลือก
5. **สเกลต่างกัน** — SHAP ให้ค่าต่อ "หน่วยฟีเจอร์ที่เพิ่มขึ้น" ส่วน LIME ให้ค่าต่อ
   "ช่วงที่ discretize" → ขนาดไม่เทียบกันตรง ๆ (จึงใช้ Spearman/อันดับ ไม่ใช้ Pearson)

### ข้อเสนอเชิงปฏิบัติ

- ใช้ **SHAP เป็นแกนหลัก** (deterministic + มีทฤษฎีรับประกัน) แล้วใช้ **LIME เป็น cross-check**
- ถ้า record ใดทั้งสองวิธี **เห็นต่างกันมาก** → อย่านำคำอธิบายนั้นไปใช้ตัดสินใจ
  โดยไม่ตรวจสอบเพิ่ม
- รายงาน **ข้อจำกัดของ XAI** คู่กับผลเสมอ — โดยเฉพาะเมื่อนำไปใช้กับงานสุขภาพ

## 7. สรุปทั้งโปรเจกต์ (Final Analysis)

| คำถาม | คำตอบสั้น |
|---|---|
| **Q1** | อายุคือปัจจัยที่สัมพันธ์แรงที่สุด (rank-biserial 0.669) ถัดมาคือน้ำตาลในเลือด, BMI, โรคหัวใจ (OR 4.7), ความดันสูง (OR 3.7); เพศและที่อยู่อาศัยไม่มีนัยสำคัญ |
| **Q2** | ข้อมูลไม่สมดุล 19.5:1 ทำให้ Accuracy ไร้ความหมาย ต้องใช้ Recall/F1/PR-AUC |
| **Q3** | โมเดลเชิงเส้น (LogReg) ให้ผลดีที่สุดหรือเทียบเท่า; threshold tuning สำคัญกว่าเปลี่ยนโมเดล |
| **Q4** | อายุครองอิทธิพลส่วนใหญ่ในทุกมุมมอง; ต้องระวังการตีความฟีเจอร์ไบนารีที่หายาก |
| **Q5** | ระดับความสอดคล้อง SHAP–LIME ต้องอ่านจากตัวเลขจริง — ไม่ควรเชื่อวิธีใดวิธีหนึ่งเดี่ยว ๆ |

### Insight รวม 3 ชั้น

1. **Data layer** — Stroke สัมพันธ์กับอายุอย่างชัดเจนที่สุด และมีรูปแบบ dose-response
   ตามภาระความเสี่ยงหลอดเลือด (ความดัน + โรคหัวใจ) ซึ่งสนับสนุนสมมติฐานเชิงสาเหตุ
2. **Model layer** — ข้อมูลขนาดเล็ก + คลาสบวกน้อย ทำให้โมเดลเรียบง่ายชนะโมเดลซับซ้อน
   และการปรับ threshold มีผลต่อ Recall มากกว่าการเปลี่ยนอัลกอริทึม
3. **XAI layer** — โมเดลให้คำอธิบายที่สอดคล้องกับสถิติในระดับ global
   แต่ในระดับ local ยังมีความไม่แน่นอน จึงต้องใช้ XAI เป็นเครื่องมือ "ตรวจสอบ"
   ไม่ใช่ "คำตอบสุดท้าย"

### ข้อเสนอแนะเชิงปฏิบัติ

- **คัดกรอง**: ใช้อายุ + ความดัน + โรคหัวใจ เป็นเกณฑ์คัดกรองเบื้องต้น (ต้นทุนต่ำ ได้ผลดี)
- **เก็บข้อมูลเพิ่ม**: ลดสัดส่วน `smoking_status = Unknown` (30%) เพื่อประเมินบทบาท
  การสูบบุหรี่ได้จริง และควรเก็บข้อมูล follow-up แบบ longitudinal
- **ก่อนใช้งานจริง**: ต้องมี external validation กับประชากรอื่น และประเมิน fairness
  ข้ามกลุ่มเพศ/อาชีพก่อน
"""),
    ]


# ===========================================================================
def main() -> None:
    write_notebook("01_eda.ipynb", build_01())
    write_notebook("02_statistical_analysis.ipynb", build_02())
    write_notebook("03_modeling.ipynb", build_03())
    write_notebook("04_shap_explanations.ipynb", build_04())
    write_notebook("05_shap_vs_lime.ipynb", build_05())
    print("\nสร้าง notebook ทั้ง 5 เล่มเรียบร้อย")


def write_notebook(name: str, cells: list) -> None:
    path = NB_DIR / name
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("เขียนแล้ว:", path)


if __name__ == "__main__":
    main()
