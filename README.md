# 🧠 Stroke Prediction + XAI (SHAP / LIME)

โปรเจกต์วิเคราะห์ข้อมูลโรคหลอดเลือดสมอง (Stroke) ครบวงจร:
**ศึกษาปัจจัย → ทดลองจำแนก → อธิบายผล**

| | |
|---|---|
| **Dataset** | Stroke Prediction Dataset — 5,110 records, 12 คอลัมน์ |
| **เป้าหมาย** | ศึกษาปัจจัยที่สัมพันธ์กับ Stroke + จำแนก + อธิบายผลด้วย XAI |
| **เทคนิค** | EDA, Statistical Testing, Machine Learning, SHAP, LIME |
| **Deliverables** | Notebooks · Reproducible scripts · Streamlit Dashboard · Process Book |

> ⚠️ โปรเจกต์นี้เป็นงานวิเคราะห์เชิงวิชาการ **ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์**

---

## ❓ Research Questions (Q1–Q5)

| # | คำถามวิจัย | เครื่องมือ | ผลลัพธ์ |
|---|---|---|---|
| **Q1** | ปัจจัยใด**สัมพันธ์**กับ Stroke อย่างมีนัยสำคัญ? | Chi-square + Cramér's V, Mann-Whitney U + rank-biserial, Odds Ratio, BH-FDR | `reports/tables/q1_*.csv` |
| **Q2** | ข้อมูลกระจายอย่างไร และความไม่สมดุลของคลาสส่งผลต่อการเลือก metric อย่างไร? | Univariate / Bivariate EDA + Visualization | `reports/figures/eda/`, `q2_*.csv` |
| **Q3** | โมเดลใดจำแนกผู้ป่วยได้ดีที่สุดภายใต้ข้อมูลไม่สมดุล? | Logistic Regression / Random Forest / XGBoost / LightGBM × (class weight, SMOTE) + threshold tuning | `q3_*.csv`, `figures/model/` |
| **Q4** | ฟีเจอร์ใดขับเคลื่อนการทำนาย (global) และทำไม record นั้นจึงถูกทำนายเช่นนั้น (local)? | SHAP: bar, beeswarm, waterfall | `q4_*.csv`, `figures/xai/` |
| **Q5** | SHAP และ LIME ให้คำอธิบาย**สอดคล้อง**กันหรือไม่? | Top-k overlap, Jaccard, Spearman ρ, sign agreement | `q5_shap_vs_lime.csv` |

---
| ชื่อ-นามสกุล | รหัสนักศึกษา |
|---|---|
| **นางสาวทักษพร มูลมณี** | 653380198-5 |
| **นางสาว/นาย** |  |
| **นางสาว/นาย** |  |
---

## 📁 โครงสร้างโปรเจกต์

```
Stroke-Prediction-XAI/
├── data/
│   ├── raw/                     # ข้อมูลต้นฉบับ (สำเนา ไม่แก้ไข)
│   └── processed/               # ข้อมูลหลังทำความสะอาด
├── src/                         # โค้ดหลัก (import ได้ รันซ้ำได้)
│   ├── config.py                # พาธ + นิยามคอลัมน์ + ค่าคงที่
│   ├── data_loader.py           # โหลด + ทำความสะอาด + data quality report
│   ├── preprocessing.py         # sklearn ColumnTransformer (impute/scale/one-hot)
│   ├── stats_analysis.py        # Q1: การทดสอบทางสถิติทั้งหมด
│   ├── models.py                # Q3: model zoo + threshold tuning + evaluation
│   ├── xai.py                   # Q4–Q5: SHAP + LIME + วัดความสอดคล้อง
│   ├── visualization.py         # ฟังก์ชันสร้างกราฟทั้งหมด
│   └── run_pipeline.py          # รันทั้งโปรเจกต์แบบ end-to-end
├── notebooks/                   # 01_eda → 05_shap_vs_lime (อ่านตามลำดับได้)
├── tools/
│   └── build_notebooks.py       # สร้าง notebook ใหม่จากโค้ด
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── reports/
│   ├── figures/{eda,model,xai}/ # กราฟทั้งหมด
│   ├── tables/                  # ตารางผลลัพธ์ทั้งหมด (CSV)
│   ├── metrics.json             # ตัวเลขสรุปสำคัญ
│   ├── PROCESS_BOOK.md          # บันทึกกระบวนการทั้งหมด
│   └── FINAL_ANALYSIS.md        # สรุป Insight: Data + Model + XAI
├── models/                      # โมเดลที่เทรนแล้ว (.joblib)
└── requirements.txt
```

---

## 🚀 วิธีใช้งาน

### 1) ติดตั้งสภาพแวดล้อม

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) รัน pipeline ทั้งหมด

```powershell
# รันเต็ม (เทรน 6 โมเดล + SHAP + LIME)
.\.venv\Scripts\python.exe -m src.run_pipeline

# โหมดเร็ว (ลดขนาดตัวอย่าง SHAP/LIME)
.\.venv\Scripts\python.exe -m src.run_pipeline --quick

# ข้าม LIME
.\.venv\Scripts\python.exe -m src.run_pipeline --skip-lime
```

### 3) เปิด Dashboard

```powershell
.\.venv\Scripts\streamlit.exe run dashboard/app.py
```

### 4) เปิด Notebooks

```powershell
.\.venv\Scripts\jupyter.exe lab notebooks
```

### 5) สร้าง notebook ใหม่จากโค้ด (ถ้าแก้เนื้อหาใน `tools/build_notebooks.py`)

```powershell
.\.venv\Scripts\python.exe tools\build_notebooks.py
```

---

## 🔬 ระเบียบวิธีโดยสรุป

**Class Imbalance** — คลาสบวกมีเพียง **4.87%** (249/5,109) ทำให้ Accuracy หลอกตา
(ทำนาย "ไม่เป็น stroke" ทุกคนได้ Accuracy ~95% แต่ Recall = 0) จึงใช้:

- **Recall / Sensitivity** — ต้นทุนการพลาดผู้ป่วย (FN) สูงกว่าการเตือนเกิน (FP) มาก
- **F1** — สมดุล Recall กับ Precision
- **PR-AUC (Average Precision)** — เหมาะกับข้อมูลไม่สมดุลกว่า ROC-AUC
- **Confusion Matrix** — นับ FN/FP เป็นจำนวนคนจริงเพื่อประเมินทางคลินิก

**Threshold** — ไม่ใช้ 0.5 ตายตัว แต่เลือกจาก **Out-of-Fold (OOF) predictions**
บน training set แล้วจึงประเมินบน test set ที่บริสุทธิ์ → กัน leakage
(พร้อมรายงานผลที่ threshold 0.5 เทียบกันด้วย เพราะ **ทิศทางของ Recall ต่างกันตามโมเดล** —
ดู [`FINAL_ANALYSIS.md`](reports/FINAL_ANALYSIS.md) §Insight M3)

**Leakage Prevention** — imputation, scaling, SMOTE อยู่ใน Pipeline และ fit
เฉพาะบน training fold

**Statistical Rigor** — เลือกการทดสอบตามชนิดตัวแปร + รายงาน effect size
(ไม่ดูแต่ p-value) + ปรับ p ด้วย Benjamini-Hochberg FDR

---

## 📊 ผลลัพธ์

### Q1 — ปัจจัยที่สัมพันธ์กับ Stroke

| อันดับ | ปัจจัย | หลักฐาน | ขนาด |
|---|---|---|---|
| 1 | **อายุ** | rank-biserial **0.669** (median 43 → 71 ปี) | แรงที่สุด |
| 2 | **โรคหัวใจ** | Odds Ratio **4.71** (95% CI 3.34–6.64) | เสี่ยงเพิ่ม 4.7 เท่า |
| 3 | **ความดันโลหิตสูง** | Odds Ratio **3.70** (95% CI 2.74–4.98) | เสี่ยงเพิ่ม 3.7 เท่า |
| 4 | น้ำตาลในเลือด | rank-biserial 0.221 | น้อย |
| 5 | BMI | rank-biserial 0.158 | น้อยมาก |
| — | เพศ / ที่อยู่อาศัย | p = 0.56 / 0.30 | **ไม่มีนัยสำคัญ** |

### Q3 — โมเดล (test 1,022 records, ผู้ป่วยจริง 50 ราย)

| โมเดล | Recall | F1 | PR-AUC |
|---|---|---|---|
| **LogReg (class_weight=balanced)** ⭐ | 0.60 | **0.326** | 0.259 |
| LogReg (SMOTE) | **0.70** | 0.315 | **0.279** |
| RF / XGBoost / LightGBM | 0.44–0.66 | 0.218–0.258 | 0.142–0.194 |

⭐ = เลือกโดย F1 · ที่ threshold 0.5 ทั้งสอง LogReg ให้ Recall **0.80**

### Q4 — SHAP (log-odds space)

`age` เพียงตัวเดียว = **50.0%** ของอิทธิพลรวมทั้งหมด → รองลงมา `work_type` (13.1%),
`smoking_status` (8.8%) ซึ่งเป็น proxy ของอายุ
`heart_disease` อยู่อันดับสุดท้ายโดย `mean(|SHAP|)` แต่อิทธิพล **เมื่อพบผู้ป่วยจริง
สูงกว่าค่าเฉลี่ยรวม 10 เท่า** → ต้องรายงาน `mean_abs_shap_when_active` คู่กันเสมอ

### Q5 — SHAP vs LIME (15 records)

Sign agreement **0.964** · Spearman **0.782** · Jaccard **0.660**
→ **ทั้งสองวิธีสอดคล้องกัน (convergent)** แต่ยังใช้ SHAP เป็นแกนหลัก

ตัวเลขทั้งหมดสร้างใหม่ได้จาก `python -m src.run_pipeline`
ดูรายละเอียดที่ [`reports/PROCESS_BOOK.md`](reports/PROCESS_BOOK.md) และ
[`reports/FINAL_ANALYSIS.md`](reports/FINAL_ANALYSIS.md)

---

## ⚠️ ข้อจำกัด

- เป็นข้อมูล cross-sectional → สรุปได้แค่ **ความสัมพันธ์** ไม่ใช่ **causation**
- `smoking_status = "Unknown"` สูงถึง 30% → บทบาทการสูบบุหรี่อาจถูกประเมินต่ำกว่าจริง
- `bmi` ขาด 3.9% → เติมด้วย median ภายใน pipeline
- คลาสบวชมีเพียง 249 ราย → metric มี variance สูง ควรดู CV ประกอบ
- SHAP/LIME อธิบาย **โมเดล** ไม่ได้อธิบาย **โรค**

---

## 📜 License

ดู `LICENSE` ของชุดข้อมูลต้นฉบับ
