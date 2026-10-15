"""
config.py
=========
Central configuration: paths, column roles, and shared constants.

ทุกโมดูลในโปรเจกต์อ่านค่าคงที่จากไฟล์นี้ที่เดียว
เพื่อให้แก้ไขพาธหรือนิยามคอลัมน์ได้จากจุดเดียว
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Console encoding (Windows console เป็น cp1252 โดยค่าเริ่มต้น -> ข้อความไทยพัง)
# บังคับ UTF-8 ให้ทุก entry point ที่ import โมดูลนี้
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# ---------------------------------------------------------------------------
# Paths (อ้างอิงจากตำแหน่งไฟล์นี้ -> ย้ายโปรเจกต์ได้โดยไม่ต้องแก้โค้ด)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
TABLES_DIR: Path = REPORTS_DIR / "tables"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# ไฟล์ข้อมูลต้นฉบับ (สำเนาไว้ใน data/raw เพื่อไม่แตะไฟล์เดิมที่ root)
RAW_CSV_CANDIDATES: tuple[Path, ...] = (
    RAW_DATA_DIR / "healthcare-dataset-stroke-data.csv",
    PROJECT_ROOT / "healthcare-dataset-stroke-data.csv",
    PROJECT_ROOT / "archive" / "healthcare-dataset-stroke-data.csv",
    PROJECT_ROOT.parent / "archive" / "healthcare-dataset-stroke-data.csv",
)

# ---------------------------------------------------------------------------
# Column roles
# ---------------------------------------------------------------------------
TARGET: str = "stroke"
ID_COL: str = "id"

NUMERIC_FEATURES: tuple[str, ...] = ("age", "avg_glucose_level", "bmi")
BINARY_FEATURES: tuple[str, ...] = ("hypertension", "heart_disease")
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
)

ALL_FEATURES: tuple[str, ...] = (
    NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
)

# ป้ายกำกับภาษาไทยสำหรับรายงาน/กราฟ
FEATURE_LABELS_TH: dict[str, str] = {
    "age": "อายุ (ปี)",
    "avg_glucose_level": "ระดับน้ำตาลในเลือดเฉลี่ย",
    "bmi": "ดัชนีมวลกาย (BMI)",
    "hypertension": "ความดันโลหิตสูง",
    "heart_disease": "โรคหัวใจ",
    "gender": "เพศ",
    "ever_married": "สถานะสมรส",
    "work_type": "ประเภทงาน",
    "Residence_type": "ที่อยู่อาศัย",
    "smoking_status": "สถานะการสูบบุหรี่",
    "stroke": "Stroke",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 130
PALETTE_STROKE: dict[int, str] = {0: "#4C9F70", 1: "#D64550"}
COLORS: dict[str, str] = {
    "primary": "#2E5A88",
    "secondary": "#D64550",
    "accent": "#F0A202",
    "neutral": "#6C757D",
    "positive": "#4C9F70",
}


def ensure_dirs() -> None:
    """สร้างโฟลเดอร์ผลลัพธ์ทั้งหมดถ้ายังไม่มี"""
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
