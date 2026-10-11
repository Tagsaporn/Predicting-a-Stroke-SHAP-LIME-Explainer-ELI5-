"""
data_loader.py
==============
อ่านข้อมูลดิบ + ทำความสะอาดเบื้องต้น (basic cleaning)

หลักการ:
    * ``load_raw()``      -> อ่าน CSV ตามสภาพจริง ไม่แก้อะไร (ใช้ทำ EDA)
    * ``basic_clean()``   -> ลบ id / จัดการค่าผิดปกติเชิงโครงสร้าง (ใช้ป้อนโมเดล)
    * ค่า missing ของ bmi **ไม่** ถูกเติมที่นี่ แต่ปล่อยให้ imputer ใน
      sklearn Pipeline เติม เพื่อกัน data leakage จาก train -> test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg


def _resolve_raw_path(path: str | Path | None = None) -> Path:
    """หาไฟล์ CSV ต้นฉบับจากหลายตำแหน่งที่เป็นไปได้"""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล: {p}")
        return p

    for candidate in cfg.RAW_CSV_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n  - ".join(str(p) for p in cfg.RAW_CSV_CANDIDATES)
    raise FileNotFoundError("หาไฟล์ healthcare-dataset-stroke-data.csv ไม่เจอใน:\n  - " + searched)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """อ่านข้อมูลดิบจาก CSV โดยไม่แก้ไขค่าใด ๆ"""
    csv_path = _resolve_raw_path(path)
    df = pd.read_csv(csv_path)
    # ตัดช่องว่างหัวคอลัมน์ (กันปัญหา unseen whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_dataframe(processed_path: str | Path | None = None) -> pd.DataFrame:
    """
    อ่านข้อมูลที่ทำความสะอาดแล้วจาก data/processed ถ้ามี
    ไม่มีก็จะสร้างจากข้อมูลดิบให้ทันที
    """
    if processed_path is None:
        processed_path = cfg.PROCESSED_DATA_DIR / "stroke_clean.csv"
    processed_path = Path(processed_path)
    if processed_path.exists():
        return pd.read_csv(processed_path)
    df = basic_clean(load_raw())
    cfg.ensure_dirs()
    df.to_csv(processed_path, index=False)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    ทำความสะอาดเบื้องต้น:
        1. ลบคอลัมน์ id (ไม่มีข้อมูลเชิงทำนาย)
        2. ลบแถวที่ gender == 'Other' (มีเพียง 1 แถว ไม่พอสร้างสมดุลให้โมเดล)
        3. จัดการค่าที่หาย/ค่าผิดปกติเชิงโครงสร้าง:
           - bmi = 'N/A' หรือ 0 -> NaN (เตรียมให้ imputer เติม)
           - smoking_status: คง 'Unknown' ไว้เป็น level ของตัวเอง
             เพราะการเดาค่าอาจสร้างอคติ (จะอภิปรายใน Process Book)
    """
    df = df.copy()

    if cfg.ID_COL in df.columns:
        df = df.drop(columns=[cfg.ID_COL])

    # bmi อาจถูกอ่านมาเป็น object ถ้ามีค่า 'N/A' ปน
    if df["bmi"].dtype == object:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").astype("float64")
    # ค่า bmi <= 0 ไม่สมเหตุสมผลทางชีววิทยา -> ถือเป็นค่าที่หาย
    df.loc[df["bmi"] <= 0, "bmi"] = np.nan

    # ลบเพศ 'Other' (n = 1) — ไม่มีอำนาจทางสถิติ
    if "gender" in df.columns:
        df = df[df["gender"] != "Other"].reset_index(drop=True)

    # ตัด Leading/Trailing space ของคอลัมน์หมวดหมู่
    for col in cfg.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df[cfg.TARGET] = df[cfg.TARGET].astype(int)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """แยก X (features) ออกจาก y (target)"""
    missing = [c for c in cfg.ALL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"คอลัมน์หายไปจากข้อมูล: {missing}")
    X = df[list(cfg.ALL_FEATURES)].copy()
    y = df[cfg.TARGET].astype(int).copy()
    return X, y


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    สรุปคุณภาพข้อมูลรายคอลัมน์: ชนิดข้อมูล, ค่าที่หาย,
    จำนวนค่าที่ไม่ซ้ำ, ตัวอย่างค่า
    """
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "n_missing": int(series.isna().sum()),
                "pct_missing": round(float(series.isna().mean() * 100), 2),
                "n_unique": int(series.nunique(dropna=True)),
                "sample_values": ", ".join(
                    map(str, series.dropna().unique()[:4])
                ),
            }
        )
    return pd.DataFrame(rows)
