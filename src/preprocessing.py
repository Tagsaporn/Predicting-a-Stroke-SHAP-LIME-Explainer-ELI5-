"""
preprocessing.py
================
สร้างและจัดการ preprocessing pipeline ของ scikit-learn

เหตุผลที่ใช้ sklearn Pipeline:
    * imputer/scaler/encoder ถูก fit บน **train fold เท่านั้น**
      -> ป้องกัน data leakage ไปยัง test set และตอนทำ cross-validation
    * SHAP/LIME ทำงานกับฟีเจอร์หลัง transform ได้สอดคล้องกัน
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


def build_column_transformer(scale_numeric: bool = True) -> ColumnTransformer:
    """
    สร้าง ColumnTransformer สำหรับ 3 กลุ่มคอลัมน์

    Parameters
    ----------
    scale_numeric : bool
        - True  -> StandardScaler หลัง impute (จำเป็นสำหรับ Logistic Regression
                   และทำให้ค่า SHAP เทียบกันได้ระดับหนึ่ง)
        - False -> ส่งค่า numeric ที่ impute แล้วออกไปตรง ๆ (เหมาะกับ tree model
                   แต่ในทางปฏิบัติเปิด True ไว้ก็ไม่มีผลเสีย)
    """
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(steps=numeric_steps)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(cfg.NUMERIC_FEATURES)),
            ("bin", "passthrough", list(cfg.BINARY_FEATURES)),
            ("cat", categorical_pipe, list(cfg.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """ดึงชื่อฟีเจอร์หลัง transform เรียงตามลำดับจริงในเมทริกซ์"""
    names = preprocessor.get_feature_names_out()
    return [str(n) for n in names]


def transform_to_frame(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    prefix: str = "",
) -> pd.DataFrame:
    """แปลง X แล้วคืนเป็น DataFrame พร้อมชื่อคอลัมน์ (สะดวกต่อการอ่าน/SHAP)"""
    Xt = preprocessor.transform(X)
    if hasattr(Xt, "toarray"):  # เผื่อ sparse
        Xt = Xt.toarray()
    names = get_feature_names(preprocessor)
    df = pd.DataFrame(Xt, columns=names, index=X.index)
    if prefix:
        df = df.add_prefix(prefix)
    return df


def base_feature_of(
    col: str,
    base_features: set[str] | None = None,
) -> str:
    """
    แปลงชื่อคอลัมน์หลัง transform ให้เป็นชื่อ "base feature" ต้นทาง

    ``work_type_Private`` -> ``work_type``
    ``age``               -> ``age``

    ใช้ร่วมกันทั้งตอนรวม SHAP และตอนแปลงชื่อฟีเจอร์ของ LIME
    เพื่อให้สองวิธีอ้างอิงฟีเจอร์ชุดเดียวกันจริง ๆ (สำคัญมากต่อ Q5)
    """
    if base_features is None:
        base_features = set(cfg.CATEGORICAL_FEATURES) | set(cfg.NUMERIC_FEATURES) | set(
            cfg.BINARY_FEATURES
        )
    if col in base_features:
        return col
    candidates = [b for b in base_features if col.startswith(b + "_")]
    return max(candidates, key=len) if candidates else col


def group_onehot_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    รวมคอลัมน์ one-hot ของตัวแปรเชิงกลุ่มกลับเป็นฟีเจอร์ต้นทาง

    เช่น ``work_type_Private``, ``work_type_Govt_job`` -> ``work_type``
    ใช้เวลาสรุป SHAP ระดับ "base feature" ให้อ่านง่าย
    """
    mapping = {col: base_feature_of(col) for col in X.columns}
    grouped = X.T.groupby(pd.Series(mapping, name="base_feature")).sum().T
    return grouped


# ---------------------------------------------------------------------------
# Feature engineering (ทางเลือก — เปิดใช้เมื่อต้องการทดลองใน Q3)
# ---------------------------------------------------------------------------
def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่มฟีเจอร์เชิงคลินิกที่ตีความได้ง่าย (ไม่บังคับ)

    * age_band        : ช่วงอายุ 10 ปี
    * glucose_band    : กลุ่มระดับน้ำตาล (ใช้เกณฑ์ ADA: <100 ปกติ, 100-125 เสี่ยง, >=126 เบาหวาน)
    * bmi_band        : กลุ่ม BMI ตามเกณฑ์ WHO
    * vascular_risk   : จำนวนปัจจัยเสี่ยงหัวใจและหลอดเลือด (hypertension + heart_disease)
    """
    X = X.copy()

    X["age_band"] = pd.cut(
        X["age"],
        bins=[-0.1, 30, 45, 60, 75, 120],
        labels=["<30", "30-44", "45-59", "60-74", "75+"],
    ).astype(str)

    X["glucose_band"] = pd.cut(
        X["avg_glucose_level"],
        bins=[-0.1, 100, 125, 400],
        labels=["normal(<100)", "prediabetes(100-125)", "diabetes(>=126)"],
    ).astype(str)

    X["bmi_band"] = pd.cut(
        X["bmi"],
        bins=[-0.1, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    ).astype(str)

    X["vascular_risk"] = X["hypertension"].astype(int) + X["heart_disease"].astype(int)

    return X


ENGINEERED_CATEGORICAL: tuple[str, ...] = ("age_band", "glucose_band", "bmi_band")
ENGINEERED_NUMERIC: tuple[str, ...] = ("vascular_risk",)


def build_engineered_column_transformer(scale_numeric: bool = True) -> ColumnTransformer:
    """ColumnTransformer เวอร์ชันที่รวมฟีเจอร์วิศวกรรมเพิ่มเติม"""
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_cols = list(cfg.NUMERIC_FEATURES) + list(ENGINEERED_NUMERIC)
    categorical_cols = list(cfg.CATEGORICAL_FEATURES) + list(ENGINEERED_CATEGORICAL)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("bin", "passthrough", list(cfg.BINARY_FEATURES)),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def positive_rate(y: pd.Series) -> float:
    """สัดส่วนคลาสบวก (ใช้คำนวณ class weight)"""
    y = np.asarray(y)
    return float((y == 1).mean())
