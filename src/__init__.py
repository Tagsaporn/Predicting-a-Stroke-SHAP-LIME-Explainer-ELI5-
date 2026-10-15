"""
src
===
Stroke Prediction + XAI (SHAP / LIME) — โมดูลหลักของโปรเจกต์

โมดูล:
    config          ค่าคงที่และพาธทั้งหมด
    data_loader     อ่านข้อมูล + ทำความสะอาดเบื้องต้น
    preprocessing   สร้าง sklearn preprocessor (impute/scale/encode)
    stats_analysis  การทดสอบทางสถิติสำหรับ Q1
    models          เทรนและเปรียบเทียบโมเดลสำหรับ Q3
    xai             SHAP / LIME explanations สำหรับ Q4–Q5
    visualization   ฟังก์ชันสร้างกราฟทั้งหมด
    run_pipeline    รันทั้งโปรเจกต์แบบ end-to-end
"""

from __future__ import annotations

__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "stats_analysis",
    "models",
    "xai",
    "visualization",
    "run_pipeline",
]
