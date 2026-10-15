"""
smoke_test_dashboard.py
=======================
ตรวจว่า Streamlit dashboard ทุกหน้า "รันได้จริงโดยไม่ error"

ใช้ `streamlit.testing.v1.AppTest` ซึ่งรันสคริปต์ app.py จริงในหน่วยความจำ
(ไม่ต้องเปิดเซิร์ฟเวอร์ / ไม่ต้องมีเบราว์เซอร์)

รันด้วย:
    python tools/smoke_test_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"

PAGES = [
    "🏠 ภาพรวมโปรเจกต์",
    "📊 EDA (Q2)",
    "🔬 สถิติ (Q1)",
    "🤖 โมเดล (Q3)",
    "🔍 SHAP (Q4)",
    "⚖️ SHAP vs LIME (Q5)",
]


def main() -> int:
    from streamlit.testing.v1 import AppTest

    failures = 0

    # --- หน้าแรก (ค่าเริ่มต้น) ---
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    if at.exception:
        print("❌ หน้าเริ่มต้น: มี exception")
        for e in at.exception:
            print("   ", e.value)
        failures += 1
    else:
        print("✅ หน้าเริ่มต้น (ภาพรวม) รันผ่าน")

    # --- ไล่ทุกหน้า ---
    for page in PAGES:
        at = AppTest.from_file(str(APP), default_timeout=600)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        if at.exception:
            print(f"❌ {page}: มี exception")
            for e in at.exception:
                print("   ", e.value)
            failures += 1
        else:
            print(f"✅ {page} รันผ่าน")

    # --- หน้า 'ทดลองทำนาย' + กด submit ---
    try:
        at = AppTest.from_file(str(APP), default_timeout=900)
        at.run()
        at.sidebar.radio[0].set_value("🩺 ทดลองทำนาย").run()
        if at.exception:
            raise RuntimeError(at.exception[0].value)
        at.button[0].click().run()
        if at.exception:
            print("❌ 🩺 ทดลองทำนาย (กด submit): มี exception")
            for e in at.exception:
                print("   ", e.value)
            failures += 1
        else:
            print("✅ 🩺 ทดลองทำนาย + กด submit รันผ่าน")
    except Exception as exc:  # noqa: BLE001
        print("❌ 🩺 ทดลองทำนาย: ล้มเหลว")
        print("   ", exc)
        failures += 1

    print()
    if failures:
        print(f"สรุป: พบปัญหา {failures} จุด")
        return 1
    print("สรุป: dashboard ทุกหน้ารันผ่านทั้งหมด ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
