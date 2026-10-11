"""
execute_notebooks.py
====================
รัน notebook ทุกเล่มเพื่อฝัง output ลงในไฟล์ .ipynb
(ทำให้เปิดดูแล้วเห็นผลลัพธ์ทันที ไม่ต้องรันเอง)

รันด้วย:
    python tools/execute_notebooks.py            # รันทุกเล่ม
    python tools/execute_notebooks.py 01 02      # รันเฉพาะเล่มที่ระบุ (ตาม prefix)

หมายเหตุ: ควรรัน `python -m src.run_pipeline` ให้เสร็จก่อน
เพราะ notebook 03–05 อ่านผลลัพธ์จาก reports/
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def execute_notebook(path: Path, timeout: int = 2400) -> bool:
    print(f"-> กำลังรัน {path.name} ...", flush=True)
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"   ❌ ล้มเหลว: {exc}")
        return False

    nbformat.write(nb, path)
    print(f"   ✅ สำเร็จ ({len(nb.cells)} cells) และบันทึก output แล้ว")
    return True


def main(argv: list[str]) -> int:
    prefixes = argv or [""]
    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    targets = [nb for nb in notebooks if any(nb.name.startswith(p) for p in prefixes)]

    if not targets:
        print("ไม่พบ notebook ที่จะรัน")
        return 1

    failures = 0
    for nb in targets:
        if not execute_notebook(nb):
            failures += 1

    print()
    print(f"รันเสร็จ {len(targets) - failures}/{len(targets)} เล่ม")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
