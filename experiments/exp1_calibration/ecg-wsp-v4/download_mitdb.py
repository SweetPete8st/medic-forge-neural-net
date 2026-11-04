"""Download the held-out evaluation data to David's machine.

Usage (from the ecg-wsp folder, with wfdb installed: pip install wfdb):

    python download_mitdb.py

Fetches from PhysioNet:
  - mitdb : MIT-BIH Arrhythmia Database, 48 records, 360 Hz, beat + rhythm
            annotations (.atr)
  - afdb  : MIT-BIH Atrial Fibrillation Database, 23 records, 250 Hz, rhythm
            annotations (.atr)

Both are public, de-identified, and the standard benchmarks for beat detection
and AF detection. About 110 MB total.
"""
import os
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

for db in ("mitdb", "afdb"):
    dest = os.path.join(OUT, db)
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {db} to {dest} ...")
    wfdb.dl_database(db, dl_dir=dest)
    print(f"  done: {len(os.listdir(dest))} files")

print("Finished. Record 100 is the development record; everything else is held out.")
