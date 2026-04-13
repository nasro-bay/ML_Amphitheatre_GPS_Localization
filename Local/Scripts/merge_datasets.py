#!/usr/bin/env python3
"""
Merge `brahimi_original.xlsx` into `ensia_gps_data .csv`.

Important behavior:
- Output CSV keeps EXACTLY the same schema and column order as the existing
  `ensia_gps_data .csv` file.
- No synthetic metadata fields are generated.
- Existing rows are preserved, Brahimi rows are appended.
"""

from pathlib import Path
import shutil
from datetime import datetime, timezone
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
ENSIA_CSV_PATH = DATA_DIR / "ensia_gps_data .csv"
BRAHIMI_XLSX_PATH = DATA_DIR / "brahimi_original.xlsx"
BACKUP_CSV_PATH = DATA_DIR / "ensia_gps_data_v1.csv"


def create_backup() -> Path:
    """Create/overwrite the backup copy of ENSIA CSV."""
    if not ENSIA_CSV_PATH.exists():
        raise FileNotFoundError(f"ENSIA CSV not found: {ENSIA_CSV_PATH}")
    shutil.copyfile(ENSIA_CSV_PATH, BACKUP_CSV_PATH)
    print(f"Backup created: {BACKUP_CSV_PATH}")
    return BACKUP_CSV_PATH


def normalize_amphitheatre_name(raw_name: object) -> str:
    """Convert 'Amphitheater X' to 'Amphi X' to match ENSIA naming."""
    if pd.isna(raw_name):
        return "Outside"
    text = str(raw_name).strip()
    if "Amphitheater" in text:
        return text.replace("Amphitheater", "Amphi").strip()
    return text


def get_next_ids(df_ensia: pd.DataFrame, count: int) -> list[int]:
    """Generate sequential ids continuing from max existing id."""
    max_id = pd.to_numeric(df_ensia.get("id"), errors="coerce").max()
    start = 1 if pd.isna(max_id) else int(max_id) + 1
    return list(range(start, start + count))


def build_brahimi_rows(df_ensia: pd.DataFrame, df_brahimi: pd.DataFrame) -> pd.DataFrame:
    """Create rows from Brahimi data using ENSIA schema as source of truth."""
    expected_excel_cols = {"name", "location_lat", "location_lng"}
    missing = expected_excel_cols - set(df_brahimi.columns)
    if missing:
        raise ValueError(f"Brahimi file missing required columns: {sorted(missing)}")

    ensia_columns = list(df_ensia.columns)
    now = datetime.now(timezone.utc).isoformat()
    new_ids = get_next_ids(df_ensia, len(df_brahimi))

    records = []
    for idx, row in enumerate(df_brahimi.itertuples(index=False)):
        # Start with empty values for ALL ENSIA columns to preserve schema exactly.
        rec = {col: pd.NA for col in ensia_columns}

        rec["id"] = new_ids[idx]
        rec["year"] = pd.NA
        rec["section"] = pd.NA
        rec["user"] = "brahimi_merge"
        rec["amphitheatre"] = normalize_amphitheatre_name(getattr(row, "name"))
        rec["module"] = "Brahimi Original"
        rec["seat_block"] = pd.NA
        rec["seat_row"] = pd.NA
        rec["seat_column"] = pd.NA
        rec["latitude_mean"] = float(getattr(row, "location_lat"))
        rec["longitude_mean"] = float(getattr(row, "location_lng"))
        rec["accuracy_mean"] = pd.NA
        rec["gps_variance"] = pd.NA
        rec["is_outside"] = False
        rec["sample_count"] = 1
        rec["raw_gps_readings"] = "[]"
        rec["collection_metadata"] = "{}"
        rec["navigator_context"] = "{}"
        rec["screen_context"] = "{}"
        rec["network_information"] = "{}"
        rec["battery_status"] = "{}"
        rec["timestamp"] = now
        rec["created_at"] = now
        rec["device_info"] = "brahimi_original.xlsx"

        records.append(rec)

    return pd.DataFrame(records, columns=ensia_columns)


def merge_and_save(df_ensia: pd.DataFrame, df_brahimi_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new rows and save CSV with the exact original schema."""
    merged = pd.concat([df_ensia, df_brahimi_rows], ignore_index=True)
    merged = merged[df_ensia.columns]  # enforce exact schema + order
    merged.to_csv(ENSIA_CSV_PATH, index=False)
    return merged


def main() -> None:
    print("Starting ENSIA/Brahimi merge...")
    backup_path = create_backup()

    df_ensia = pd.read_csv(ENSIA_CSV_PATH)
    df_brahimi = pd.read_excel(BRAHIMI_XLSX_PATH)
    print(f"Loaded ENSIA rows: {len(df_ensia)}")
    print(f"Loaded Brahimi rows: {len(df_brahimi)}")

    df_brahimi_rows = build_brahimi_rows(df_ensia, df_brahimi)
    merged = merge_and_save(df_ensia, df_brahimi_rows)

    print("Merge complete.")
    print(f"Backup file: {backup_path}")
    print(f"Rows added: {len(df_brahimi_rows)}")
    print(f"Total rows after merge: {len(merged)}")
    print(f"Output: {ENSIA_CSV_PATH}")


if __name__ == "__main__":
    main()