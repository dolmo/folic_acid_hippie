"""
Patch metadata.csv and labels.csv with well-level condition and sex labels.

Reads:
  /home/jesus/datasets_hippie/FA_T4/metadata.csv   (97,525 rows, condition='unknown')

Writes:
  /home/jesus/datasets_hippie/FA_T4/metadata.csv   (condition + sex + div columns added)
  /home/jesus/datasets_hippie/FA_T4/labels.csv     (condition column updated)

Legend from FolicAcid_T4_02252025_SA.tsv:
  A = Control (2mg)
  B = FA deficiency (0mg)
  C = FA excess (10mg)
  D = FA Super Excess (20mg)
  E = Folinic Acid Excess
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import METADATA_CSV, LABELS_CSV

META_PATH   = METADATA_CSV
LABELS_PATH = LABELS_CSV

# ── Well-level condition + sex mapping ──────────────────────────────────────
# Key: (mouse_id, well_index)   well_index = int("well000"[-3:]) = 0..5
# Value: (condition_label, sex)

WELL_CONDITION = {
    # M07708: A_Male, B_Male, D_Male, A_Female, B_Female, D_Female
    ("M07708", 0): ("2mg_control",          "Male"),
    ("M07708", 1): ("0mg_deficient",        "Male"),
    ("M07708", 2): ("20mg_super_excess",    "Male"),
    ("M07708", 3): ("2mg_control",          "Female"),
    ("M07708", 4): ("0mg_deficient",        "Female"),
    ("M07708", 5): ("20mg_super_excess",    "Female"),

    # M07137: A_Male, B_Male, D_Male, E_Male, A_Female, B_Female
    ("M07137", 0): ("2mg_control",          "Male"),
    ("M07137", 1): ("0mg_deficient",        "Male"),
    ("M07137", 2): ("20mg_super_excess",    "Male"),
    ("M07137", 3): ("folinic_acid_excess",  "Male"),
    ("M07137", 4): ("2mg_control",          "Female"),
    ("M07137", 5): ("0mg_deficient",        "Female"),

    # M08092: A_Male, B_Male, D_Male, E_Male, D_Female, E_Female
    ("M08092", 0): ("2mg_control",          "Male"),
    ("M08092", 1): ("0mg_deficient",        "Male"),
    ("M08092", 2): ("20mg_super_excess",    "Male"),
    ("M08092", 3): ("folinic_acid_excess",  "Male"),
    ("M08092", 4): ("20mg_super_excess",    "Female"),
    ("M08092", 5): ("folinic_acid_excess",  "Female"),

    # M07865: E_Male, A_Female, B_Female, D_Female, E_Female, E_Female
    ("M07865", 0): ("folinic_acid_excess",  "Male"),
    ("M07865", 1): ("2mg_control",          "Female"),
    ("M07865", 2): ("0mg_deficient",        "Female"),
    ("M07865", 3): ("20mg_super_excess",    "Female"),
    ("M07865", 4): ("folinic_acid_excess",  "Female"),
    ("M07865", 5): ("folinic_acid_excess",  "Female"),

    # M08068: C_Male, E_Female, C_Male, C_Male, E_Female, E_Male
    ("M08068", 0): ("10mg_excess",          "Male"),
    ("M08068", 1): ("folinic_acid_excess",  "Female"),
    ("M08068", 2): ("10mg_excess",          "Male"),
    ("M08068", 3): ("10mg_excess",          "Male"),
    ("M08068", 4): ("folinic_acid_excess",  "Female"),
    ("M08068", 5): ("folinic_acid_excess",  "Male"),

    # M08032: C_Male, E_Male, C_Male, C_Male, E_Male, E_Female
    ("M08032", 0): ("10mg_excess",          "Male"),
    ("M08032", 1): ("folinic_acid_excess",  "Male"),
    ("M08032", 2): ("10mg_excess",          "Male"),
    ("M08032", 3): ("10mg_excess",          "Male"),
    ("M08032", 4): ("folinic_acid_excess",  "Male"),
    ("M08032", 5): ("folinic_acid_excess",  "Female"),
}

# ── DIV mapping ─────────────────────────────────────────────────────────────
# Key: (date_str_YYMMDD, mouse_id)  e.g. ("250228", "M07708")
# Value: DIV (int)

DATE_MOUSE_DIV = {
    # 2/28/2025 = DIV 3 for main cohort
    ("250228", "M07708"): 3,
    ("250228", "M07137"): 3,
    ("250228", "M08092"): 3,
    ("250228", "M07865"): 3,
    # 3/3/2025 = DIV 6
    ("250303", "M07708"): 6,
    ("250303", "M07137"): 6,
    ("250303", "M08092"): 6,
    ("250303", "M07865"): 6,
    # 3/6/2025 = DIV 9
    ("250306", "M07708"): 9,
    ("250306", "M07137"): 9,
    ("250306", "M08092"): 9,
    ("250306", "M07865"): 9,
    # 3/10/2025 = DIV 13
    ("250310", "M07708"): 13,
    ("250310", "M07137"): 13,
    ("250310", "M08092"): 13,
    ("250310", "M07865"): 13,
    # 3/13/2025 = DIV 16
    ("250313", "M07708"): 16,
    ("250313", "M07137"): 16,
    ("250313", "M08092"): 16,
    ("250313", "M07865"): 16,
    # 3/17/2025 = DIV 20 main cohort / DIV 5 new plates
    ("250317", "M07708"): 20,
    ("250317", "M07137"): 20,
    ("250317", "M07865"): 20,
    ("250317", "M08092"): 20,
    ("250317", "M08068"):  5,
    ("250317", "M08032"):  5,
    # 3/20/2025 = DIV 23 main cohort / DIV 8 new plates
    ("250320", "M07708"): 23,
    ("250320", "M07137"): 23,
    ("250320", "M08092"): 23,
    ("250320", "M07865"): 23,
    ("250320", "M08068"):  8,
    ("250320", "M08032"):  8,
    # ── New T4 dates (FA_remaining) ──────────────────────────────────────
    # 3/24/2025 = DIV 27 main / DIV 12 late
    ("250324", "M07708"): 27,
    ("250324", "M07137"): 27,
    ("250324", "M08092"): 27,
    ("250324", "M07865"): 27,
    ("250324", "M08068"): 12,
    ("250324", "M08032"): 12,
    # 3/27/2025 = DIV 30 main / DIV 15 late
    ("250327", "M07708"): 30,
    ("250327", "M07137"): 30,
    ("250327", "M08092"): 30,
    ("250327", "M07865"): 30,
    ("250327", "M08068"): 15,
    ("250327", "M08032"): 15,
    # 3/31/2025 = DIV 34 main / DIV 19 late
    ("250331", "M07708"): 34,
    ("250331", "M07137"): 34,
    ("250331", "M08092"): 34,
    ("250331", "M07865"): 34,
    ("250331", "M08068"): 19,
    ("250331", "M08032"): 19,
    # 4/3/2025 = DIV 22 (late cohort only)
    ("250403", "M08068"): 22,
    ("250403", "M08032"): 22,
    # 4/7/2025 = DIV 26
    ("250407", "M08068"): 26,
    ("250407", "M08032"): 26,
    # 4/10/2025 = DIV 29
    ("250410", "M08068"): 29,
    ("250410", "M08032"): 29,
    # 4/14/2025 = DIV 33
    ("250414", "M08068"): 33,
    ("250414", "M08032"): 33,
}


def main():
    print(f"Loading {META_PATH}")
    meta = pd.read_csv(META_PATH)
    print(f"  {len(meta)} rows, columns: {list(meta.columns)}")

    # ── Extract well index from "well000" → 0 ──────────────────────────────
    well_idx = meta["well"].str.extract(r"(\d+)$").astype(int).squeeze()

    # ── Map (mouse_id, well_idx) → condition, sex ──────────────────────────
    conditions = []
    sexes      = []
    for mid, widx in zip(meta["mouse_id"], well_idx):
        key = (mid, widx)
        cond, sex = WELL_CONDITION.get(key, ("unknown", "unknown"))
        conditions.append(cond)
        sexes.append(sex)

    meta["condition"] = conditions
    meta["sex"]       = sexes

    # ── Map (date_str, mouse_id) → DIV ─────────────────────────────────────
    divs = []
    for date_str, mid in zip(meta["date"], meta["mouse_id"]):
        divs.append(DATE_MOUSE_DIV.get((str(date_str), mid), -1))

    meta["div"] = divs

    # ── Save ────────────────────────────────────────────────────────────────
    meta.to_csv(META_PATH, index=False)
    print(f"Saved updated metadata.csv")

    labels_df = pd.DataFrame({"label": meta["condition"].values})
    labels_df.to_csv(LABELS_PATH, index=False)
    print(f"Saved updated labels.csv")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\nUnits per condition:")
    print(meta["condition"].value_counts().to_string())
    print("\nUnits per sex:")
    print(meta["sex"].value_counts().to_string())
    print("\nDIV coverage:")
    print(meta.groupby(["mouse_id", "div"]).size().to_string())
    print("\nUnknown condition rows:")
    unk = meta[meta["condition"] == "unknown"]
    if len(unk):
        print(unk[["mouse_id", "well", "date"]].value_counts().to_string())
    else:
        print("  None — all wells mapped successfully.")


if __name__ == "__main__":
    main()
