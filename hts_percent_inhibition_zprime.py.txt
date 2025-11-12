#!/usr/bin/env python3
"""
Batch %Inhibition + Z' calculator for 384-well plate matrices (CLARIOstar-style),
with replicate alignment, perimeter-well exclusion, and controls excluded only
from the side-by-side replicate export.

Outputs:
- combined_percent_inhibition.csv (Plate, Well, Signal, %Inhibition) [perimeter removed, controls kept]
- plate_qc_summary.csv (per-plate Neg/Pos stats + Z')
- combined_replicates_side_by_side.csv (replicates side-by-side; controls in col 02 & 23 removed)
- combined_replicate_stats.csv (per BasePlate & Well: N, mean/SD/CV of %Inhibition and Signal)

Assumptions:
- Each .xlsx has a single 384-well block (A..P x 1..24), row labels in col 0.
- Data block begins at the row where col 0 == "A".
- Plate filenames end with '-<rep>' (e.g., RunA-1.xlsx, RunA-2.xlsx).
"""

import os
import re
import glob
from string import ascii_uppercase
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# ---------------- USER CONFIG ----------------
NEG_CONTROL_SPANS = ["B02-H02", "B23-O23"]  # 0% inhibition reference
POS_CONTROL_SPANS = ["I02-O02"]             # 100% inhibition reference

EXCLUDE_PERIMETER = True                    # remove rows A/P and cols 1/24 globally
CLIP_PERCENT = False                        # clip %Inhibition to [0, 100]
INPUT_GLOB = "*.xlsx"                       # which files to process
# ---------------------------------------------


def expand_column_span(span: str) -> List[str]:
    """Expand a span like 'B02-H02' (same column, rows B..H) to well IDs."""
    try:
        left, right = span.split("-")
        rs, cs = left[0], int(left[1:])
        re, ce = right[0], int(right[1:])
    except Exception:
        raise ValueError(f"Bad span format: {span}. Expected like 'B02-H02'.")
    if cs != ce:
        raise ValueError(f"Span must be same column (got {span})")
    rows = ascii_uppercase[ascii_uppercase.index(rs):ascii_uppercase.index(re) + 1]
    return [f"{r}{cs:02d}" for r in rows]


def find_plate_block(df: pd.DataFrame) -> pd.DataFrame:
    """Locate and return the 384-well plate block (A..P x 1..24) as a DataFrame."""
    start_idx = None
    for i, v in df[0].items():
        if isinstance(v, str) and v.strip() == "A":
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find row label 'A' in column 0.")
    block = df.iloc[start_idx:start_idx + 16, 1:25].copy()
    if block.shape != (16, 24):
        raise ValueError(f"Detected block is {block.shape}, expected (16,24).")
    block.index = list(ascii_uppercase[:16])   # A..P
    block.columns = list(range(1, 25))         # 1..24
    return block


def melt_plate_block(block: pd.DataFrame, plate_name: str) -> pd.DataFrame:
    """Convert matrix to tidy long dataframe with (Plate, Well, Signal)."""
    recs = []
    for r in block.index:
        for c in block.columns:
            well = f"{r}{c:02d}"
            val = block.loc[r, c]
            recs.append((plate_name, well, float(val) if pd.notna(val) else np.nan))
    return pd.DataFrame(recs, columns=["Plate", "Well", "Signal"])


def filter_perimeter_wells(tidy: pd.DataFrame) -> pd.DataFrame:
    """Remove perimeter wells: rows A,P and columns 01,24 (keeps B..O and 02..23)."""
    valid_rows = list("BCDEFGHIJKLMNO")             # B..O
    valid_cols = {f"{i:02d}" for i in range(2, 24)} # 02..23
    rows = tidy["Well"].str[0]
    cols = tidy["Well"].str[1:]
    keep = rows.isin(valid_rows) & cols.isin(valid_cols)
    return tidy.loc[keep].reset_index(drop=True)


def control_stats(tidy: pd.DataFrame,
                  neg_wells: List[str],
                  pos_wells: List[str]) -> Tuple[float, float, float, float, int, int]:
    """Return (neg_mean, neg_sd, pos_mean, pos_sd, neg_n, pos_n)."""
    neg = tidy.loc[tidy["Well"].isin(neg_wells), "Signal"].astype(float)
    pos = tidy.loc[tidy["Well"].isin(pos_wells), "Signal"].astype(float)
    if neg.empty or pos.empty:
        raise ValueError("Control wells not found in plate after perimeter filtering.")
    return neg.mean(), neg.std(ddof=1), pos.mean(), pos.std(ddof=1), neg.size, pos.size


def compute_percent_inhibition(tidy: pd.DataFrame,
                               neg_mean: float,
                               pos_mean: float) -> pd.Series:
    """%Inhibition = 100 * (NegMean - Signal) / (NegMean - PosMean)"""
    denom = (neg_mean - pos_mean)
    if denom == 0 or np.isclose(denom, 0):
        return pd.Series([np.nan] * len(tidy), index=tidy.index)
    perc = 100.0 * (neg_mean - tidy["Signal"]) / denom
    if CLIP_PERCENT:
        perc = perc.clip(lower=0, upper=100)
    return perc


def z_prime(neg_mean: float, neg_sd: float, pos_mean: float, pos_sd: float) -> float:
    """Z' = 1 - 3*(SD_neg + SD_pos) / |Mean_neg - Mean_pos|"""
    denom = abs(neg_mean - pos_mean)
    if denom == 0 or np.isclose(denom, 0):
        return float("nan")
    return 1.0 - 3.0 * (neg_sd + pos_sd) / denom


def split_base_and_rep(filename: str) -> Tuple[str, int]:
    """Extract (base, rep) from 'Base-<rep>.xlsx'. If no match, rep is None."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^(.*)-(\d+)$", base)
    if not m:
        return base, None
    return m.group(1), int(m.group(2))


def main():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files:
        print(f"No files matched {INPUT_GLOB}.")
        return

    # Expand control spans once
    neg_wells = sum((expand_column_span(s) for s in NEG_CONTROL_SPANS), [])
    pos_wells = sum((expand_column_span(s) for s in POS_CONTROL_SPANS), [])

    # Warn if any configured controls would be removed by perimeter logic
    def well_is_perimeter(well: str) -> bool:
        r, c = well[0], int(well[1:])
        return (r in ("A", "P")) or (c in (1, 24))
    excluded_controls = [w for w in (neg_wells + pos_wells) if well_is_perimeter(w)]
    if excluded_controls:
        print("[WARN] The following control wells lie on the perimeter and will be excluded:")
        for w in excluded_controls:
            print("   -", w)

    all_rows = []
    qc_rows = []

    for path in files:
        try:
            plate_name = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_excel(path, header=None)
            block = find_plate_block(df)

            # Melt and optionally remove perimeter wells
            tidy = melt_plate_block(block, plate_name)
            if EXCLUDE_PERIMETER:
                before_n = len(tidy)
                tidy = filter_perimeter_wells(tidy)
                after_n = len(tidy)
                print(f"[INFO] {plate_name}: removed {before_n - after_n} perimeter wells; {after_n} kept.")

            # Controls + Z'
            n_mean, n_sd, p_mean, p_sd, n_n, p_n = control_stats(tidy, neg_wells, pos_wells)
            zprime = z_prime(n_mean, n_sd, p_mean, p_sd)

            # %Inhibition
            tidy["%Inhibition"] = compute_percent_inhibition(tidy, n_mean, p_mean)

            all_rows.append(tidy)

            qc_rows.append({
                "Plate": plate_name,
                "Neg_N": int(n_n),
                "Pos_N": int(p_n),
                "Neg_Mean": n_mean,
                "Neg_SD": n_sd,
                "Pos_Mean": p_mean,
                "Pos_SD": p_sd,
                "Delta_Mean": (n_mean - p_mean),
                "ZPrime": zprime
            })

            print(f"[OK] {plate_name}: Z'={zprime:.3f}, Neg={n_mean:.1f}±{n_sd:.1f} (n={n_n}), "
                  f"Pos={p_mean:.1f}±{p_sd:.1f} (n={p_n})")

        except Exception as e:
            print(f"[ERROR] {path}: {e}")

    if not all_rows:
        print("No plates processed.")
        return

    # ---- Save long/tidy and QC summary ----
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv("combined_percent_inhibition.csv", index=False)

    qc = pd.DataFrame(qc_rows).sort_values("Plate")
    qc.to_csv("plate_qc_summary.csv", index=False)

    # ---- Replicate-aware side-by-side table ----
    # Expect Plate like Base-1, Base-2, ...
    parts = combined["Plate"].apply(lambda x: split_base_and_rep(x))
    combined["BasePlate"] = [b for b, r in parts]
    combined["Rep"] = [r for b, r in parts]

    if combined["Rep"].isna().any():
        bad = combined.loc[combined["Rep"].isna(), "Plate"].unique()
        print("\n[WARN] These plate names do not end with '-<rep>' and won't be grouped as replicates:")
        for b in bad:
            print(f"   - {b}")

    pivot = combined.pivot_table(
        index=["BasePlate", "Well"],
        columns="Rep",
        values=["Signal", "%Inhibition"]
    ).reset_index()

    # ---- EXCLUDE CONTROL WELLS ONLY FROM SIDE-BY-SIDE EXPORT ----
    # Drop wells in column 02 or 23 (your control columns), perimeter already removed earlier
    pivot = pivot[~pivot["Well"].str[1:].isin(["02", "23"])]

    # Flatten columns: ('Signal', 1) -> 'Signal_rep1', ('%Inhibition', 2) -> 'PercentInhibition_rep2'
    def flat_name(metric, rep):
        metric_name = "PercentInhibition" if metric == "%Inhibition" else metric
        return f"{metric_name}_rep{int(rep)}" if pd.notna(rep) else f"{metric_name}"

    pivot.columns = (
        ["BasePlate", "Well"] +
        [flat_name(m, r) for (m, r) in pivot.columns[2:]]
    )
    pivot = pivot.sort_values(["BasePlate", "Well"])
    pivot.to_csv("combined_replicates_side_by_side.csv", index=False)

    # ---- Replicate stats per BasePlate & Well (controls kept) ----
    def agg_stats(group: pd.DataFrame) -> pd.Series:
        out: Dict[str, float] = {"N_reps": group["Plate"].nunique()}
        pinh = group["%Inhibition"].astype(float)
        sig = group["Signal"].astype(float)
        out.update({
            "PctInh_Mean": pinh.mean(),
            "PctInh_SD": pinh.std(ddof=1),
            "PctInh_CV": (pinh.std(ddof=1) / pinh.mean() * 100.0) if pinh.notna().any() and pinh.mean() != 0 else np.nan,
            "Signal_Mean": sig.mean(),
            "Signal_SD": sig.std(ddof=1),
            "Signal_CV": (sig.std(ddof=1) / sig.mean() * 100.0) if sig.notna().any() and sig.mean() != 0 else np.nan,
        })
        return pd.Series(out)

    rep_stats = (
        combined
        .groupby(["BasePlate", "Well"], as_index=False)
        .apply(agg_stats)
    )
    rep_stats.to_csv("combined_replicate_stats.csv", index=False)

    print("\nWrote:")
    print("  - combined_percent_inhibition.csv")
    print("  - plate_qc_summary.csv")
    print("  - combined_replicates_side_by_side.csv  [controls in col 02 & 23 removed]")
    print("  - combined_replicate_stats.csv")


if __name__ == "__main__":
    main()
