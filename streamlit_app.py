"""
Streamlit app for calculating percent inhibition and Z' QC metrics from uploaded
384-well plate Excel files, reusing the logic from percent_inhibition.py.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

import percent_inhibition as pi


st.set_page_config(
    page_title="Percent Inhibition Calculator",
    page_icon="🧪",
    layout="wide",
)

st.title("% Inhibition Calculator")
st.write(
    "Upload one or more 384-well plate Excel files (single block A–P × 1–24). "
    "The app will compute % inhibition using your configured controls, generate "
    "plate QC metrics (including Z'), and provide downloadable CSV outputs."
)


def parse_spans(span_text: str) -> List[str]:
    """Parse a comma-separated list of spans into a normalized list."""
    return [span.strip() for span in span_text.split(",") if span.strip()]


def expand_control_spans(spans: List[str]) -> List[str]:
    """Expand configured control spans to explicit well IDs."""
    wells: List[str] = []
    for span in spans:
        wells.extend(pi.expand_column_span(span))
    return wells


def process_uploaded_files(
    uploaded_files: List[Any],
    exclude_perimeter: bool,
    clip_percent: bool,
    neg_spans: List[str],
    pos_spans: List[str],
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    """Run the percent inhibition pipeline on uploaded Excel files."""
    pi.CLIP_PERCENT = clip_percent

    neg_wells = expand_control_spans(neg_spans)
    pos_wells = expand_control_spans(pos_spans)

    def well_is_perimeter(well: str) -> bool:
        row, col = well[0], int(well[1:])
        return row in ("A", "P") or col in (1, 24)

    warnings: List[str] = []
    excluded_controls = [w for w in neg_wells + pos_wells if exclude_perimeter and well_is_perimeter(w)]
    if excluded_controls:
        warnings.append(
            "These control wells are on the perimeter and will be excluded: "
            + ", ".join(excluded_controls)
        )

    all_rows: List[pd.DataFrame] = []
    qc_rows: List[Dict[str, Any]] = []
    log_messages: List[str] = []

    for uploaded in uploaded_files:
        plate_name = uploaded.name.rsplit(".", 1)[0]
        if uploaded.name.startswith("~$"):
            warnings.append(f"{uploaded.name}: skipped temporary Excel lock file.")
            continue

        try:
            uploaded.seek(0)
            df = pd.read_excel(uploaded, header=None)
            block = pi.find_plate_block(df)

            tidy = pi.melt_plate_block(block, plate_name)
            if exclude_perimeter:
                before = len(tidy)
                tidy = pi.filter_perimeter_wells(tidy)
                removed = before - len(tidy)
                log_messages.append(f"{plate_name}: removed {removed} perimeter wells; {len(tidy)} kept.")

            n_mean, n_sd, p_mean, p_sd, n_n, p_n = pi.control_stats(tidy, neg_wells, pos_wells)
            zprime = pi.z_prime(n_mean, n_sd, p_mean, p_sd)

            tidy["%Inhibition"] = pi.compute_percent_inhibition(tidy, n_mean, p_mean)
            all_rows.append(tidy)

            qc_rows.append(
                {
                    "Plate": plate_name,
                    "Neg_N": int(n_n),
                    "Pos_N": int(p_n),
                    "Neg_Mean": n_mean,
                    "Neg_SD": n_sd,
                    "Pos_Mean": p_mean,
                    "Pos_SD": p_sd,
                    "Delta_Mean": n_mean - p_mean,
                    "ZPrime": zprime,
                }
            )

            log_messages.append(
                f"{plate_name}: Z'={zprime:.3f}, Neg={n_mean:.1f}±{n_sd:.1f} (n={n_n}), "
                f"Pos={p_mean:.1f}±{p_sd:.1f} (n={p_n})"
            )
        except Exception as exc:  # noqa: BLE001 - surface full error in UI
            warnings.append(f"{plate_name}: {exc}")

    if not all_rows:
        raise ValueError("No plates processed successfully.")

    combined = pd.concat(all_rows, ignore_index=True)
    qc = pd.DataFrame(qc_rows).sort_values("Plate")

    # Split base plate and replicate index
    parts = combined["Plate"].apply(lambda x: pi.split_base_and_rep(x))
    combined["BasePlate"] = [base for base, _ in parts]
    combined["Rep"] = [rep for _, rep in parts]

    bad_reps = combined.loc[combined["Rep"].isna(), "Plate"].unique()
    if len(bad_reps) > 0:
        warnings.append(
            "The following plate names do not end with '-<rep>' and will not pivot side-by-side: "
            + ", ".join(sorted(bad_reps))
        )

    pivot = combined.pivot_table(
        index=["BasePlate", "Well"],
        columns="Rep",
        values=["Signal", "%Inhibition"],
    ).reset_index()
    pivot = pivot[~pivot["Well"].str[1:].isin(["02", "23"])]

    def flat_name(metric: str, rep: Any) -> str:
        metric_name = "PercentInhibition" if metric == "%Inhibition" else metric
        return f"{metric_name}_rep{int(rep)}" if pd.notna(rep) else metric_name

    pivot.columns = ["BasePlate", "Well"] + [flat_name(m, r) for (m, r) in pivot.columns[2:]]
    pivot = pivot.sort_values(["BasePlate", "Well"])

    def agg_stats(group: pd.DataFrame) -> pd.Series:
        pinh = group["%Inhibition"].astype(float)
        sig = group["Signal"].astype(float)
        out = {
            "N_reps": group["Plate"].nunique(),
            "PctInh_Mean": pinh.mean(),
            "PctInh_SD": pinh.std(ddof=1),
            "PctInh_CV": (pinh.std(ddof=1) / pinh.mean() * 100.0)
            if pinh.notna().any() and pinh.mean() != 0
            else np.nan,
            "Signal_Mean": sig.mean(),
            "Signal_SD": sig.std(ddof=1),
            "Signal_CV": (sig.std(ddof=1) / sig.mean() * 100.0)
            if sig.notna().any() and sig.mean() != 0
            else np.nan,
        }
        return pd.Series(out)

    rep_stats = (
        combined.groupby(["BasePlate", "Well"], as_index=False)
        .apply(agg_stats, include_groups=False)
        .reset_index(drop=True)
    )

    outputs = {
        "combined_percent_inhibition.csv": combined,
        "plate_qc_summary.csv": qc,
        "combined_replicates_side_by_side.csv": pivot,
        "combined_replicate_stats.csv": rep_stats,
    }

    return outputs, log_messages, warnings


uploaded_files = st.file_uploader(
    "Upload Excel files",
    type=["xlsx"],
    accept_multiple_files=True,
    help="Select one or more .xlsx files containing a single 384-well block (A–P × 1–24).",
)

with st.expander("Processing options", expanded=False):
    exclude_perimeter = st.checkbox(
        "Exclude perimeter wells (rows A/P and columns 1/24)",
        value=pi.EXCLUDE_PERIMETER,
    )
    clip_percent = st.checkbox(
        "Clip % inhibition to [0, 100]",
        value=pi.CLIP_PERCENT,
    )
    neg_span_text = st.text_input(
        "Negative control spans",
        ", ".join(pi.NEG_CONTROL_SPANS),
        help="Comma-separated spans like B02-H02 (inclusive).",
    )
    pos_span_text = st.text_input(
        "Positive control spans",
        ", ".join(pi.POS_CONTROL_SPANS),
        help="Comma-separated spans like I02-O02 (inclusive).",
    )

if uploaded_files:
    if st.button("Run analysis", type="primary"):
        with st.spinner("Processing plates..."):
            try:
                outputs, logs, warnings = process_uploaded_files(
                    uploaded_files=uploaded_files,
                    exclude_perimeter=exclude_perimeter,
                    clip_percent=clip_percent,
                    neg_spans=parse_spans(neg_span_text),
                    pos_spans=parse_spans(pos_span_text),
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Processing failed: {exc}")
            else:
                if warnings:
                    for msg in warnings:
                        st.warning(msg)
                st.success("Processing complete.")
                if logs:
                    st.markdown("\n".join(f"- {msg}" for msg in logs))

                combined_df = outputs["combined_percent_inhibition.csv"]
                percent_values = combined_df["%Inhibition"].astype(float)
                valid = percent_values.dropna()
                if not valid.empty:
                    hist, bins = np.histogram(valid, bins=30)
                    bin_midpoints = (bins[:-1] + bins[1:]) / 2
                    hist_df = pd.DataFrame({"BinMid": bin_midpoints, "Count": hist})
                    st.subheader("Distribution of % Inhibition")
                    st.bar_chart(hist_df.set_index("BinMid"))

                tabs = st.tabs(
                    [
                        "Combined % Inhibition",
                        "Plate QC Summary",
                        "Replicates Side-by-Side",
                        "Replicate Stats",
                    ]
                )

                for tab, (filename, df) in zip(tabs, outputs.items()):
                    with tab:
                        st.subheader(filename)
                        st.dataframe(df.head(50))
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"Download {filename}",
                            data=csv_bytes,
                            file_name=filename,
                            mime="text/csv",
                        )
else:
    st.info("Upload one or more Excel files to begin.")

