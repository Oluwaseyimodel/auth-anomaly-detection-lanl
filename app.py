"""
SOC Triage Framework — Evaluation Dashboard (Streamlit)

Run with:
    streamlit run app.py

Expects to sit inside (or be pointed at) your project folder, e.g.:
    auth_anomaly_detection_LANL/
        app.py                                      <- this file
        siem_metrics.csv
        contamination_sensitivity_FINE.csv            <- real 0.005-0.030 sweep (matches Table 5)
        cross_validation_results.csv                 <- per-fold detail, one setting
        cross_validation_ACROSS_SETTINGS.csv         <- 3-setting summary (mean/std)
        shap_phase3_metrics.csv
        shap_feature_importance.csv
        precision_at_k.csv
        final_integrated_table.csv

If a CSV is missing, the dashboard falls back to the values reconstructed
from the dissertation's tables so it still runs end to end. Replace the
DATA_DIR constant below to point at your actual project folder.
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("SOC_DATA_DIR", ".")

CONFIG_COLORS = {
    "SIEM Rules": "#60A5FA",
    "IF Only": "#F87171",
    "IF+SHAP": "#F5A524",
    "IF+SHAP+MITRE": "#34D399",
}

st.set_page_config(page_title="SOC Triage Framework", layout="wide", page_icon=":shield:")

# ---------------------------------------------------------------------------
# DATA LOADING — tries the real CSV first, falls back to reconstructed values
# ---------------------------------------------------------------------------

def load_csv(filename, fallback_df):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Found {filename} but couldn't parse it ({e}). Using fallback data.")
    return fallback_df


def load_csv_first_match(filenames, fallback_df):
    """Try a list of candidate filenames in order, return the first that exists and parses."""
    for filename in filenames:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            try:
                return pd.read_csv(path), filename
            except Exception:
                continue
    return fallback_df, None


siem_fallback = pd.DataFrame([
    {"metric": "Precision", "value": 0.4599},
    {"metric": "Recall", "value": 0.8922},
    {"metric": "F1", "value": 0.6069},
    {"metric": "FPR", "value": 0.0123},
])

contamination_fallback = pd.DataFrame([
    {"contamination": 0.005, "alerts": 14, "precision": 0.7143, "recall": 0.3030, "f1": 0.4255, "fpr": 0.0014},
    {"contamination": 0.008, "alerts": 27, "precision": 0.6296, "recall": 0.5152, "f1": 0.5667, "fpr": 0.0035},
    {"contamination": 0.010, "alerts": 29, "precision": 0.6207, "recall": 0.5455, "f1": 0.5806, "fpr": 0.0039},
    {"contamination": 0.012, "alerts": 37, "precision": 0.5946, "recall": 0.6667, "f1": 0.6286, "fpr": 0.0053},
    {"contamination": 0.015, "alerts": 46, "precision": 0.6087, "recall": 0.8485, "f1": 0.7089, "fpr": 0.0063},
    {"contamination": 0.020, "alerts": 64, "precision": 0.4688, "recall": 0.9091, "f1": 0.6186, "fpr": 0.0119},
    {"contamination": 0.025, "alerts": 79, "precision": 0.3924, "recall": 0.9394, "f1": 0.5536, "fpr": 0.0168},
    {"contamination": 0.030, "alerts": 89, "precision": 0.3483, "recall": 0.9394, "f1": 0.5082, "fpr": 0.0203},
])

cv_fallback = pd.DataFrame([
    {"contamination": "0.010", "mean_f1": 0.6829, "std": 0.0594},
    {"contamination": "0.012", "mean_f1": 0.7108, "std": 0.0224},
    {"contamination": "0.015", "mean_f1": 0.7111, "std": 0.0500},
])

shap_fallback = pd.DataFrame([
    {"metric": "Latency (ms/alert)", "value": 4.72},
    {"metric": "Top-3 feature stability (%)", "value": 100.0},
    {"metric": "Consistency (CV)", "value": 0.0},
])

precision_k_fallback = pd.DataFrame([
    {"k": 10, "precision": 0.70},
    {"k": 50, "precision": 0.50},
    {"k": 100, "precision": 0.58},
    {"k": 167, "precision": 0.7006},
])

tiers_fallback = pd.DataFrame([
    {"tier": "MEDIUM-LOW", "range": "7-19 dest/day", "alerts": 4017, "precision": 39.8},
    {"tier": "CRITICAL", "range": ">=30 dest/day", "alerts": 1027, "precision": 77.9},
    {"tier": "HIGH", "range": "20-29 dest/day", "alerts": 4537, "precision": 97.0},
])

integrated_fallback = pd.DataFrame([
    {"metric": "Precision", "SIEM Rules": 0.4599, "IF Only": 0.7154, "IF+SHAP": 0.7154, "IF+SHAP+MITRE": 0.7154},
    {"metric": "Recall", "SIEM Rules": 0.8922, "IF Only": 0.7187, "IF+SHAP": 0.7187, "IF+SHAP+MITRE": 0.7187},
    {"metric": "F1", "SIEM Rules": 0.6069, "IF Only": 0.7108, "IF+SHAP": 0.7108, "IF+SHAP+MITRE": 0.7108},
    {"metric": "FPR", "SIEM Rules": 0.0123, "IF Only": 0.0053, "IF+SHAP": 0.0053, "IF+SHAP+MITRE": 0.0053},
])

operational_fallback = pd.DataFrame([
    {"metric": "Mean time to triage (min)", "SIEM Rules": 8, "IF Only": 10, "IF+SHAP": 5, "IF+SHAP+MITRE": 3},
    {"metric": "Analyst hours/day", "SIEM Rules": 43.2, "IF Only": 28.8, "IF+SHAP": 14.4, "IF+SHAP+MITRE": 8.7},
    {"metric": "Explanation usefulness (/15)", "SIEM Rules": 7, "IF Only": 3, "IF+SHAP": 13, "IF+SHAP+MITRE": 15},
    {"metric": "Analyst confidence (/5)", "SIEM Rules": 2.0, "IF Only": 1.8, "IF+SHAP": 3.7, "IF+SHAP+MITRE": 4.1},
])

def pick_col(df, candidates):
    """Case-insensitive match: return the real column name in df matching any candidate, else None."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def to_metric_value(df, metric_cols=("precision", "recall", "f1", "fpr"), row_filter_col=None, row_filter_value=None):
    """
    Normalize a metrics dataframe into two columns: metric, value.
    Handles both:
      - long format already: has 'metric' and 'value' columns -> returned as-is
      - wide format: one row (optionally selected by row_filter_col/value) with
        precision/recall/f1/fpr etc as separate columns -> melted into metric/value
    """
    metric_col = pick_col(df, ["metric"])
    value_col = pick_col(df, ["value"])
    if metric_col and value_col:
        return df[[metric_col, value_col]].rename(columns={metric_col: "metric", value_col: "value"})

    row = df
    if row_filter_col and row_filter_value:
        fcol = pick_col(df, [row_filter_col])
        if fcol:
            matched = df[df[fcol].astype(str).str.contains(row_filter_value, case=False, na=False)]
            if not matched.empty:
                row = matched

    present = [pick_col(df, [c]) for c in metric_cols]
    present = [c for c in present if c is not None]
    if not present:
        raise ValueError(
            f"None of the expected columns {metric_cols} were found. "
            f"Actual columns: {list(df.columns)}"
        )
    first_row = row.iloc[0]
    return pd.DataFrame({"metric": [c.capitalize() for c in present], "value": [first_row[c] for c in present]})


siem_df_raw = load_csv("siem_metrics.csv", siem_fallback)
contamination_df, contamination_source = load_csv_first_match(
    [
        "contamination_sensitivity_FINE.csv",       # real 0.005-0.030 sweep matching Table 5
        "contamination_sensitivity_FINAL.csv",      # coarse fallback (0.05-0.30), not report-accurate
    ],
    contamination_fallback,
)

# cross_validation_results.csv (fold, precision, recall, mean_f1, fpr) is PER-FOLD data for
# one contamination setting, not the 3-setting summary. The summary (if it exists) likely
# lives in a differently-named file — try a few likely candidates before falling back.
cv_fold_df = load_csv("cross_validation_results.csv", None)
cv_summary_df, cv_summary_source = load_csv_first_match(
    [
        "cross_validation_ACROSS_SETTINGS.csv",
        "cross_validation_across_settings.csv",
        "cv_summary.csv",
        "cross_validation_summary.csv",
    ],
    None,
)

shap_df_raw = load_csv("shap_phase3_metrics.csv", shap_fallback)
precision_k_df_raw = load_csv("precision_at_k.csv", precision_k_fallback)
tiers_df_raw = load_csv("attack_tiers.csv", tiers_fallback)
integrated_df_raw, integrated_source = load_csv_first_match(
    ["final_integrated_table.csv", "integrated_comparison_table.csv"],
    integrated_fallback,
)

# Normalize possibly-wide real CSVs into the long metric/value shape charts expect.
try:
    siem_df = to_metric_value(siem_df_raw, row_filter_col="system", row_filter_value="SIEM")
except Exception as e:
    siem_df = siem_fallback

# contamination_df is already expected in wide form with contamination/precision/recall/f1/fpr columns;
# just normalize column name casing so downstream .loc lookups don't fail.
_rename = {}
for target in ["contamination", "precision", "recall", "f1", "fpr", "alerts"]:
    found = pick_col(contamination_df, [target, target + "_fired" if target == "alerts" else target])
    if found and found != target:
        _rename[found] = target
contamination_df = contamination_df.rename(columns=_rename)

cv_summary_error = None
if cv_summary_df is not None:
    cv_col = pick_col(cv_summary_df, ["contamination", "setting"])
    fold_col_in_summary = pick_col(cv_summary_df, ["fold"])
    f1_col = pick_col(cv_summary_df, ["mean_f1", "meanf1", "mean cv f1", "f1", "mean"])
    std_col = pick_col(cv_summary_df, ["std", "std_dev", "stddev"])

    if cv_col and fold_col_in_summary and f1_col:
        # Real shape: per-fold-per-setting long table (contamination, fold, precision,
        # recall, f1, fpr) — compute mean/std ourselves rather than expecting the
        # notebook to have pre-aggregated it. This reproduces Table 6 exactly.
        grouped = cv_summary_df.groupby(cv_col)[f1_col].agg(["mean", "std"]).reset_index()
        cv_df = grouped.rename(columns={cv_col: "contamination", "mean": "mean_f1", "std": "std"})
        cv_mode = "summary"
    elif cv_col and f1_col and std_col:
        # Already-aggregated shape: contamination, mean(_f1), std
        cv_df = cv_summary_df.rename(columns={cv_col: "contamination", f1_col: "mean_f1", std_col: "std"})
        cv_mode = "summary"
    else:
        cv_df = None
        cv_mode = "fold"
        cv_summary_error = (
            f"Found {cv_summary_source}, but couldn't match its columns. "
            f"Columns found: {list(cv_summary_df.columns)}"
        )
else:
    cv_df = None
    cv_mode = "fold"

try:
    prec_col = pick_col(precision_k_df_raw, ["precision_at_k", "precision"])
    k_col = pick_col(precision_k_df_raw, ["k"])
    precision_k_df = precision_k_df_raw.rename(columns={prec_col: "precision", k_col: "k"})
except Exception:
    precision_k_df = precision_k_fallback

try:
    metric_col = pick_col(integrated_df_raw, ["metric", "Metric"])
    integrated_df = integrated_df_raw.rename(columns={metric_col: "metric"})
    # map display config names to whatever the real column headers actually are
    CONFIG_COLUMN_MAP = {}
    for display_name, aliases in {
        "SIEM Rules": ["SIEM Rules", "Rules", "SIEM"],
        "IF Only": ["IF Only", "IF"],
        "IF+SHAP": ["IF+SHAP"],
        "IF+SHAP+MITRE": ["IF+SHAP+MITRE", "IF+SHAP+ATT&CK", "IF+SHAP+ATTACK"],
    }.items():
        found = pick_col(integrated_df, aliases)
        if found:
            CONFIG_COLUMN_MAP[display_name] = found

    # Your real file bundles 0-1 detection ratios (precision/recall/f1/fpr) together
    # with very differently-scaled operational numbers (alert volume in the hundreds,
    # analyst hours up to 40+, explanation quality /15, etc) in the same table. Split
    # them automatically by looking at the actual values, rather than guessing labels,
    # so each group gets a chart with an appropriate axis instead of one shared 0-1 axis
    # that clips everything larger than 1.
    config_cols_present = list(CONFIG_COLUMN_MAP.values())
    row_max = integrated_df[config_cols_present].max(axis=1)
    is_ratio_row = row_max <= 1.05  # small tolerance for values like 1.0 exactly

    detection_df = integrated_df[is_ratio_row].reset_index(drop=True)
    operational_real_df = integrated_df[~is_ratio_row].reset_index(drop=True)

    if operational_real_df.empty:
        # file only had detection ratios — keep using the reconstructed operational numbers
        operational_df = operational_fallback
    else:
        operational_df = operational_real_df
except Exception:
    integrated_df = integrated_fallback
    detection_df = integrated_fallback
    CONFIG_COLUMN_MAP = {k: k for k in CONFIG_COLORS}
    operational_df = operational_fallback

try:
    shap_df = to_metric_value(
        shap_df_raw,
        metric_cols=["latency", "top3_stability", "consistency", "coefficient_of_variation"],
    )
except Exception:
    shap_df = shap_fallback

try:
    tier_col = pick_col(tiers_df_raw, ["tier"])
    alerts_col = pick_col(tiers_df_raw, ["alerts", "alert_count"])
    prec_col = pick_col(tiers_df_raw, ["precision"])
    tiers_df = tiers_df_raw.rename(columns={tier_col: "tier", alerts_col: "alerts", prec_col: "precision"})
    # precision may be stored as 0-1 fraction rather than 0-100 percent — normalize to percent
    if tiers_df["precision"].max() <= 1.0:
        tiers_df["precision"] = tiers_df["precision"] * 100
except Exception:
    tiers_df = tiers_fallback

# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------

st.markdown(
    "<h2 style='margin-bottom:0;'>SOC Triage Framework — Evaluation Console</h2>"
    "<p style='color:#7D8590;font-family:monospace;font-size:0.85rem;'>"
    "14,416 user-day profiles &middot; 1.16% attack base rate &middot; LANL auth logs</p>",
    unsafe_allow_html=True,
)

tab_overview, tab_siem, tab_if, tab_shap, tab_tiers, tab_integrated = st.tabs(
    ["Overview", "SIEM", "Isolation Forest", "TreeSHAP", "ATT&CK Tiers", "Integrated"]
)

with tab_overview:
    cols = st.columns(4)
    cols[0].metric("SIEM F1", "0.6069", "baseline")
    cols[1].metric("IF F1 (CV mean)", "0.7108", "±0.0224 std")
    cols[2].metric("FPR reduction", "69.7%", "McNemar p<0.000001")
    cols[3].metric("HIGH-tier precision", "97.0%", "83x baseline")
    cols = st.columns(4)
    cols[0].metric("TreeSHAP latency", "4.72 ms", "per alert")
    cols[1].metric("Analyst hours/day", "8.7", "-79.9% vs SIEM")
    cols[2].metric("Mean time to triage", "3 min", "-62.5% vs SIEM")
    cols[3].metric("Analyst confidence", "4.1 / 5", "vs 2.0/5 SIEM")

def show_schema_error(tab_name, df_name, df, e):
    st.error(
        f"Couldn't build the {tab_name} chart from `{df_name}` — its columns don't "
        f"match what this script expects.\n\n"
        f"**Error:** {e}\n\n"
        f"**Columns found:** {list(df.columns)}\n\n"
        f"Paste these column names back to Claude and the script can be adjusted "
        f"to match your real file."
    )


with tab_siem:
    try:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Detection metrics")
            fig = px.bar(siem_df, x="metric", y="value", color_discrete_sequence=["#60A5FA"])
            fig.update_layout(yaxis_range=[0, 1], template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Alert volume")
            alerts_col = pick_col(siem_df_raw, ["alerts", "alert_volume", "alerts_fired"])
            alerts_val = siem_df_raw.iloc[0][alerts_col] if alerts_col else 324
            st.metric("User-day alerts fired", alerts_val)
            st.metric("Analyst hours/day @ 8 min/alert", "43.2")
    except Exception as e:
        show_schema_error("SIEM", "siem_metrics.csv", siem_df_raw, e)

with tab_if:
    try:
        st.subheader("Contamination sensitivity sweep")
        fig = go.Figure()
        for col, color in [("precision", "#60A5FA"), ("recall", "#F87171"), ("f1", "#34D399"), ("fpr", "#F5A524")]:
            fig.add_trace(go.Scatter(x=contamination_df["contamination"], y=contamination_df[col],
                                      mode="lines+markers", name=col, line=dict(color=color)))
        fig.add_vline(x=0.012, line_dash="dash", line_color="gray")
        fig.update_layout(template="plotly_dark", height=400, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        show_schema_error("contamination sweep", contamination_source or "contamination_sensitivity_FINE.csv", contamination_df, e)

    c1, c2 = st.columns(2)
    with c1:
        try:
            if cv_mode == "summary":
                st.subheader("5-fold CV stability")
                cv_plot = cv_df.copy()
                cv_plot["contamination"] = cv_plot["contamination"].astype(str)
                fig = go.Figure(go.Bar(
                    x=cv_plot["contamination"],
                    y=cv_plot["mean_f1"],
                    error_y=dict(type="data", array=cv_plot["std"], visible=True, color="#7D8590"),
                    marker_color=["#60A5FA", "#34D399", "#F5A524"][: len(cv_plot)],
                    text=[f"{m:.4f} ± {s:.4f}" for m, s in zip(cv_plot["mean_f1"], cv_plot["std"])],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="plotly_dark", height=380, yaxis_range=[0, 0.85],
                    xaxis_title="Contamination rate", yaxis_title="Mean F1 ± std dev",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("F1 per fold (selected contamination setting)")
                fold_col = pick_col(cv_fold_df, ["fold"])
                f1_col = pick_col(cv_fold_df, ["mean_f1", "f1"])
                fig = px.line(cv_fold_df, x=fold_col, y=f1_col, markers=True,
                              color_discrete_sequence=["#34D399"])
                fig.update_layout(template="plotly_dark", height=350, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                if cv_summary_error:
                    st.warning(cv_summary_error)
                else:
                    st.caption(
                        "No cross_validation_ACROSS_SETTINGS.csv (or similar) found — "
                        "showing per-fold results for one contamination setting instead. "
                        "If you have a file with the 0.010 / 0.012 / 0.015 comparison, "
                        "tell me its filename and I'll wire it in."
                    )
        except Exception as e:
            show_schema_error("CV stability", "cross_validation_results.csv", cv_fold_df if cv_fold_df is not None else pd.DataFrame(), e)
    with c2:
        try:
            st.subheader("Precision@K ranking quality")
            fig = px.line(precision_k_df, x="k", y="precision", markers=True,
                          color_discrete_sequence=["#F87171"])
            fig.add_hline(y=0.7006, line_dash="dash", line_color="gray")
            fig.update_layout(template="plotly_dark", height=350, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            show_schema_error("Precision@K", "precision_at_k.csv", precision_k_df_raw, e)

with tab_shap:
    try:
        cols = st.columns(len(shap_df))
        for i, row in shap_df.iterrows():
            cols[i].metric(row["metric"], row["value"])
        st.info(
            "unique_logon_types is the strongest global driver of anomaly scores "
            "(mean |SHAP| = 0.333), ranking above unique_destinations (rank 5, "
            "mean |SHAP| = 0.225)."
        )
    except Exception as e:
        show_schema_error("TreeSHAP", "shap_phase3_metrics.csv", shap_df_raw, e)

with tab_tiers:
    try:
        st.subheader("ATT&CK confidence tier calibration")
        fig = go.Figure()
        tier_colors = {"HIGH": "#34D399", "CRITICAL": "#F87171", "MEDIUM-LOW": "#F5A524"}
        bar_colors = [tier_colors.get(t, "#60A5FA") for t in tiers_df["tier"]]
        fig.add_trace(go.Bar(x=tiers_df["tier"], y=tiers_df["precision"], name="Precision %",
                              marker_color=bar_colors, yaxis="y1"))
        fig.add_trace(go.Scatter(x=tiers_df["tier"], y=tiers_df["alerts"], name="Alert count",
                                  mode="lines+markers", yaxis="y2", line=dict(color="#E6EDF3")))
        fig.update_layout(
            template="plotly_dark", height=420,
            yaxis=dict(title="Precision %", range=[0, 100]),
            yaxis2=dict(title="Alert count", overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "CRITICAL's lower precision (77.9%) vs HIGH (97.0%) traces to a single "
            "automated infrastructure account sitting at the tier boundary."
        )
    except Exception as e:
        show_schema_error("ATT&CK tiers", "attack_tiers.csv", tiers_df_raw, e)

with tab_integrated:
    try:
        st.subheader("Detection metrics across configurations")
        fig = go.Figure()
        for display_name, color in CONFIG_COLORS.items():
            actual_col = CONFIG_COLUMN_MAP.get(display_name)
            if actual_col and actual_col in detection_df.columns:
                fig.add_trace(go.Bar(x=detection_df["metric"], y=detection_df[actual_col], name=display_name, marker_color=color))
        fig.update_layout(template="plotly_dark", height=350, yaxis_range=[0, 1], barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        show_schema_error("integrated comparison", integrated_source or "final_integrated_table.csv", integrated_df_raw, e)

    try:
        st.subheader("Operational metrics")
        op_cols = st.columns(2)
        for i, (_, row) in enumerate(operational_df.iterrows()):
            y_vals = []
            for display_name in CONFIG_COLORS.keys():
                actual_col = CONFIG_COLUMN_MAP.get(display_name, display_name)
                y_vals.append(row[actual_col] if actual_col in row.index else row.get(display_name, 0))
            fig = go.Figure(go.Bar(
                x=list(CONFIG_COLORS.keys()),
                y=y_vals,
                marker_color=list(CONFIG_COLORS.values()),
            ))
            fig.update_layout(template="plotly_dark", height=280, title=row["metric"])
            op_cols[i % 2].plotly_chart(fig, use_container_width=True)
    except Exception as e:
        show_schema_error("operational metrics", integrated_source or "final_integrated_table.csv", operational_df, e)

    st.subheader("McNemar's test — SIEM vs IF false positive rate")
    m1, m2, m3 = st.columns(3)
    m1.metric("SIEM FPR", "0.0123")
    m2.metric("IF FPR", "0.0053", "-57%")
    m3.metric("Relative reduction", "69.7%")
    m4, m5, m6 = st.columns(3)
    m4.metric("McNemar statistic", "88.1988")
    m5.metric("p-value", "< 0.000001")
    m6.metric("FP Jaccard overlap", "5.6%", "systems are complementary")
