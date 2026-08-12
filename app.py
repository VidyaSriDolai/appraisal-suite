"""
Real Estate Appraisal Regression & Error Diagnostics Suite
------------------------------------------------------------
Streamlit dashboard entry point.

Run with:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline import (  # noqa: E402
    load_data,
    run_multi_split_evaluation,
    get_model_zoo,
    TARGET,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "AmesHousing.csv")

# --------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Appraisal Diagnostics Suite",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --accent: #2563eb;
    --accent-soft: #eff6ff;
    --ink: #0f172a;
    --muted: #64748b;
}
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

.main-header {
    padding: 1.4rem 1.8rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
    color: white;
    margin-bottom: 1.4rem;
}
.main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
.main-header p { margin: .35rem 0 0 0; color: #cbd5e1; font-size: .95rem; }

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.metric-card h3 { margin: 0; font-size: .78rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: var(--ink); margin-top: .15rem;}

.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--ink);
    margin: 1.6rem 0 .4rem 0;
    border-left: 4px solid var(--accent);
    padding-left: .6rem;
}
.callout {
    background: var(--accent-soft);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: .85rem 1rem;
    color: #1e3a8a;
    font-size: .9rem;
    margin-bottom: 1rem;
}
.warn-callout {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 12px;
    padding: .85rem 1rem;
    color: #9a3412;
    font-size: .9rem;
    margin-bottom: 1rem;
}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏠 Real Estate Appraisal Regression & Error Diagnostics Suite</h1>
    <p>Ames Housing Dataset &nbsp;·&nbsp; Multi-split evaluation &nbsp;·&nbsp; MAE / RMSE / R² &nbsp;·&nbsp;
    Error broken down by price bracket, so expensive-home mistakes can't hide behind a good R².</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load():
    return load_data(DATA_PATH)

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. Make sure AmesHousing.csv is in the /data folder.")
    st.stop()

df = _load()

# --------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Evaluation Settings")

all_models = list(get_model_zoo().keys())
selected_models = st.sidebar.multiselect(
    "Models to compare",
    options=all_models,
    default=["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"],
    help="Pick which regression models to train and compare.",
)

n_splits = st.sidebar.slider(
    "Number of test splits", min_value=2, max_value=10, value=5, step=1,
    help="Runs the full train/test cycle this many times with different random seeds, so results aren't a fluke of one lucky/unlucky split."
)

test_size = st.sidebar.slider(
    "Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
    help="Fraction of homes held out for testing on each split."
)

bracket_choice = st.sidebar.radio(
    "Price brackets", options=[3, 4, 5],
    format_func=lambda x: {3: "Low / Mid / High", 4: "Low / Mid / High / Luxury", 5: "5 brackets (finer-grained)"}[x],
    index=0,
    help="Sale prices are split into brackets (by quantile) so we can check whether errors quietly balloon for expensive homes."
)

base_seed = st.sidebar.number_input("Random seed base", min_value=0, max_value=9999, value=42, step=1)

run_btn = st.sidebar.button("🚀 Run Evaluation", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: **{df.shape[0]:,} homes**, **{df.shape[1]-1} features**. "
    f"Target: `{TARGET}` (min ${df[TARGET].min():,.0f} — max ${df[TARGET].max():,.0f})."
)

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.config = None

if run_btn:
    if not selected_models:
        st.sidebar.error("Select at least one model.")
    else:
        with st.spinner("Training models across all splits... this usually takes a few seconds."):
            result = run_multi_split_evaluation(
                df,
                model_names=selected_models,
                n_splits=n_splits,
                test_size=test_size,
                n_brackets=bracket_choice,
                base_seed=base_seed,
            )
        st.session_state.result = result
        st.session_state.config = dict(
            models=selected_models, n_splits=n_splits, test_size=test_size,
            n_brackets=bracket_choice, base_seed=base_seed,
        )

# --------------------------------------------------------------------------
# EMPTY STATE
# --------------------------------------------------------------------------
if st.session_state.result is None:
    st.markdown("""
    <div class="callout">
    👋 <b>Step 1</b> — Choose your models and split settings on the left.<br>
    👋 <b>Step 2</b> — Click <b>Run Evaluation</b>.<br>
    👋 <b>Step 3</b> — Explore overall accuracy, per-model comparisons, and (most importantly)
    whether any model is quietly failing on high-priced homes.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📄 Preview raw dataset", expanded=True):
        st.dataframe(df.head(25), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Median Sale Price", f"${df[TARGET].median():,.0f}")

        fig = px.histogram(
            df, x=TARGET, nbins=50, title="Sale Price Distribution",
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(bargap=0.02, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.stop()

# --------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------
result = st.session_state.result
cfg = st.session_state.config
agg = result.aggregated.copy()
agg_bracket = result.aggregated_bracket.copy()

tab_overview, tab_compare, tab_bracket, tab_residuals, tab_data = st.tabs(
    ["📊 Overview", "🏆 Model Comparison", "💰 Price-Bracket Diagnostics", "📉 Residual Analysis", "🔎 Raw Results"]
)

# ---------------- TAB 1: OVERVIEW ----------------
with tab_overview:
    best_model_row = agg.sort_values("RMSE (mean)").iloc[0]
    worst_model_row = agg.sort_values("RMSE (mean)").iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Best Model (by RMSE)", best_model_row["Model"], f"RMSE ${best_model_row['RMSE (mean)']:,.0f}"),
        (c2, "Best Model MAE", f"${best_model_row['MAE (mean)']:,.0f}", f"± {best_model_row['MAE (std)']:,.0f} across splits"),
        (c3, "Best Model R²", f"{best_model_row['R2 (mean)']:.3f}", "closer to 1.0 is better"),
        (c4, "Splits Run", cfg["n_splits"], f"test size {cfg['test_size']:.0%}"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <div class="value">{value}</div>
            <div style="color:#64748b;font-size:.8rem;margin-top:.2rem;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Why not just trust R²?</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="warn-callout">
    ⚠️ R² measures overall variance explained — it can look great (0.85+) while a model is still off by
    tens of thousands of dollars on expensive homes, because those errors get diluted by the much larger
    pool of average-priced homes. Check the <b>Price-Bracket Diagnostics</b> tab to see MAE/RMSE
    for <i>{worst_model_row['Model']}</i> and every other model, broken out by price tier.
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    for metric, color in [("MAE (mean)", "#2563eb"), ("RMSE (mean)", "#f97316")]:
        fig.add_trace(go.Bar(
            x=agg["Model"], y=agg[metric], name=metric.replace(" (mean)", ""),
            marker_color=color,
            error_y=dict(type="data", array=agg[metric.replace("mean", "std")], visible=True),
        ))
    fig.update_layout(
        barmode="group", title="MAE & RMSE by Model (mean ± std across splits)",
        plot_bgcolor="white", yaxis_title="Error ($)", legend_title="Metric",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: MODEL COMPARISON ----------------
with tab_compare:
    st.markdown('<div class="section-title">Aggregated metrics (mean ± std across all splits)</div>', unsafe_allow_html=True)
    display_cols = ["Model", "MAE (mean)", "MAE (std)", "RMSE (mean)", "RMSE (std)",
                     "R2 (mean)", "R2 (std)", "MAPE (%) (mean)", "Max Error (mean)"]
    st.dataframe(
        agg[display_cols].sort_values("RMSE (mean)").style.format({
            "MAE (mean)": "${:,.0f}", "MAE (std)": "${:,.0f}",
            "RMSE (mean)": "${:,.0f}", "RMSE (std)": "${:,.0f}",
            "R2 (mean)": "{:.3f}", "R2 (std)": "{:.3f}",
            "MAPE (%) (mean)": "{:.2f}%", "Max Error (mean)": "${:,.0f}",
        }).background_gradient(subset=["RMSE (mean)"], cmap="Reds"),
        use_container_width=True,
    )

    st.markdown('<div class="section-title">Stability across splits</div>', unsafe_allow_html=True)
    st.caption("Each point is one train/test split. A model with a tight cluster is more reliably accurate; wide spread means performance depends heavily on which homes ended up in the test set.")
    fig2 = px.box(
        result.overall, x="Model", y="RMSE", points="all", color="Model",
        title="RMSE distribution across all splits",
    )
    fig2.update_layout(plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        result.overall, x="MAE", y="R2", color="Model", symbol="Model",
        hover_data=["Split", "RMSE"], title="MAE vs. R² — every split, every model",
    )
    fig3.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- TAB 3: PRICE BRACKET DIAGNOSTICS ----------------
with tab_bracket:
    st.markdown('<div class="section-title">Error by price bracket — the core diagnostic of this suite</div>', unsafe_allow_html=True)
    st.caption("Brackets are built from quantiles of sale price, so each bracket has roughly the same number of homes. Watch how MAE/RMSE change as price increases.")

    metric_choice = st.radio("Metric to plot", ["MAE (mean)", "RMSE (mean)", "MAPE (%) (mean)"], horizontal=True)

    bracket_order = list(pd.unique(agg_bracket["Bracket"]))
    fig4 = px.bar(
        agg_bracket, x="Bracket", y=metric_choice, color="Model", barmode="group",
        category_orders={"Bracket": bracket_order},
        title=f"{metric_choice.replace(' (mean)', '')} by price bracket and model",
    )
    fig4.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.line(
        agg_bracket.sort_values("Avg Price (mean)"), x="Avg Price (mean)", y=metric_choice,
        color="Model", markers=True,
        title=f"{metric_choice.replace(' (mean)', '')} vs. average bracket price",
    )
    fig5.update_layout(plot_bgcolor="white", xaxis_title="Average Sale Price in Bracket ($)")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">Full bracket table</div>', unsafe_allow_html=True)
    bracket_cols = ["Model", "Bracket", "Avg Price (mean)", "MAE (mean)", "RMSE (mean)", "MAPE (%) (mean)", "R2 (mean)"]
    st.dataframe(
        agg_bracket[bracket_cols].sort_values(["Model", "Bracket"]).style.format({
            "Avg Price (mean)": "${:,.0f}", "MAE (mean)": "${:,.0f}",
            "RMSE (mean)": "${:,.0f}", "MAPE (%) (mean)": "{:.2f}%", "R2 (mean)": "{:.3f}",
        }),
        use_container_width=True,
    )

    worst_bracket = agg_bracket.sort_values("MAPE (%) (mean)", ascending=False).iloc[0]
    st.markdown(f"""
    <div class="warn-callout">
    🔎 <b>Biggest blind spot found:</b> <i>{worst_bracket['Model']}</i> on <b>{worst_bracket['Bracket']}</b> homes
    (avg price ${worst_bracket['Avg Price (mean)']:,.0f}) has a mean absolute percentage error of
    <b>{worst_bracket['MAPE (%) (mean)']:.1f}%</b> — meaning appraisals in that segment can be off by tens of
    thousands of dollars even if the model's overall R² looked strong.
    </div>
    """, unsafe_allow_html=True)

# ---------------- TAB 4: RESIDUALS ----------------
with tab_residuals:
    st.markdown('<div class="section-title">Predicted vs. actual (final split)</div>', unsafe_allow_html=True)
    preds = result.predictions
    model_for_plot = st.selectbox("Model", options=sorted(preds["Model"].unique()))
    sub = preds[preds["Model"] == model_for_plot]

    c1, c2 = st.columns(2)
    with c1:
        fig6 = px.scatter(
            sub, x="y_true", y="y_pred", color="bracket", opacity=0.7,
            title=f"{model_for_plot}: Predicted vs. Actual Sale Price",
            labels={"y_true": "Actual Price ($)", "y_pred": "Predicted Price ($)"},
        )
        max_val = max(sub["y_true"].max(), sub["y_pred"].max())
        fig6.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                   line=dict(dash="dash", color="gray"), name="Perfect prediction"))
        fig6.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig6, use_container_width=True)

    with c2:
        fig7 = px.scatter(
            sub, x="y_true", y="residual", color="bracket", opacity=0.7,
            title=f"{model_for_plot}: Residuals vs. Actual Price",
            labels={"y_true": "Actual Price ($)", "residual": "Residual (Actual − Predicted, $)"},
        )
        fig7.add_hline(y=0, line_dash="dash", line_color="gray")
        fig7.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig7, use_container_width=True)

    fig8 = px.histogram(
        sub, x="residual", nbins=40, title=f"{model_for_plot}: Residual Distribution",
        color_discrete_sequence=["#2563eb"],
    )
    fig8.add_vline(x=0, line_dash="dash", line_color="gray")
    fig8.update_layout(plot_bgcolor="white", bargap=0.02)
    st.plotly_chart(fig8, use_container_width=True)

    st.caption(
        "Homes far above the dashed 'perfect prediction' line are under-valued by the model; "
        "homes far below are over-valued. A fan-shaped residual pattern that widens for expensive "
        "homes is a classic sign of heteroscedastic error — accuracy that degrades with price."
    )

# ---------------- TAB 5: RAW RESULTS ----------------
with tab_data:
    st.markdown('<div class="section-title">Per-split, per-model results</div>', unsafe_allow_html=True)
    st.dataframe(result.overall.sort_values(["Model", "Split"]), use_container_width=True)

    st.markdown('<div class="section-title">Per-split, per-bracket results</div>', unsafe_allow_html=True)
    st.dataframe(result.by_bracket.sort_values(["Model", "Split", "Bracket"]), use_container_width=True)

    st.markdown('<div class="section-title">Download</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "⬇️ Overall metrics (CSV)", result.overall.to_csv(index=False).encode(),
        file_name="overall_metrics.csv", mime="text/csv", use_container_width=True,
    )
    c2.download_button(
        "⬇️ Bracket metrics (CSV)", result.by_bracket.to_csv(index=False).encode(),
        file_name="bracket_metrics.csv", mime="text/csv", use_container_width=True,
    )
    c3.download_button(
        "⬇️ Predictions (CSV)", result.predictions.to_csv(index=False).encode(),
        file_name="predictions.csv", mime="text/csv", use_container_width=True,
    )

st.markdown("---")
st.caption("Real Estate Appraisal Regression & Error Diagnostics Suite · Ames Housing Dataset · Built with scikit-learn, pandas & Streamlit.")
