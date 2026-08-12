"""
Optional CLI runner — produces a static HTML report of the diagnostics
WITHOUT needing to launch Streamlit. Useful for quick checks, CI, or
if you just want a file to email someone.

Usage:
    python run_cli_report.py
    python run_cli_report.py --splits 8 --test-size 0.25 --brackets 4
"""

import argparse
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline import load_data, run_multi_split_evaluation, get_model_zoo, TARGET  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "AmesHousing.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "diagnostics_report.html")


def main():
    parser = argparse.ArgumentParser(description="Run the appraisal diagnostics suite from the CLI.")
    parser.add_argument("--models", nargs="+", default=[
        "Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"
    ], help="Which models to compare (see pipeline.get_model_zoo for full list).")
    parser.add_argument("--splits", type=int, default=5, help="Number of train/test splits.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for testing.")
    parser.add_argument("--brackets", type=int, default=3, help="Number of price brackets (3-5).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = parser.parse_args()

    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    print(f"Running evaluation: models={args.models}, splits={args.splits}, "
          f"test_size={args.test_size}, brackets={args.brackets}")
    result = run_multi_split_evaluation(
        df, model_names=args.models, n_splits=args.splits,
        test_size=args.test_size, n_brackets=args.brackets, base_seed=args.seed,
    )

    agg = result.aggregated
    agg_bracket = result.aggregated_bracket

    print("\n=== AGGREGATED METRICS (mean across splits) ===")
    print(agg[["Model", "MAE (mean)", "RMSE (mean)", "R2 (mean)", "MAPE (%) (mean)"]]
          .sort_values("RMSE (mean)").to_string(index=False))

    print("\n=== PRICE-BRACKET METRICS (mean across splits) ===")
    print(agg_bracket[["Model", "Bracket", "Avg Price (mean)", "MAE (mean)", "RMSE (mean)", "MAPE (%) (mean)"]]
          .sort_values(["Model", "Bracket"]).to_string(index=False))

    # ---- build a simple static HTML report ----
    fig1 = go.Figure()
    for metric, color in [("MAE (mean)", "#2563eb"), ("RMSE (mean)", "#f97316")]:
        fig1.add_trace(go.Bar(x=agg["Model"], y=agg[metric], name=metric.replace(" (mean)", ""), marker_color=color))
    fig1.update_layout(barmode="group", title="MAE & RMSE by Model", plot_bgcolor="white")

    fig2 = px.bar(agg_bracket, x="Bracket", y="MAE (mean)", color="Model", barmode="group",
                  title="MAE by Price Bracket and Model")
    fig2.update_layout(plot_bgcolor="white")

    fig3 = px.bar(agg_bracket, x="Bracket", y="RMSE (mean)", color="Model", barmode="group",
                  title="RMSE by Price Bracket and Model")
    fig3.update_layout(plot_bgcolor="white")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("<html><head><title>Appraisal Diagnostics Report</title></head><body>")
        f.write("<h1>Real Estate Appraisal Regression & Error Diagnostics Report</h1>")
        f.write(f"<p>Dataset: {df.shape[0]} homes, target = {TARGET}. "
                f"Splits: {args.splits}, test size: {args.test_size}, brackets: {args.brackets}.</p>")
        f.write("<h2>Aggregated Metrics</h2>")
        f.write(agg.round(2).to_html(index=False))
        f.write("<h2>Price-Bracket Metrics</h2>")
        f.write(agg_bracket.round(2).to_html(index=False))
        f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")

    print(f"\nHTML report saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
