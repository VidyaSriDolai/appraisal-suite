"""
Real Estate Appraisal Regression & Error Diagnostics Suite
------------------------------------------------------------
Core pipeline module.

Responsible for:
    1. Loading & cleaning the Ames Housing dataset
    2. Building preprocessing + model pipelines
    3. Running repeated train/test splits ("multi-split evaluation")
    4. Computing rich error diagnostics (MAE, RMSE, MAPE, R2,
       and — the whole point of this project — error broken down
       by PRICE BRACKET, so a model that looks great on R2 but
       is terrible on expensive homes gets caught.)

This module has NO Streamlit / UI code in it on purpose, so it can be
reused from the dashboard, a notebook, or a plain CLI script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

TARGET = "SalePrice"

# Columns that are IDs / leakage-prone / almost entirely empty -> drop
DROP_COLS = ["Id", "PoolQC", "MiscFeature", "Alley", "Fence", "Utilities"]


# --------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the raw Ames Housing CSV."""
    df = pd.read_csv(path)
    return df


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split remaining feature columns into numeric / categorical lists."""
    feature_df = df.drop(columns=[TARGET])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop unusable columns, separate features (X) from target (y)."""
    df = df.copy()
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing)
    df = df.dropna(subset=[TARGET])
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - median-imputes + scales numeric columns
      - most-frequent-imputes + one-hot-encodes categorical columns
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return preprocessor


# --------------------------------------------------------------------------
# 2. MODELS
# --------------------------------------------------------------------------
def get_model_zoo() -> Dict[str, object]:
    """Return the candidate regression models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=42),
        "Lasso Regression": Lasso(alpha=200.0, random_state=42, max_iter=20000),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


# --------------------------------------------------------------------------
# 3. PRICE BRACKETS  (this is the "hidden error" diagnostic)
# --------------------------------------------------------------------------
def assign_price_brackets(y: pd.Series, n_brackets: int = 3) -> pd.Series:
    """
    Bucket sale prices into labeled brackets (e.g. Low / Mid / High) using
    quantiles of the FULL target distribution, so brackets are comparable
    across every split/model.
    """
    labels_3 = ["Low-price", "Mid-price", "High-price"]
    labels_4 = ["Low-price", "Mid-price", "High-price", "Luxury"]
    labels_5 = ["Low-price", "Lower-mid", "Mid-price", "Upper-mid", "Luxury"]
    label_map = {3: labels_3, 4: labels_4, 5: labels_5}
    labels = label_map.get(n_brackets, [f"Bracket {i+1}" for i in range(n_brackets)])
    brackets = pd.qcut(y, q=n_brackets, labels=labels, duplicates="drop")
    return brackets


# --------------------------------------------------------------------------
# 4. METRICS
# --------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    max_err = float(np.max(np.abs(y_true - y_pred)))
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape, "Max Error": max_err}


# --------------------------------------------------------------------------
# 5. MULTI-SPLIT EVALUATION ENGINE
# --------------------------------------------------------------------------
@dataclass
class EvalResult:
    overall: pd.DataFrame                 # one row per (model, split)
    by_bracket: pd.DataFrame              # one row per (model, split, bracket)
    predictions: pd.DataFrame             # raw predictions from the LAST split (for plots)
    aggregated: pd.DataFrame = field(default=None)          # mean/std across splits, per model
    aggregated_bracket: pd.DataFrame = field(default=None)  # mean/std across splits, per model+bracket


def run_multi_split_evaluation(
    df: pd.DataFrame,
    model_names: List[str],
    n_splits: int = 5,
    test_size: float = 0.2,
    n_brackets: int = 3,
    base_seed: int = 0,
) -> EvalResult:
    """
    The heart of the suite: trains each selected model across `n_splits`
    different random train/test partitions, and records:
      - overall metrics per (model, split)
      - price-bracket-level metrics per (model, split, bracket)
      - predictions from the final split (used for residual/scatter plots)
    """
    X, y = prepare_xy(df)
    numeric_cols, categorical_cols = get_feature_lists(
        pd.concat([X, y.rename(TARGET)], axis=1)
    )
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    all_models = get_model_zoo()
    selected_models = {name: all_models[name] for name in model_names if name in all_models}

    overall_rows = []
    bracket_rows = []
    last_predictions = []

    price_brackets_full = assign_price_brackets(y, n_brackets=n_brackets)

    for split_i in range(n_splits):
        seed = base_seed + split_i
        X_train, X_test, y_train, y_test, br_train, br_test = train_test_split(
            X, y, price_brackets_full, test_size=test_size, random_state=seed
        )

        for model_name, model in selected_models.items():
            pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # ---- overall metrics for this split ----
            m = compute_metrics(y_test.values, y_pred)
            m.update({"Model": model_name, "Split": split_i, "Seed": seed, "n_test": len(y_test)})
            overall_rows.append(m)

            # ---- per price-bracket metrics for this split ----
            pred_df = pd.DataFrame({
                "y_true": y_test.values,
                "y_pred": y_pred,
                "bracket": br_test.values,
            })
            for bracket_name, sub in pred_df.groupby("bracket", observed=True):
                if len(sub) == 0:
                    continue
                bm = compute_metrics(sub["y_true"].values, sub["y_pred"].values)
                bm.update({
                    "Model": model_name,
                    "Split": split_i,
                    "Bracket": bracket_name,
                    "n_test": len(sub),
                    "Avg Price": sub["y_true"].mean(),
                })
                bracket_rows.append(bm)

            # keep predictions from the LAST split only (for scatter/residual plots)
            if split_i == n_splits - 1:
                tmp = pred_df.copy()
                tmp["Model"] = model_name
                tmp["residual"] = tmp["y_true"] - tmp["y_pred"]
                tmp["abs_pct_error"] = (tmp["residual"].abs() / tmp["y_true"]) * 100
                last_predictions.append(tmp)

    overall = pd.DataFrame(overall_rows)
    by_bracket = pd.DataFrame(bracket_rows)
    predictions = pd.concat(last_predictions, ignore_index=True) if last_predictions else pd.DataFrame()

    metric_cols = ["MAE", "RMSE", "R2", "MAPE (%)", "Max Error"]
    aggregated = (
        overall.groupby("Model")[metric_cols]
        .agg(["mean", "std"])
        .round(2)
    )
    aggregated.columns = [f"{a} ({b})" for a, b in aggregated.columns]
    aggregated = aggregated.reset_index()

    aggregated_bracket = (
        by_bracket.groupby(["Model", "Bracket"], observed=True)[metric_cols + ["Avg Price"]]
        .agg(["mean", "std"])
        .round(2)
    )
    aggregated_bracket.columns = [f"{a} ({b})" for a, b in aggregated_bracket.columns]
    aggregated_bracket = aggregated_bracket.reset_index()

    return EvalResult(
        overall=overall,
        by_bracket=by_bracket,
        predictions=predictions,
        aggregated=aggregated,
        aggregated_bracket=aggregated_bracket,
    )
