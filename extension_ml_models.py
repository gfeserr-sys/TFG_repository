# Databricks notebook source
# MAGIC %md
# MAGIC # Extension: Machine Learning Models for Weekly Corporate Credit Spread Changes
# MAGIC
# MAGIC This notebook is an exploratory ML extension to the base TFG project.
# MAGIC
# MAGIC It keeps the same core target:
# MAGIC
# MAGIC \[
# MAGIC \Delta OAS_t = OAS_t - OAS_{t-1}
# MAGIC \]
# MAGIC
# MAGIC and the same IG vs HY comparison, but tests whether non-linear machine learning models can improve forecasts of weekly spread changes.
# MAGIC
# MAGIC Models included:
# MAGIC
# MAGIC 1. **Random Forest Regressor**
# MAGIC 2. **Gradient Boosting Regressor** using `HistGradientBoostingRegressor`
# MAGIC
# MAGIC These models are intentionally limited to a maximum of two to avoid losing the focus of the project.

# COMMAND ----------

# ==========================================================
# DATABRICKS SETUP
# ==========================================================

# Usually scikit-learn is already available in Databricks.
# Uncomment only if your cluster does not have it.
# %pip install scikit-learn

# COMMAND ----------

# ==========================================================
# IMPORTS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

plt.style.use("default")
pd.set_option("display.max_columns", 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC This section should mirror the structure used in the ARX-GARCH extension.
# MAGIC
# MAGIC The notebook expects the same base credit spread tables and optional external market variables in `tfg_data.original_data`.

# COMMAND ----------

# ==========================================================
# CONFIGURATION
# ==========================================================

CATALOG = "tfg_data"
ORIGINAL_SCHEMA = "original_data"
GOLD_SCHEMA = "gold_data"

WEEK_RULE = "W-FRI"
TRAIN_RATIO = 0.80
RANDOM_STATE = 42

# Rolling settings.
# Refit every 26 weeks to keep the exercise faithful but computationally light.
RUN_ROLLING = True
REFIT_EVERY = 26

# Optional safety cap. Use None for full test set.
# If rolling is too slow, set for example ROLLING_MAX_TEST_OBS = 120.
ROLLING_MAX_TEST_OBS = None

# Feature importance settings
RUN_FEATURE_IMPORTANCE = True
PERMUTATION_REPEATS = 5

# Base credit spread tables from the original notebook
BASE_SERIES_CONFIG = {
    "IG": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.ig_aggregate_oas",
        "date_col": "observation_date",
        "value_col": "BAMLC0A0CM",
        "kind": "oas_pct"
    },
    "HY": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.hy_aggregate_oas",
        "date_col": "observation_date",
        "value_col": "BAMLH0A0HYM2",
        "kind": "oas_pct"
    }
}

# Optional external variables.
# If a table does not exist, the notebook skips it automatically.
EXTERNAL_SERIES_CONFIG = {
    "VIX": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.vix",
        "date_col": "observation_date",
        "value_col": "VIXCLS",
        "kind": "level"
    },
    "SP500": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.sp500",
        "date_col": "observation_date",
        "value_col": "SP500",
        "kind": "price"
    },
    "DGS10": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.treasury_10y",
        "date_col": "observation_date",
        "value_col": "DGS10",
        "kind": "rate_pct"
    },
    "DGS2": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.treasury_2y",
        "date_col": "observation_date",
        "value_col": "DGS2",
        "kind": "rate_pct"
    }
}

# Lag structure for ML models.
# Lag 1 captures immediate weekly dynamics.
# Lag 2 and lag 4 allow the model to capture short-term momentum/reversal.
LAGS = (1, 2, 4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Helper functions

# COMMAND ----------

# ==========================================================
# DATA LOADING AND WEEKLY TRANSFORMATION HELPERS
# ==========================================================

def table_exists(table_name):
    """Return True if a Spark table exists."""
    try:
        spark.table(table_name).limit(1).collect()
        return True
    except Exception:
        return False


def load_spark_table_to_pandas(table_name, date_col, value_col):
    """Load a Spark table and return a pandas dataframe with ['date', 'value']."""
    df_spark = (
        spark.sql(f"SELECT {date_col} AS date, {value_col} AS value FROM {table_name}")
        .dropna()
    )

    df = df_spark.toPandas()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date")

    return df


def build_weekly_external_series(df, prefix, kind, week_rule="W-FRI"):
    """
    Convert raw daily data into weekly features.

    kind options:
    - 'oas_pct': FRED OAS in percentage points. Converted to basis points.
    - 'rate_pct': interest rate in percentage points. Difference converted to basis points.
    - 'price': price/index level. Return computed as weekly percentage return.
    - 'level': generic level variable. Difference kept in original units.
    """

    data = df.copy()
    data = data.set_index("date").sort_index().asfreq("D")

    data["value"] = (
        data["value"]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )

    if kind == "price":
        weekly = data.resample(week_rule).last()
    else:
        weekly = data.resample(week_rule).mean(numeric_only=True)

    weekly = weekly.rename(columns={"value": f"{prefix}_level"})

    if kind == "oas_pct":
        weekly[f"{prefix}_value_bp"] = weekly[f"{prefix}_level"] * 100
        weekly[f"{prefix}_diff_1"] = weekly[f"{prefix}_value_bp"].diff()

    elif kind == "rate_pct":
        weekly[f"{prefix}_rate_pct"] = weekly[f"{prefix}_level"]
        weekly[f"{prefix}_diff_bp"] = weekly[f"{prefix}_rate_pct"].diff() * 100

    elif kind == "price":
        weekly[f"{prefix}_return_pct"] = weekly[f"{prefix}_level"].pct_change() * 100

    elif kind == "level":
        weekly[f"{prefix}_diff_1"] = weekly[f"{prefix}_level"].diff()

    else:
        raise ValueError(f"Unknown kind: {kind}")

    weekly = weekly.reset_index().rename(columns={"date": "week_end_date"})

    return weekly


def add_lags(df, cols, lags=(1,)):
    """Add lagged columns to a dataframe keyed by week_end_date."""
    out = df.copy().sort_values("week_end_date")

    for col in cols:
        if col in out.columns:
            for lag in lags:
                out[f"{col}_lag{lag}"] = out[col].shift(lag)

    return out

# COMMAND ----------

# ==========================================================
# MODELING HELPERS
# ==========================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan

    # A zero forecast has no positive/negative direction.
    # The benchmark will therefore look weak in this metric, which is expected.
    return np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))


def evaluate_forecasts(df, actual_col, pred_cols):
    rows = []

    for col in pred_cols:
        rows.append({
            "model": col,
            "RMSE": rmse(df[actual_col], df[col]),
            "MAE": mean_absolute_error(df[actual_col], df[col]),
            "Directional_Accuracy": directional_accuracy(df[actual_col], df[col])
        })

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def diebold_mariano_test(y_true, pred_1, pred_2, power=2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.

    Positive mean_loss_diff means model 2 improves on model 1:
    loss(model 1) - loss(model 2) > 0.
    """

    y_true = np.asarray(y_true)
    pred_1 = np.asarray(pred_1)
    pred_2 = np.asarray(pred_2)

    e1 = y_true - pred_1
    e2 = y_true - pred_2

    d = np.abs(e1)**power - np.abs(e2)**power
    d = pd.Series(d).dropna().values

    T = len(d)
    if T < 5:
        return {
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_loss_diff": np.nan
        }

    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)

    if h > 1:
        gamma = []
        for lag in range(1, h):
            cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
            gamma.append(cov)
        var_d = gamma0 + 2 * np.sum(gamma)
    else:
        var_d = gamma0

    if var_d <= 0:
        return {
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_loss_diff": d_bar
        }

    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "DM_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_bar
    }


def export_to_delta(df, table_name):
    """
    Export a pandas dataframe to a Delta table in Databricks.
    """

    out = df.copy()

    # If dataframe still has a meaningful index, preserve it as a date column.
    if out.index.name is not None:
        out = out.reset_index()
    elif not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index().rename(columns={"index": "date"})

    spark_df = spark.createDataFrame(out)
    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

    print(f"Exported: {table_name}")

# COMMAND ----------

# ==========================================================
# MODEL DEFINITIONS
# ==========================================================

def make_ml_models():
    """
    Returns the two ML models used in this extension.

    Random Forest:
    - Good benchmark for non-linear tabular ML.
    - Robust and interpretable through feature importance.

    Gradient Boosting:
    - Usually stronger for tabular prediction.
    - Can capture non-linear and threshold effects.
    """

    models = {
        "Random_Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient_Boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
    }

    return models

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the ML dataset
# MAGIC
# MAGIC All predictors are lagged to avoid look-ahead bias.
# MAGIC
# MAGIC Compared with the ARX-GARCH notebook, this ML version adds lags 1, 2 and 4 because tree-based models can benefit from a slightly richer lag structure.

# COMMAND ----------

# ==========================================================
# LOAD BASE CREDIT SERIES
# ==========================================================

weekly_frames = []

for name, cfg in BASE_SERIES_CONFIG.items():
    raw = load_spark_table_to_pandas(
        table_name=cfg["table"],
        date_col=cfg["date_col"],
        value_col=cfg["value_col"]
    )

    weekly = build_weekly_external_series(
        raw,
        prefix=name,
        kind=cfg["kind"],
        week_rule=WEEK_RULE
    )

    weekly_frames.append(weekly)

    print(f"{name}: loaded {len(raw)} daily rows -> {len(weekly)} weekly rows")

# Merge IG and HY
ml_df = weekly_frames[0]

for frame in weekly_frames[1:]:
    ml_df = ml_df.merge(frame, on="week_end_date", how="inner")

ml_df["HY_IG_gap_bp"] = ml_df["HY_value_bp"] - ml_df["IG_value_bp"]

print("Base ML dataset shape:", ml_df.shape)
display(ml_df.head())

# COMMAND ----------

# ==========================================================
# LOAD OPTIONAL EXTERNAL VARIABLES
# ==========================================================

loaded_external = []

for name, cfg in EXTERNAL_SERIES_CONFIG.items():
    if not table_exists(cfg["table"]):
        print(f"Skipped {name}: table not found -> {cfg['table']}")
        continue

    raw = load_spark_table_to_pandas(
        table_name=cfg["table"],
        date_col=cfg["date_col"],
        value_col=cfg["value_col"]
    )

    weekly = build_weekly_external_series(
        raw,
        prefix=name,
        kind=cfg["kind"],
        week_rule=WEEK_RULE
    )

    ml_df = ml_df.merge(weekly, on="week_end_date", how="left")
    loaded_external.append(name)

    print(f"{name}: loaded {len(raw)} daily rows -> {len(weekly)} weekly rows")

# Term spread if both DGS10 and DGS2 are available
if {"DGS10", "DGS2"}.issubset(set(loaded_external)):
    ml_df["TERM_SPREAD_10Y2Y"] = ml_df["DGS10_rate_pct"] - ml_df["DGS2_rate_pct"]
    ml_df["TERM_SPREAD_10Y2Y_diff_bp"] = ml_df["TERM_SPREAD_10Y2Y"].diff() * 100

print("Loaded external variables:", loaded_external)
print("ML dataset shape after external merge:", ml_df.shape)
display(ml_df.head())

# COMMAND ----------

# ==========================================================
# CREATE LAGGED FEATURES
# ==========================================================

candidate_feature_cols = [
    # Credit own and cross-market dynamics
    "IG_diff_1",
    "IG_value_bp",
    "HY_diff_1",
    "HY_value_bp",
    "HY_IG_gap_bp",

    # Market risk / volatility
    "VIX_level",
    "VIX_diff_1",

    # Equity market
    "SP500_return_pct",

    # Rates
    "DGS10_rate_pct",
    "DGS10_diff_bp",
    "DGS2_rate_pct",
    "DGS2_diff_bp",
    "TERM_SPREAD_10Y2Y",
    "TERM_SPREAD_10Y2Y_diff_bp"
]

available_feature_cols = [col for col in candidate_feature_cols if col in ml_df.columns]

ml_df = add_lags(
    ml_df,
    cols=available_feature_cols,
    lags=LAGS
)

print("Available raw features:")
print(available_feature_cols)

print("\nAvailable lagged features:")
lagged_cols = [c for c in ml_df.columns if any(c.endswith(f"_lag{lag}") for lag in LAGS)]
print(lagged_cols)

display(ml_df.head())

# COMMAND ----------

# ==========================================================
# DEFINE TARGET-SPECIFIC FEATURE SETS
# ==========================================================

# Use only lagged variables to avoid look-ahead bias.

base_ig_feature_candidates = [
    "IG_diff_1",
    "IG_value_bp",
    "HY_diff_1",
    "HY_value_bp",
    "HY_IG_gap_bp",
    "VIX_level",
    "VIX_diff_1",
    "SP500_return_pct",
    "DGS10_diff_bp",
    "TERM_SPREAD_10Y2Y"
]

base_hy_feature_candidates = [
    "HY_diff_1",
    "HY_value_bp",
    "IG_diff_1",
    "IG_value_bp",
    "HY_IG_gap_bp",
    "VIX_level",
    "VIX_diff_1",
    "SP500_return_pct",
    "DGS10_diff_bp",
    "TERM_SPREAD_10Y2Y"
]


def expand_with_lags(base_features, available_cols, lags):
    out = []

    for feature in base_features:
        for lag in lags:
            col = f"{feature}_lag{lag}"
            if col in available_cols:
                out.append(col)

    return out


FEATURES = {
    "IG": expand_with_lags(base_ig_feature_candidates, ml_df.columns, LAGS),
    "HY": expand_with_lags(base_hy_feature_candidates, ml_df.columns, LAGS)
}

TARGETS = {
    "IG": "IG_diff_1",
    "HY": "HY_diff_1"
}

for name in ["IG", "HY"]:
    print(f"{name} target:", TARGETS[name])
    print(f"{name} number of features:", len(FEATURES[name]))
    print(FEATURES[name])
    print("-" * 80)

# COMMAND ----------

# ==========================================================
# CLEAN MODEL DATASETS AND TRAIN/TEST SPLIT
# ==========================================================

ml_model_data = {}
ml_split_data = {}

for name in ["IG", "HY"]:
    target_col = TARGETS[name]
    feature_cols = FEATURES[name]

    model_df = (
        ml_df[["week_end_date", target_col] + feature_cols]
        .dropna()
        .sort_values("week_end_date")
        .set_index("week_end_date")
    )

    split_idx = int(len(model_df) * TRAIN_RATIO)

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    ml_model_data[name] = model_df
    ml_split_data[name] = {
        "full": model_df,
        "train": train_df,
        "test": test_df,
        "target_col": target_col,
        "feature_cols": feature_cols
    }

    print(f"{name}")
    print(f"Full sample:  {len(model_df)} observations")
    print(f"Train sample: {len(train_df)} observations")
    print(f"Test sample:  {len(test_df)} observations")
    print(f"Train period: {train_df.index.min().date()} to {train_df.index.max().date()}")
    print(f"Test period:  {test_df.index.min().date()} to {test_df.index.max().date()}")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Static out-of-sample ML forecasts
# MAGIC
# MAGIC This is the main ML extension.
# MAGIC
# MAGIC The models are trained on the training sample and evaluated on the final 20% of the sample.

# COMMAND ----------

# ==========================================================
# STATIC OUT-OF-SAMPLE ML FORECASTS
# ==========================================================

static_ml_forecasts = {}
static_ml_evaluations = {}
static_ml_models = {}

for name, parts in ml_split_data.items():
    print("=" * 100)
    print(f"{name} - Static OOS ML models")
    print("=" * 100)

    train_df = parts["train"]
    test_df = parts["test"]
    target_col = parts["target_col"]
    feature_cols = parts["feature_cols"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    fcst = pd.DataFrame({
        "actual": y_test,
        "No_change_forecast": 0.0
    }, index=y_test.index)

    fitted_models = {}

    for model_name, model in make_ml_models().items():
        print(f"Fitting {model_name}...")

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        fcst[f"{model_name}_forecast"] = pred
        fitted_models[model_name] = model

    static_ml_forecasts[name] = fcst
    static_ml_models[name] = fitted_models

    pred_cols = [c for c in fcst.columns if c.endswith("_forecast")]
    eval_table = evaluate_forecasts(
        fcst,
        actual_col="actual",
        pred_cols=pred_cols
    )

    eval_table["segment"] = name
    eval_table["method"] = "Static OOS"

    static_ml_evaluations[name] = eval_table

    display(eval_table.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Rolling one-step-ahead ML forecasts
# MAGIC
# MAGIC This section is optional but useful.
# MAGIC
# MAGIC To keep it computationally reasonable, the models are refitted every 26 weeks instead of every week.

# COMMAND ----------

# ==========================================================
# ROLLING ONE-STEP-AHEAD ML FORECASTS
# Periodic expanding-window refit
# ==========================================================

rolling_ml_forecasts = {}
rolling_ml_evaluations = {}

if RUN_ROLLING:

    for name, parts in ml_split_data.items():
        print("=" * 100)
        print(f"{name} - Rolling one-step ML models, refit every {REFIT_EVERY} weeks")
        print("=" * 100)

        full_df = parts["full"]
        train_df = parts["train"]
        test_df = parts["test"]
        target_col = parts["target_col"]
        feature_cols = parts["feature_cols"]

        if ROLLING_MAX_TEST_OBS is not None:
            test_dates = test_df.index[:ROLLING_MAX_TEST_OBS]
        else:
            test_dates = test_df.index

        train_size = len(train_df)

        models = make_ml_models()
        current_models = {}

        actuals = []
        no_change_preds = []
        model_preds = {model_name: [] for model_name in models.keys()}

        for i, current_date in enumerate(test_dates):

            # Refit at the first iteration and every REFIT_EVERY weeks
            if i == 0 or i % REFIT_EVERY == 0:
                hist_df = full_df.iloc[:train_size + i].copy()

                X_hist = hist_df[feature_cols]
                y_hist = hist_df[target_col]

                current_models = {}

                for model_name, model in make_ml_models().items():
                    model.fit(X_hist, y_hist)
                    current_models[model_name] = model

                print(f"{name}: refitted models at step {i}/{len(test_dates)} ({current_date.date()})")

            # Forecast next observation
            next_row = full_df.iloc[[train_size + i]].copy()

            y_true = float(next_row[target_col].iloc[0])
            X_next = next_row[feature_cols]

            actuals.append(y_true)
            no_change_preds.append(0.0)

            for model_name, model in current_models.items():
                pred = float(model.predict(X_next)[0])
                model_preds[model_name].append(pred)

        fcst_dict = {
            "actual": actuals,
            "No_change_forecast": no_change_preds
        }

        for model_name, preds in model_preds.items():
            fcst_dict[f"{model_name}_forecast"] = preds

        fcst = pd.DataFrame(fcst_dict, index=test_dates)

        rolling_ml_forecasts[name] = fcst

        pred_cols = [c for c in fcst.columns if c.endswith("_forecast")]
        eval_table = evaluate_forecasts(
            fcst,
            actual_col="actual",
            pred_cols=pred_cols
        )

        eval_table["segment"] = name
        eval_table["method"] = f"Rolling 1-step, refit every {REFIT_EVERY} weeks"

        rolling_ml_evaluations[name] = eval_table

        display(eval_table.round(4))

else:
    print("Rolling ML forecasts skipped because RUN_ROLLING = False")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Combined evaluation and Diebold-Mariano tests

# COMMAND ----------

# ==========================================================
# COMBINED ML EVALUATION TABLE
# ==========================================================

evaluation_tables = list(static_ml_evaluations.values())

if RUN_ROLLING and len(rolling_ml_evaluations) > 0:
    evaluation_tables += list(rolling_ml_evaluations.values())

ml_forecast_eval = pd.concat(
    evaluation_tables,
    axis=0,
    ignore_index=True
)

ml_forecast_eval["model"] = ml_forecast_eval["model"].replace({
    "No_change_forecast": "No-change benchmark",
    "Random_Forest_forecast": "Random Forest",
    "Gradient_Boosting_forecast": "Gradient Boosting"
})

ml_forecast_eval = ml_forecast_eval[
    ["segment", "method", "model", "RMSE", "MAE", "Directional_Accuracy"]
]

print("ML forecast evaluation summary:")
display(ml_forecast_eval.round(4))

# COMMAND ----------

# ==========================================================
# DIEBOLD-MARIANO TESTS AGAINST NO-CHANGE BENCHMARK
# ==========================================================

dm_rows = []

for name in ["IG", "HY"]:

    forecast_sets = [("Static OOS", static_ml_forecasts[name])]

    if RUN_ROLLING and name in rolling_ml_forecasts:
        forecast_sets.append((f"Rolling 1-step, refit every {REFIT_EVERY} weeks", rolling_ml_forecasts[name]))

    for method, fcst in forecast_sets:
        for model_name in ["Random_Forest", "Gradient_Boosting"]:

            forecast_col = f"{model_name}_forecast"

            if forecast_col not in fcst.columns:
                continue

            dm_result = diebold_mariano_test(
                y_true=fcst["actual"],
                pred_1=fcst["No_change_forecast"],
                pred_2=fcst[forecast_col],
                power=2,
                h=1
            )

            dm_rows.append({
                "segment": name,
                "method": method,
                "comparison": f"No-change vs {model_name.replace('_', ' ')}",
                **dm_result
            })

ml_dm_results = pd.DataFrame(dm_rows)

print("ML Diebold-Mariano test results:")
display(ml_dm_results.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature importance

# COMMAND ----------

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

feature_importance_rows = []

if RUN_FEATURE_IMPORTANCE:

    for name, parts in ml_split_data.items():
        print("=" * 100)
        print(f"{name} - Feature importance")
        print("=" * 100)

        test_df = parts["test"]
        target_col = parts["target_col"]
        feature_cols = parts["feature_cols"]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        for model_name, model in static_ml_models[name].items():

            print(f"Computing feature importance for {name} - {model_name}")

            # Random Forest has native impurity-based importances.
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                for feature, importance in zip(feature_cols, importances):
                    feature_importance_rows.append({
                        "segment": name,
                        "model": model_name,
                        "importance_type": "native",
                        "feature": feature,
                        "importance": float(importance),
                        "importance_std": np.nan
                    })

            # Permutation importance is model-agnostic and more comparable.
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=PERMUTATION_REPEATS,
                random_state=RANDOM_STATE,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            for feature, importance_mean, importance_std in zip(
                feature_cols,
                perm.importances_mean,
                perm.importances_std
            ):
                feature_importance_rows.append({
                    "segment": name,
                    "model": model_name,
                    "importance_type": "permutation",
                    "feature": feature,
                    "importance": float(importance_mean),
                    "importance_std": float(importance_std)
                })

    ml_feature_importance = pd.DataFrame(feature_importance_rows)

    print("Top permutation importances:")
    display(
        ml_feature_importance
        .query("importance_type == 'permutation'")
        .sort_values(["segment", "model", "importance"], ascending=[True, True, False])
        .groupby(["segment", "model"])
        .head(10)
        .round(4)
    )

else:
    ml_feature_importance = pd.DataFrame()
    print("Feature importance skipped because RUN_FEATURE_IMPORTANCE = False")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Plots

# COMMAND ----------

# ==========================================================
# FORECAST PLOTS
# ==========================================================

for name in ["IG", "HY"]:

    static_fcst = static_ml_forecasts[name]

    plt.figure(figsize=(14, 5))
    plt.plot(static_fcst.index, static_fcst["actual"], label="Actual")
    plt.plot(static_fcst.index, static_fcst["No_change_forecast"], label="No-change")

    for col in ["Random_Forest_forecast", "Gradient_Boosting_forecast"]:
        if col in static_fcst.columns:
            plt.plot(static_fcst.index, static_fcst[col], label=col.replace("_forecast", "").replace("_", " "))

    plt.title(f"{name} - Static OOS ML forecasts")
    plt.xlabel("Date")
    plt.ylabel("Weekly spread change (bp)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if RUN_ROLLING and name in rolling_ml_forecasts:
        rolling_fcst = rolling_ml_forecasts[name]

        plt.figure(figsize=(14, 5))
        plt.plot(rolling_fcst.index, rolling_fcst["actual"], label="Actual")
        plt.plot(rolling_fcst.index, rolling_fcst["No_change_forecast"], label="No-change")

        for col in ["Random_Forest_forecast", "Gradient_Boosting_forecast"]:
            if col in rolling_fcst.columns:
                plt.plot(rolling_fcst.index, rolling_fcst[col], label=col.replace("_forecast", "").replace("_", " "))

        plt.title(f"{name} - Rolling 1-step ML forecasts")
        plt.xlabel("Date")
        plt.ylabel("Weekly spread change (bp)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# ==========================================================
# FEATURE IMPORTANCE PLOTS
# ==========================================================

if RUN_FEATURE_IMPORTANCE and not ml_feature_importance.empty:

    for name in ["IG", "HY"]:
        for model_name in ["Random_Forest", "Gradient_Boosting"]:

            subset = (
                ml_feature_importance
                .query("segment == @name and model == @model_name and importance_type == 'permutation'")
                .sort_values("importance", ascending=False)
                .head(10)
                .sort_values("importance", ascending=True)
            )

            if subset.empty:
                continue

            plt.figure(figsize=(10, 6))
            plt.barh(subset["feature"], subset["importance"])
            plt.title(f"{name} - {model_name.replace('_', ' ')}: top permutation importances")
            plt.xlabel("Permutation importance, RMSE-based")
            plt.tight_layout()
            plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Export outputs

# COMMAND ----------

# ==========================================================
# EXPORT ML EXTENSION OUTPUTS TO GOLD TABLES
# ==========================================================

# Model dataset
export_to_delta(
    ml_df,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_ml_dataset"
)

# Forecast tables
forecast_exports = []

for segment, fcst in static_ml_forecasts.items():
    temp = fcst.copy()
    temp["segment"] = segment
    temp["method"] = "Static OOS"
    temp["date"] = temp.index
    forecast_exports.append(temp)

if RUN_ROLLING and len(rolling_ml_forecasts) > 0:
    for segment, fcst in rolling_ml_forecasts.items():
        temp = fcst.copy()
        temp["segment"] = segment
        temp["method"] = f"Rolling 1-step, refit every {REFIT_EVERY} weeks"
        temp["date"] = temp.index
        forecast_exports.append(temp)

ml_forecasts_export = pd.concat(forecast_exports, axis=0).reset_index(drop=True)

export_to_delta(
    ml_forecasts_export,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_ml_forecasts"
)

# Evaluation tables
export_to_delta(
    ml_forecast_eval,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_ml_evaluation"
)

export_to_delta(
    ml_dm_results,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_ml_dm_results"
)

if RUN_FEATURE_IMPORTANCE and not ml_feature_importance.empty:
    export_to_delta(
        ml_feature_importance,
        f"{CATALOG}.{GOLD_SCHEMA}.extension_ml_feature_importance"
    )

print("Export completed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Interpretation guide
# MAGIC
# MAGIC Use this extension as a side experiment, not as the core of the TFG.
# MAGIC
# MAGIC A good interpretation would be:
# MAGIC
# MAGIC > The machine learning extension tests whether non-linear models can extract additional predictive information from lagged credit and market variables. If they do not clearly outperform the no-change benchmark, this reinforces the idea that weekly spread changes contain limited predictable signal. If they do outperform it, the result should be interpreted carefully because tree-based models are less transparent and more vulnerable to overfitting than the baseline econometric models.
# MAGIC
# MAGIC Main points to check:
# MAGIC
# MAGIC 1. Do Random Forest or Gradient Boosting reduce RMSE/MAE versus the no-change benchmark?
# MAGIC 2. Is the improvement statistically significant according to the Diebold-Mariano test?
# MAGIC 3. Is the result stronger for IG or HY?
# MAGIC 4. Which features matter most?
# MAGIC 5. Does the model improve direction, magnitude, or neither?