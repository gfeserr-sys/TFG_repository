# Databricks notebook source
# MAGIC %md
# MAGIC # Extension: ARX-GJR-GARCH with macro/market variables
# MAGIC
# MAGIC This notebook updates the previous `extension_arx_garch` notebook in two ways:
# MAGIC
# MAGIC 1. The credit universe is now **IG, BBB, BB and B** instead of IG vs HY.
# MAGIC 2. The macro/market predictor block is expanded to include:
# MAGIC    - VIX
# MAGIC    - S&P 500 weekly return
# MAGIC    - 2-year Treasury yield
# MAGIC    - 10-year Treasury yield
# MAGIC    - 10Y-2Y term spread
# MAGIC    - Fed funds rate
# MAGIC    - Chicago Fed National Financial Conditions Index (NFCI)
# MAGIC
# MAGIC The target remains the weekly change in OAS in basis points:
# MAGIC
# MAGIC \[
# MAGIC \Delta OAS_t = OAS_t - OAS_{t-1}
# MAGIC \]
# MAGIC
# MAGIC The model estimates an ARX / ARMAX mean equation and then fits GJR-GARCH on the residuals to model conditional volatility.
# MAGIC

# COMMAND ----------

# Databricks setup
%pip install statsmodels arch scikit-learn

# COMMAND ----------

# ==========================================================
# IMPORTS
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

plt.style.use("default")
pd.set_option("display.max_columns", 140)
pd.set_option("display.width", 180)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC The credit block is now **IG, BBB, BB and B**.
# MAGIC
# MAGIC The macro/market block is intentionally compact. The idea is not to add every possible macro variable, but to test whether a small set of economically motivated predictors improves forecasting performance.

# COMMAND ----------

# ==========================================================
# CONFIGURATION
# ==========================================================

CATALOG = "tfg_data"
ORIGINAL_SCHEMA = "original_data"
GOLD_SCHEMA = "gold_data"

WEEK_RULE = "W-FRI"
TRAIN_RATIO = 0.80

# ----------------------------------------------------------
# Credit series
# ----------------------------------------------------------
# IG comes from its own table.
# BBB, BB and B come from the rating bucket table used in the rating-buckets notebook.

CREDIT_SERIES_CONFIG = {
    "IG": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.ig_aggregate_oas",
        "date_col": "observation_date",
        "value_col": "BAMLC0A0CM",
        "kind": "oas_pct"
    },
    "BBB": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.bbb_bb_b_extensiondata",
        "date_col": "observation_date",
        "value_col": "BAMLC0A4CBBB",
        "kind": "oas_pct"
    },
    "BB": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.bbb_bb_b_extensiondata",
        "date_col": "observation_date",
        "value_col": "BAMLH0A1HYBB",
        "kind": "oas_pct"
    },
    "B": {
        "table": f"{CATALOG}.{ORIGINAL_SCHEMA}.bbb_bb_b_extensiondata",
        "date_col": "observation_date",
        "value_col": "BAMLH0A2HYB",
        "kind": "oas_pct"
    }
}

TARGET_SERIES = ["IG", "BBB", "BB", "B"]

# ----------------------------------------------------------
# Macro / market predictors
# ----------------------------------------------------------
# If a table exists in Databricks, the notebook loads it.
# If it does not exist and fred_id is provided, the notebook tries to download it from FRED.
#
# Suggested table names in Databricks:
# - vix
# - sp500
# - treasury_10y
# - treasury_2y
# - fed_funds
# - financial_conditions_index

EXTERNAL_SERIES_CONFIG = {
    "VIX": {
        "table_candidates": [
            f"{CATALOG}.{ORIGINAL_SCHEMA}.vix"
        ],
        "date_col": "observation_date",
        "value_col_candidates": ["VIXCLS", "VIX", "value"],
        "kind": "level",
        "fred_id": "VIXCLS"
    },
    "SP500": {
        "table_candidates": [
            f"{CATALOG}.{ORIGINAL_SCHEMA}.sp500"
        ],
        "date_col": "observation_date",
        "value_col_candidates": ["SP500", "S&P500", "value"],
        "kind": "price",
        "fred_id": "SP500"
    },
    "DGS10": {
        "table_candidates": [
            f"{CATALOG}.{ORIGINAL_SCHEMA}.treasury_10y"
        ],
        "date_col": "observation_date",
        "value_col_candidates": ["DGS10", "treasury_10y", "value"],
        "kind": "rate_pct",
        "fred_id": "DGS10"
    },
    "DGS2": {
        "table_candidates": [
            f"{CATALOG}.{ORIGINAL_SCHEMA}.treasury_2y"
        ],
        "date_col": "observation_date",
        "value_col_candidates": ["DGS2", "treasury_2y", "value"],
        "kind": "rate_pct",
        "fred_id": "DGS2"
    },
    "FEDFUNDS": {
        "table_candidates": [
            f"{CATALOG}.{ORIGINAL_SCHEMA}.fed_funds",
            f"{CATALOG}.{ORIGINAL_SCHEMA}.dff",
            f"{CATALOG}.{ORIGINAL_SCHEMA}.effr"
        ],
        "date_col": "observation_date",
        "value_col_candidates": ["DFF", "EFFR", "FEDFUNDS", "fed_funds", "value"],
        "kind": "rate_pct",
        "fred_id": "DFF"
    },
    "NFCI": {
        "table_candidates": [
            f"{CATALOG}.{ORIGINAL_SCHEMA}.financial_conditions_index",
            f"{CATALOG}.{ORIGINAL_SCHEMA}.nfci",
            f"{CATALOG}.{ORIGINAL_SCHEMA}.chicago_fed_nfci"
        ],
        "date_col": "observation_date",
        "value_col_candidates": ["NFCI", "financial_conditions_index", "value"],
        "kind": "weekly_level",
        "fred_id": "NFCI"
    }
}

# ----------------------------------------------------------
# ARMA orders
# ----------------------------------------------------------
# These are the orders already selected in your base rating-buckets notebook.
# You can change them if a later AIC selection gives different choices.

MODEL_ORDERS = {
    "IG": (1, 0, 2),
    "BBB": (1, 0, 3),
    "BB": (3, 0, 3),
    "B": (3, 0, 3)
}

# Rolling forecast settings
REFIT_EVERY = 12
MAXITER_ARX = 100


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


def get_table_columns(table_name):
    """Return Spark table columns safely."""
    return spark.table(table_name).columns


def pick_existing_column(columns, candidates):
    """Pick the first candidate column that exists in columns."""
    col_map = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if str(cand).lower() in col_map:
            return col_map[str(cand).lower()]
    return None


def load_spark_table_to_pandas(table_name, date_col, value_col):
    """Load a Spark table and return a pandas dataframe with ['date', 'value']."""
    df_spark = (
        spark.sql(f"SELECT `{date_col}` AS date, `{value_col}` AS value FROM {table_name}")
        .dropna()
    )

    df = df_spark.toPandas()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date")

    return df


def load_external_series(name, cfg):
    """
    Load external variable from Databricks if available.
    If not available, try direct FRED CSV download when fred_id is provided.
    """

    # 1) Try Databricks tables first
    for table_name in cfg.get("table_candidates", []):
        if table_exists(table_name):
            cols = get_table_columns(table_name)

            date_col = cfg.get("date_col", "observation_date")
            if date_col not in cols:
                date_col = pick_existing_column(cols, ["observation_date", "date", "DATE"])

            value_col = pick_existing_column(cols, cfg.get("value_col_candidates", []))

            if date_col is None or value_col is None:
                print(f"Skipped {name}: could not identify date/value columns in {table_name}")
                continue

            print(f"{name}: loading from Databricks table {table_name}, column {value_col}")
            return load_spark_table_to_pandas(table_name, date_col, value_col)

    # 2) FRED fallback
    fred_id = cfg.get("fred_id", None)

    if fred_id is not None:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
            raw = pd.read_csv(url)
            raw.columns = ["date", "value"]
            raw["date"] = pd.to_datetime(raw["date"])
            raw["value"] = pd.to_numeric(raw["value"].replace(".", np.nan), errors="coerce")
            raw = raw.dropna().sort_values("date")
            print(f"{name}: loaded directly from FRED series {fred_id}")
            return raw
        except Exception as e:
            print(f"Skipped {name}: FRED download failed for {fred_id}: {e}")
            return None

    print(f"Skipped {name}: no Databricks table and no FRED fallback.")
    return None


def build_weekly_series(df, prefix, kind, week_rule="W-FRI"):
    """
    Convert raw data into weekly features.

    kind options:
    - 'oas_pct': FRED OAS in percentage points. Converted to basis points.
    - 'rate_pct': interest rate in percentage points. Difference converted to basis points.
    - 'price': price/index level. Return computed as weekly percentage return.
    - 'level': generic daily level variable. Difference kept in original units.
    - 'weekly_level': weekly level variable, resampled by last available value.
    """

    data = df.copy()
    data = data.set_index("date").sort_index()

    if kind == "weekly_level":
        # For series such as NFCI that are already weekly.
        weekly = data.resample(week_rule).last().ffill()
    else:
        data = data.asfreq("D")
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

    elif kind in ["level", "weekly_level"]:
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


def export_to_delta(df, table_name):
    """Export pandas dataframe to Spark Delta table."""
    spark_df = spark.createDataFrame(df.reset_index(drop=True))
    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
    print(f"Saved table: {table_name}")


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

    if T < 10:
        return {
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_loss_diff": np.nan,
            "n_obs": T
        }

    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)

    if h > 1:
        gamma = []
        for lag in range(1, h):
            if lag < T:
                gamma.append(np.cov(d[lag:], d[:-lag], ddof=1)[0, 1])
        var_d = gamma0 + 2 * np.sum(gamma)
    else:
        var_d = gamma0

    if var_d <= 0 or np.isnan(var_d):
        return {
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_loss_diff": d_bar,
            "n_obs": T
        }

    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "DM_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "n_obs": T
    }


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the extension dataset
# MAGIC
# MAGIC The extension is now based on **IG, BBB, BB and B**.
# MAGIC
# MAGIC All predictors are lagged by one week so that the model uses information available before the forecasted week.

# COMMAND ----------

# ==========================================================
# LOAD CREDIT SERIES: IG, BBB, BB, B
# ==========================================================

weekly_frames = []

for name, cfg in CREDIT_SERIES_CONFIG.items():
    raw = load_spark_table_to_pandas(
        table_name=cfg["table"],
        date_col=cfg["date_col"],
        value_col=cfg["value_col"]
    )

    weekly = build_weekly_series(
        raw,
        prefix=name,
        kind=cfg["kind"],
        week_rule=WEEK_RULE
    )

    weekly_frames.append(weekly)

    print(f"{name}: loaded {len(raw)} raw rows -> {len(weekly)} weekly rows")


# Merge credit series on common weekly dates
extension_df = weekly_frames[0]

for frame in weekly_frames[1:]:
    extension_df = extension_df.merge(frame, on="week_end_date", how="inner")

# Rating gaps
extension_df["BBB_IG_gap_bp"] = extension_df["BBB_value_bp"] - extension_df["IG_value_bp"]
extension_df["BB_BBB_gap_bp"] = extension_df["BB_value_bp"] - extension_df["BBB_value_bp"]
extension_df["B_BB_gap_bp"] = extension_df["B_value_bp"] - extension_df["BB_value_bp"]
extension_df["B_IG_gap_bp"] = extension_df["B_value_bp"] - extension_df["IG_value_bp"]

for gap_col in ["BBB_IG_gap_bp", "BB_BBB_gap_bp", "B_BB_gap_bp", "B_IG_gap_bp"]:
    extension_df[f"{gap_col}_diff_1"] = extension_df[gap_col].diff()

print("Credit extension dataset shape:", extension_df.shape)
display(extension_df.head())


# COMMAND ----------

# ==========================================================
# LOAD MACRO / MARKET VARIABLES
# ==========================================================

loaded_external = []

for name, cfg in EXTERNAL_SERIES_CONFIG.items():
    raw = load_external_series(name, cfg)

    if raw is None or raw.empty:
        print(f"Skipped {name}: no data loaded.")
        continue

    weekly = build_weekly_series(
        raw,
        prefix=name,
        kind=cfg["kind"],
        week_rule=WEEK_RULE
    )

    extension_df = extension_df.merge(weekly, on="week_end_date", how="left")
    loaded_external.append(name)

    print(f"{name}: loaded {len(raw)} rows -> {len(weekly)} weekly rows")

# Term spread if both DGS10 and DGS2 are available
if {"DGS10", "DGS2"}.issubset(set(loaded_external)):
    extension_df["TERM_SPREAD_10Y2Y"] = extension_df["DGS10_rate_pct"] - extension_df["DGS2_rate_pct"]
    extension_df["TERM_SPREAD_10Y2Y_diff_bp"] = extension_df["TERM_SPREAD_10Y2Y"].diff() * 100

# A compact overview of what actually loaded
print("Loaded external variables:", loaded_external)
print("Extension dataset shape after external merge:", extension_df.shape)
display(extension_df.head())


# COMMAND ----------

# ==========================================================
# CREATE LAGGED FEATURES
# ==========================================================

credit_feature_cols = []

for s in TARGET_SERIES:
    credit_feature_cols += [
        f"{s}_diff_1",
        f"{s}_value_bp"
    ]

gap_feature_cols = [
    "BBB_IG_gap_bp",
    "BB_BBB_gap_bp",
    "B_BB_gap_bp",
    "B_IG_gap_bp",
    "BBB_IG_gap_bp_diff_1",
    "BB_BBB_gap_bp_diff_1",
    "B_BB_gap_bp_diff_1",
    "B_IG_gap_bp_diff_1"
]

macro_feature_cols = [
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
    "TERM_SPREAD_10Y2Y_diff_bp",

    # Policy rate
    "FEDFUNDS_rate_pct",
    "FEDFUNDS_diff_bp",

    # Financial conditions
    "NFCI_level",
    "NFCI_diff_1"
]

candidate_feature_cols = credit_feature_cols + gap_feature_cols + macro_feature_cols

available_feature_cols = [col for col in candidate_feature_cols if col in extension_df.columns]

extension_df = add_lags(
    extension_df,
    cols=available_feature_cols,
    lags=(1,)
)

print("Available raw features:")
print(available_feature_cols)

print("\nAvailable lagged features:")
lagged_cols = [c for c in extension_df.columns if c.endswith("_lag1")]
print(lagged_cols)

display(extension_df.head())


# COMMAND ----------

# ==========================================================
# DEFINE TARGET-SPECIFIC FEATURE SETS
# ==========================================================

# Use only lagged variables to avoid look-ahead bias.
# Same information set for all target series, but the target variable changes.

common_feature_candidates = [
    # Credit level and momentum
    "IG_diff_1_lag1",
    "IG_value_bp_lag1",
    "BBB_diff_1_lag1",
    "BBB_value_bp_lag1",
    "BB_diff_1_lag1",
    "BB_value_bp_lag1",
    "B_diff_1_lag1",
    "B_value_bp_lag1",

    # Cross-rating gaps
    "BBB_IG_gap_bp_lag1",
    "BB_BBB_gap_bp_lag1",
    "B_BB_gap_bp_lag1",
    "B_IG_gap_bp_lag1",
    "BBB_IG_gap_bp_diff_1_lag1",
    "BB_BBB_gap_bp_diff_1_lag1",
    "B_BB_gap_bp_diff_1_lag1",
    "B_IG_gap_bp_diff_1_lag1",

    # Macro / market variables
    "VIX_level_lag1",
    "VIX_diff_1_lag1",
    "SP500_return_pct_lag1",
    "DGS10_rate_pct_lag1",
    "DGS10_diff_bp_lag1",
    "DGS2_rate_pct_lag1",
    "DGS2_diff_bp_lag1",
    "TERM_SPREAD_10Y2Y_lag1",
    "TERM_SPREAD_10Y2Y_diff_bp_lag1",
    "FEDFUNDS_rate_pct_lag1",
    "FEDFUNDS_diff_bp_lag1",
    "NFCI_level_lag1",
    "NFCI_diff_1_lag1"
]

common_features = [c for c in common_feature_candidates if c in extension_df.columns]

FEATURES = {}
TARGETS = {}

for name in TARGET_SERIES:
    TARGETS[name] = f"{name}_diff_1"
    FEATURES[name] = common_features.copy()

for name in TARGET_SERIES:
    print(f"{name} target:", TARGETS[name])
    print(f"{name} number of features:", len(FEATURES[name]))
    print(FEATURES[name])
    print("-" * 80)


# COMMAND ----------

# ==========================================================
# CLEAN MODEL DATASETS AND TRAIN/TEST SPLIT
# ==========================================================

extension_model_data = {}
extension_split_data = {}

for name in TARGET_SERIES:
    target_col = TARGETS[name]
    feature_cols = FEATURES[name]

    model_df = (
        extension_df[["week_end_date", target_col] + feature_cols]
        .dropna()
        .sort_values("week_end_date")
        .set_index("week_end_date")
    )

    split_idx = int(len(model_df) * TRAIN_RATIO)

    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    extension_model_data[name] = model_df
    extension_split_data[name] = {
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
# MAGIC ## 4. ARX-GJR-GARCH estimation
# MAGIC
# MAGIC In this implementation:
# MAGIC
# MAGIC 1. SARIMAX estimates the ARX / ARMAX mean equation with exogenous variables.
# MAGIC 2. GJR-GARCH is fitted on the SARIMAX residuals.
# MAGIC 3. The point forecast comes from the ARX / ARMAX mean equation.
# MAGIC 4. GJR-GARCH adds the conditional volatility forecast.
# MAGIC

# COMMAND ----------

# ==========================================================
# STATIC OUT-OF-SAMPLE ARX-GJR-GARCH
# ==========================================================

static_forecasts = {}
static_evaluations = {}

for name, parts in extension_split_data.items():
    print("=" * 100)
    print(f"{name} - Static ARX-GJR-GARCH")
    print("=" * 100)

    train_df = parts["train"]
    test_df = parts["test"]
    target_col = parts["target_col"]
    feature_cols = parts["feature_cols"]

    y_train = train_df[target_col]
    X_train = train_df[feature_cols]

    y_test = test_df[target_col]
    X_test = test_df[feature_cols]

    order = MODEL_ORDERS[name]

    # 1) No-change benchmark
    rw_forecast = pd.Series(0.0, index=y_test.index)

    # 2) ARX / ARMAX mean equation
    arx_model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False, maxiter=300)

    arx_forecast = pd.Series(
        arx_model.forecast(steps=len(y_test), exog=X_test),
        index=y_test.index
    )

    # 3) GJR-GARCH on ARX residuals
    arx_resid = pd.Series(arx_model.resid).dropna()

    gjr_model = arch_model(
        arx_resid,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal",
        rescale=False
    ).fit(disp="off")

    gjr_var_fcst = gjr_model.forecast(horizon=len(y_test), reindex=False).variance.iloc[-1].values
    gjr_sigma_fcst = np.sqrt(gjr_var_fcst)

    # Store
    fcst = pd.DataFrame({
        "actual": y_test,
        "RW_forecast": rw_forecast,
        "ARX_forecast": arx_forecast,
        "ARX_GJR_forecast": arx_forecast,
        "ARX_GJR_sigma": gjr_sigma_fcst
    }, index=y_test.index)

    static_forecasts[name] = {
        "forecast": fcst,
        "arx_model": arx_model,
        "gjr_model": gjr_model
    }

    eval_table = evaluate_forecasts(
        fcst,
        actual_col="actual",
        pred_cols=["RW_forecast", "ARX_forecast", "ARX_GJR_forecast"]
    )
    eval_table["segment"] = name
    eval_table["method"] = "Static OOS"

    static_evaluations[name] = eval_table

    print(arx_model.summary())
    print(gjr_model.summary())
    print("\nStatic evaluation:")
    display(eval_table.round(4))


# COMMAND ----------

# ==========================================================
# EFFICIENT ROLLING ONE-STEP-AHEAD ARX + STATIC GJR-GARCH VOLATILITY
# Expanding window with periodic ARX refit
# ==========================================================

import warnings
warnings.filterwarnings("ignore", message="No frequency information was provided*")

rolling_forecasts = {}
rolling_evaluations = {}
rolling_gjr_models = {}

def get_arch_param(params, name, default=0.0):
    """
    Safely extract ARCH/GARCH parameter from fitted model.
    """
    return float(params[name]) if name in params.index else default


for name, parts in extension_split_data.items():
    print("=" * 100)
    print(f"{name} - Efficient rolling one-step ARX + static GJR-GARCH volatility")
    print("=" * 100)

    full_df = parts["full"].copy().sort_index()
    train_df = parts["train"].copy().sort_index()
    test_df = parts["test"].copy().sort_index()

    target_col = parts["target_col"]
    feature_cols = parts["feature_cols"]

    order = MODEL_ORDERS[name]
    train_size = len(train_df)
    rolling_dates = test_df.index

    actuals = []
    rw_preds = []
    arx_preds = []
    arx_gjr_preds = []
    arx_gjr_sigmas = []
    arx_gjr_lower_95 = []
    arx_gjr_upper_95 = []

    # 1) Initial ARX model estimated on training sample
    y_train = train_df[target_col]
    X_train = train_df[feature_cols]

    arx_results = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False, maxiter=MAXITER_ARX)

    # 2) Static GJR-GARCH model on initial ARX residuals
    train_resid = pd.Series(arx_results.resid, index=y_train.index).dropna()

    gjr_model = arch_model(
        train_resid,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal",
        rescale=False
    ).fit(disp="off")

    rolling_gjr_models[name] = gjr_model

    params = gjr_model.params

    omega = get_arch_param(params, "omega")
    alpha = get_arch_param(params, "alpha[1]")
    gamma = get_arch_param(params, "gamma[1]")
    beta = get_arch_param(params, "beta[1]")

    last_eps = float(train_resid.iloc[-1])
    last_sigma2 = float(gjr_model.conditional_volatility.iloc[-1] ** 2)

    print("Initial GJR-GARCH parameters:")
    print(f"omega={omega:.4f}, alpha={alpha:.4f}, gamma={gamma:.4f}, beta={beta:.4f}")

    # 3) Rolling one-step-ahead forecasting
    for i, current_date in enumerate(rolling_dates):

        # Refit ARX every REFIT_EVERY weeks using all information available up to t-1
        if i > 0 and i % REFIT_EVERY == 0:
            hist_df = full_df.iloc[:train_size + i].copy()

            y_hist = hist_df[target_col]
            X_hist = hist_df[feature_cols]

            try:
                arx_results = SARIMAX(
                    y_hist,
                    exog=X_hist,
                    order=order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False, maxiter=MAXITER_ARX)

            except Exception as e:
                print(f"Warning: ARX refit failed at {current_date}: {e}")

        # Current test observation
        next_row = full_df.iloc[[train_size + i]].copy()

        y_true = float(next_row[target_col].iloc[0])
        X_next = next_row[feature_cols]

        # 1) No-change benchmark
        rw_pred = 0.0

        # 2) ARX one-step forecast
        try:
            arx_pred = float(np.asarray(arx_results.forecast(steps=1, exog=X_next))[0])
        except Exception as e:
            print(f"Warning: ARX forecast failed at {current_date}: {e}")
            arx_pred = np.nan

        # 3) GJR-GARCH volatility forecast
        if np.isfinite(arx_pred):

            asymmetry_indicator = 1.0 if last_eps < 0 else 0.0

            sigma2_next = (
                omega
                + alpha * last_eps**2
                + gamma * asymmetry_indicator * last_eps**2
                + beta * last_sigma2
            )

            sigma2_next = max(sigma2_next, 1e-12)
            gjr_sigma_1 = np.sqrt(sigma2_next)

            lower_95 = arx_pred - 1.96 * gjr_sigma_1
            upper_95 = arx_pred + 1.96 * gjr_sigma_1

            current_eps = y_true - arx_pred
            last_eps = current_eps
            last_sigma2 = sigma2_next

            try:
                arx_results = arx_results.append(
                    endog=next_row[target_col],
                    exog=X_next,
                    refit=False
                )
            except Exception as e:
                print(f"Warning: SARIMAX append failed at {current_date}: {e}")

        else:
            gjr_sigma_1 = np.nan
            lower_95 = np.nan
            upper_95 = np.nan

        actuals.append(y_true)
        rw_preds.append(rw_pred)
        arx_preds.append(arx_pred)
        arx_gjr_preds.append(arx_pred)
        arx_gjr_sigmas.append(gjr_sigma_1)
        arx_gjr_lower_95.append(lower_95)
        arx_gjr_upper_95.append(upper_95)

        if (i + 1) % 25 == 0:
            print(f"{name}: processed {i + 1}/{len(rolling_dates)} forecasts")

    # 4) Store forecasts
    fcst = pd.DataFrame({
        "actual": actuals,
        "RW_forecast": rw_preds,
        "ARX_forecast": arx_preds,
        "ARX_GJR_forecast": arx_gjr_preds,
        "ARX_GJR_sigma": arx_gjr_sigmas,
        "ARX_GJR_lower_95": arx_gjr_lower_95,
        "ARX_GJR_upper_95": arx_gjr_upper_95
    }, index=rolling_dates).dropna()

    rolling_forecasts[name] = fcst

    # 5) Evaluation
    eval_table = evaluate_forecasts(
        fcst,
        actual_col="actual",
        pred_cols=["RW_forecast", "ARX_forecast", "ARX_GJR_forecast"]
    )

    eval_table["segment"] = name
    eval_table["method"] = f"Rolling 1-step ARX, GJR vol static, refit every {REFIT_EVERY} weeks"

    coverage_95 = (
        (fcst["actual"] >= fcst["ARX_GJR_lower_95"]) &
        (fcst["actual"] <= fcst["ARX_GJR_upper_95"])
    ).mean()

    eval_table["GJR_95_Coverage"] = np.nan

    if "model" in eval_table.columns:
        eval_table.loc[
            eval_table["model"] == "ARX_GJR_forecast",
            "GJR_95_Coverage"
        ] = coverage_95

    rolling_evaluations[name] = eval_table

    print("\nRolling evaluation:")
    display(eval_table.round(4))

    print(f"\n95% GJR prediction interval coverage for {name}: {coverage_95:.2%}")

# 6) Combine results
rolling_eval_all = pd.concat(
    rolling_evaluations.values(),
    axis=0
).reset_index(drop=True)

rolling_forecasts_all = []

for segment, df_fcst in rolling_forecasts.items():
    temp = df_fcst.copy()
    temp["segment"] = segment
    temp["date"] = temp.index
    rolling_forecasts_all.append(temp)

rolling_forecasts_all = pd.concat(
    rolling_forecasts_all,
    axis=0
).reset_index(drop=True)

print("\nCombined rolling evaluation:")
display(rolling_eval_all.round(4))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Results tables and plots

# COMMAND ----------

# ==========================================================
# COMBINED EVALUATION TABLE
# ==========================================================

extension_forecast_eval = pd.concat(
    list(static_evaluations.values()) + list(rolling_evaluations.values()),
    axis=0,
    ignore_index=True
)

extension_forecast_eval["model"] = extension_forecast_eval["model"].replace({
    "RW_forecast": "No-change benchmark",
    "ARX_forecast": "ARX / ARMAX",
    "ARX_GJR_forecast": "ARX-GJR-GARCH"
})

extension_forecast_eval = extension_forecast_eval[
    ["segment", "method", "model", "RMSE", "MAE", "Directional_Accuracy", "GJR_95_Coverage"]
    if "GJR_95_Coverage" in extension_forecast_eval.columns
    else ["segment", "method", "model", "RMSE", "MAE", "Directional_Accuracy"]
]

print("Extension forecast evaluation summary:")
display(extension_forecast_eval.round(4))


rmse_pivot = extension_forecast_eval.pivot_table(
    index=["method", "model"],
    columns="segment",
    values="RMSE"
)

mae_pivot = extension_forecast_eval.pivot_table(
    index=["method", "model"],
    columns="segment",
    values="MAE"
)

da_pivot = extension_forecast_eval.pivot_table(
    index=["method", "model"],
    columns="segment",
    values="Directional_Accuracy"
)

print("RMSE pivot:")
display(rmse_pivot.round(4))

print("MAE pivot:")
display(mae_pivot.round(4))

print("Directional accuracy pivot:")
display(da_pivot.round(4))


# COMMAND ----------

# ==========================================================
# DIEBOLD-MARIANO TESTS AGAINST NO-CHANGE BENCHMARK
# ==========================================================

dm_rows = []

for name in TARGET_SERIES:
    for method, fcst in [
        ("Static OOS", static_forecasts[name]["forecast"]),
        ("Rolling 1-step", rolling_forecasts[name])
    ]:
        dm_arx = diebold_mariano_test(
            y_true=fcst["actual"],
            pred_1=fcst["RW_forecast"],
            pred_2=fcst["ARX_forecast"],
            power=2,
            h=1
        )

        dm_gjr = diebold_mariano_test(
            y_true=fcst["actual"],
            pred_1=fcst["RW_forecast"],
            pred_2=fcst["ARX_GJR_forecast"],
            power=2,
            h=1
        )

        dm_rows.append({
            "segment": name,
            "method": method,
            "comparison": "No-change vs ARX",
            **dm_arx
        })

        dm_rows.append({
            "segment": name,
            "method": method,
            "comparison": "No-change vs ARX-GJR-GARCH",
            **dm_gjr
        })

extension_dm_results = pd.DataFrame(dm_rows)

print("Extension Diebold-Mariano test results:")
display(extension_dm_results.round(4))


# COMMAND ----------

# ==========================================================
# PLOTS: FORECASTS AND CONDITIONAL VOLATILITY
# ==========================================================

for name in TARGET_SERIES:
    static_fcst = static_forecasts[name]["forecast"]
    rolling_fcst = rolling_forecasts[name]

    # Forecast comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(static_fcst.index, static_fcst["actual"], label="Actual")
    axes[0].plot(static_fcst.index, static_fcst["RW_forecast"], label="No-change")
    axes[0].plot(static_fcst.index, static_fcst["ARX_forecast"], label="ARX")
    axes[0].set_title(f"{name} - Static OOS: actual vs ARX forecast")
    axes[0].set_ylabel("Weekly spread change (bp)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rolling_fcst.index, rolling_fcst["actual"], label="Actual")
    axes[1].plot(rolling_fcst.index, rolling_fcst["RW_forecast"], label="No-change")
    axes[1].plot(rolling_fcst.index, rolling_fcst["ARX_forecast"], label="ARX")
    axes[1].set_title(f"{name} - Rolling 1-step: actual vs ARX forecast")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Weekly spread change (bp)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Volatility comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(static_fcst.index, static_fcst["ARX_GJR_sigma"])
    axes[0].set_title(f"{name} - Static ARX-GJR conditional volatility")
    axes[0].set_ylabel("Forecast sigma")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rolling_fcst.index, rolling_fcst["ARX_GJR_sigma"])
    axes[1].set_title(f"{name} - Rolling ARX-GJR conditional volatility")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Forecast sigma")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Export outputs
# MAGIC
# MAGIC The outputs are exported into `tfg_data.gold_data`, following the catalog/schema structure created in the base notebook.

# COMMAND ----------

# ==========================================================
# EXPORT EXTENSION OUTPUTS TO GOLD TABLES
# ==========================================================

# Model dataset
export_to_delta(
    extension_df,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_arx_garch_ig_bbb_bb_b_dataset"
)

# Forecast tables
forecast_exports = []

for name in TARGET_SERIES:
    static_export = static_forecasts[name]["forecast"].copy()
    static_export["segment"] = name
    static_export["method"] = "Static OOS"
    forecast_exports.append(static_export)

    rolling_export = rolling_forecasts[name].copy()
    rolling_export["segment"] = name
    rolling_export["method"] = "Rolling 1-step"
    forecast_exports.append(rolling_export)

extension_forecasts_export = pd.concat(forecast_exports, axis=0)

export_to_delta(
    extension_forecasts_export,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_arx_garch_ig_bbb_bb_b_forecasts"
)

# Evaluation tables
export_to_delta(
    extension_forecast_eval,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_arx_garch_ig_bbb_bb_b_evaluation"
)

export_to_delta(
    extension_dm_results,
    f"{CATALOG}.{GOLD_SCHEMA}.extension_arx_garch_ig_bbb_bb_b_dm_results"
)

print("Exported ARX-GJR-GARCH extension outputs for IG / BBB / BB / B.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Interpretation guide
# MAGIC
# MAGIC Use this wording if the extension performs well:
# MAGIC
# MAGIC > The ARX-GJR-GARCH extension suggests that lagged macro-financial conditions add information beyond the univariate time-series dynamics of corporate credit spreads. The result is especially relevant if the improvement is stronger around BBB, BB or B, since this would support the idea that predictive information is unevenly distributed across rating buckets.
# MAGIC
# MAGIC Use this wording if it does not beat the no-change benchmark:
# MAGIC
# MAGIC > The macro-financial ARX-GJR-GARCH extension does not materially improve point forecast accuracy relative to the no-change benchmark. However, the conditional volatility component remains informative, suggesting that the main quantitative value of the model lies more in risk monitoring and volatility dynamics than in predicting the exact weekly spread change.
# MAGIC
# MAGIC Important caveat:
# MAGIC
# MAGIC > ARX-GJR-GARCH and ARX often have the same point forecast if the GJR component is only used for conditional volatility. In that case, the GJR part should be interpreted through the volatility forecasts and prediction intervals, not through RMSE improvements in the conditional mean.
# MAGIC