# Databricks notebook source
# MAGIC %md
# MAGIC # Extension I — Rating buckets, VAR and regime-switching VAR approximation
# MAGIC
# MAGIC This notebook extends the base TFG workflow by disaggregating corporate credit spreads into rating buckets:
# MAGIC
# MAGIC - BBB
# MAGIC - BB
# MAGIC - Single-B
# MAGIC
# MAGIC The core target remains the same as in the base project: **weekly changes in OAS, expressed in basis points**.
# MAGIC
# MAGIC The notebook implements:
# MAGIC
# MAGIC 1. Data loading from Databricks tables, with CSV fallback.
# MAGIC 2. Weekly transformation and first differences.
# MAGIC 3. Descriptive checks for BBB / BB / B.
# MAGIC 4. VAR estimation and forecasting.
# MAGIC 5. A practical MS-VAR-style approximation based on a latent credit regime.
# MAGIC
# MAGIC Important note: `statsmodels` does not provide a full multivariate Markov-switching VAR estimated by maximum likelihood.  
# MAGIC For this reason, the MS-VAR part below is implemented as a **regime-switching VAR approximation**:
# MAGIC
# MAGIC - estimate a two-regime Markov model on a common credit factor;
# MAGIC - classify calm/stress regimes;
# MAGIC - fit separate VAR dynamics by regime;
# MAGIC - forecast using the regime-specific VAR implied by the current state.
# MAGIC
# MAGIC This is not a perfect academic MS-VAR, but it is practical, transparent and defensible as an undergraduate extension.

# COMMAND ----------

# ==========================================================
# 1. INSTALLS AND IMPORTS
# ==========================================================

# In Databricks, uncomment if needed:
# %pip install statsmodels scikit-learn scipy matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import jarque_bera

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

plt.style.use("default")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

# COMMAND ----------

# ==========================================================
# 2. CONFIGURATION
# ==========================================================

CATALOG = "tfg_data"
SCHEMA = "original_data"

# Main rating-bucket table already loaded in Databricks
RATING_BUCKET_TABLE = "bbb_bb_b_extensiondata"

# Optional macro-financial tables already visible in your environment
MACRO_TABLES = {
    "sp500": "sp500",
    "vix": "vix",
    "treasury_10y": "treasury_10y",
    "treasury_2y": "treasury_2y"
}

WEEK_RULE = "W-FRI"
TRAIN_RATIO = 0.80

# Forecasting controls
RUN_ROLLING_VAR = True
RUN_ROLLING_MSVAR_PROXY = True

# VAR controls
MAX_VAR_LAGS = 8
FORCE_VAR_LAG = None   # set to an integer if you want to force a lag, e.g. 1 or 2

# Output names
OUTPUT_PREFIX = "rating_bucket_extension"

# COMMAND ----------

# ==========================================================
# 3. DATA LOADING HELPERS
# ==========================================================

def load_table_from_databricks_or_csv(table_name, csv_path=None):
    # Load a Databricks table into pandas. If Spark is not available, fall back to local CSV.
    full_name = f"{CATALOG}.{SCHEMA}.{table_name}"

    try:
        df = spark.sql(f"SELECT * FROM {full_name}").toPandas()
        print(f"Loaded from Databricks: {full_name}, shape={df.shape}")
        return df
    except Exception as e:
        print(f"Could not load {full_name} from Databricks.")
        print("Error:", str(e)[:300])

        if csv_path is None:
            csv_path = f"{table_name}.csv"

        df = pd.read_csv(csv_path)
        print(f"Loaded from CSV: {csv_path}, shape={df.shape}")
        return df


def find_date_column(df):
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in ["date", "observation_date", "week_end_date", "datetime", "time"]:
        if candidate in lower_map:
            return lower_map[candidate]

    return df.columns[0]


def find_series_column(df, candidates, exclude_cols=None):
    # Priority: exact case-sensitive, exact case-insensitive, contains candidate.
    if exclude_cols is None:
        exclude_cols = []

    cols = [c for c in df.columns if c not in exclude_cols]

    for cand in candidates:
        if cand in cols:
            return cand

    lower_cols = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]

    for cand in candidates:
        cand_low = cand.lower()
        for c in cols:
            if cand_low in c.lower():
                return c

    return None


def get_first_numeric_column(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []

    for c in df.columns:
        if c in exclude_cols:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            return c

    raise ValueError("No numeric column found.")


def standardize_single_series(df, value_name):
    out = df.copy()
    date_col = find_date_column(out)
    value_col = get_first_numeric_column(out, exclude_cols=[date_col])

    out = out[[date_col, value_col]].copy()
    out = out.rename(columns={date_col: "date", value_col: value_name})
    out["date"] = pd.to_datetime(out["date"])
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    return out

# COMMAND ----------

# ==========================================================
# 4. LOAD AND STANDARDIZE BBB / BB / B DATA
# ==========================================================

raw_buckets = load_table_from_databricks_or_csv(RATING_BUCKET_TABLE)

print("Raw rating-bucket columns:")
print(list(raw_buckets.columns))

date_col = find_date_column(raw_buckets)

# Candidate names based on common FRED / ICE BofA identifiers
rating_candidates = {
    "BBB": [
        "BAMLC0A4CBBB", 
        "BBB_OAS", 
        "BBB", 
        "bbb"
    ],
    "BB": [
        "BAMLH0A1HYBB", 
        "BB_OAS", 
        "HYBB",
        "BB", 
        "bb"
    ],
    "B": [
        "BAMLH0A2HYB", 
        "B_OAS", 
        "SINGLE_B", 
        "Single_B",
        "HYB",
        "B", 
        "b"
    ]
}

bucket_cols = {}
used_cols = [date_col]

for rating, candidates in rating_candidates.items():
    col = find_series_column(raw_buckets, candidates, exclude_cols=used_cols)
    if col is None:
        raise ValueError(
            f"Could not identify column for {rating}. "
            f"Please check raw_buckets columns and update rating_candidates."
        )
    bucket_cols[rating] = col
    used_cols.append(col)

print("Detected columns:")
print(bucket_cols)

buckets_raw = raw_buckets[[date_col] + list(bucket_cols.values())].copy()
buckets_raw = buckets_raw.rename(columns={
    date_col: "date",
    bucket_cols["BBB"]: "BBB",
    bucket_cols["BB"]: "BB",
    bucket_cols["B"]: "B"
})

buckets_raw["date"] = pd.to_datetime(buckets_raw["date"])

for c in ["BBB", "BB", "B"]:
    buckets_raw[c] = pd.to_numeric(buckets_raw[c], errors="coerce")

buckets_raw = buckets_raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)

print("Standardized rating-bucket data:")
try:
    display(buckets_raw.head())
except Exception:
    print(buckets_raw.head())
print(buckets_raw.tail())

# COMMAND ----------

# ==========================================================
# 5. OPTIONAL MACRO-FINANCIAL DATA
# ==========================================================
# These variables are not necessary for the pure VAR on rating buckets.
# They are useful as state variables, interpretation variables, or later VARX/ARX extensions.

macro_data = {}

for clean_name, table_name in MACRO_TABLES.items():
    try:
        raw_macro = load_table_from_databricks_or_csv(table_name)
        macro_data[clean_name] = standardize_single_series(raw_macro, clean_name)
    except Exception as e:
        print(f"Skipping {clean_name}. Reason:", str(e)[:300])

for name, df in macro_data.items():
    print(f"\n{name}:")
    try:
        display(df.head())
    except Exception:
        print(df.head())

# COMMAND ----------

# ==========================================================
# 6. WEEKLY TRANSFORMATION HELPERS
# ==========================================================

def build_weekly_rating_buckets(df, week_rule="W-FRI"):
    # Convert daily rating-bucket OAS series into weekly average levels and weekly changes.
    data = df.copy()
    data = data.set_index("date").asfreq("D")

    for col in ["BBB", "BB", "B"]:
        data[col] = (
            data[col]
            .interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )

    weekly = data.resample(week_rule).mean(numeric_only=True)

    # Convert from percentage points to basis points
    for col in ["BBB", "BB", "B"]:
        weekly[f"{col}_bp"] = weekly[col] * 100
        weekly[f"d{col}"] = weekly[f"{col}_bp"].diff()

    # Cross-rating differentials
    weekly["BBB_BB_gap_bp"] = weekly["BB_bp"] - weekly["BBB_bp"]
    weekly["BB_B_gap_bp"] = weekly["B_bp"] - weekly["BB_bp"]
    weekly["BBB_B_gap_bp"] = weekly["B_bp"] - weekly["BBB_bp"]

    # Gap changes
    weekly["d_BBB_BB_gap"] = weekly["BBB_BB_gap_bp"].diff()
    weekly["d_BB_B_gap"] = weekly["BB_B_gap_bp"].diff()
    weekly["d_BBB_B_gap"] = weekly["BBB_B_gap_bp"].diff()

    weekly = weekly.reset_index().rename(columns={"date": "week_end_date"})

    return weekly


def build_weekly_macro(macro_data, week_rule="W-FRI"):
    weekly_macro = None

    for name, df in macro_data.items():
        tmp = df.copy().set_index("date").asfreq("D")
        tmp[name] = (
            tmp[name]
            .interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )

        # For market variables, weekly Friday/last value is usually more natural.
        w = tmp.resample(week_rule).last()

        if weekly_macro is None:
            weekly_macro = w
        else:
            weekly_macro = weekly_macro.join(w, how="outer")

    if weekly_macro is None:
        return None

    if "sp500" in weekly_macro.columns:
        weekly_macro["sp500_ret"] = np.log(weekly_macro["sp500"]).diff()

    if "vix" in weekly_macro.columns:
        weekly_macro["vix_diff"] = weekly_macro["vix"].diff()

    if "treasury_10y" in weekly_macro.columns and "treasury_2y" in weekly_macro.columns:
        weekly_macro["term_spread_10y_2y"] = weekly_macro["treasury_10y"] - weekly_macro["treasury_2y"]
        weekly_macro["d_term_spread_10y_2y"] = weekly_macro["term_spread_10y_2y"].diff()

    weekly_macro = weekly_macro.reset_index().rename(columns={"date": "week_end_date"})
    return weekly_macro

# COMMAND ----------

# ==========================================================
# 7. BUILD WEEKLY EXTENSION DATASET
# ==========================================================

rating_weekly = build_weekly_rating_buckets(buckets_raw, week_rule=WEEK_RULE)
macro_weekly = build_weekly_macro(macro_data, week_rule=WEEK_RULE)

if macro_weekly is not None:
    extension_weekly = pd.merge(rating_weekly, macro_weekly, on="week_end_date", how="left")
else:
    extension_weekly = rating_weekly.copy()

# Main VAR target: weekly changes in rating-specific OAS
y_cols = ["dBBB", "dBB", "dB"]

# Remove the first-difference missing value
extension_model_data = extension_weekly.dropna(subset=y_cols).copy()
extension_model_data = extension_model_data.sort_values("week_end_date").reset_index(drop=True)

print("Extension weekly data shape:", extension_weekly.shape)
print("Model data shape:", extension_model_data.shape)

try:
    display(extension_model_data.head())
    display(extension_model_data.tail())
except Exception:
    print(extension_model_data.head())
    print(extension_model_data.tail())

# COMMAND ----------

# ==========================================================
# 8. EDA: LEVELS, CHANGES, GAPS AND DESCRIPTIVE STATISTICS
# ==========================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(extension_weekly["week_end_date"], extension_weekly["BBB_bp"], label="BBB")
axes[0].plot(extension_weekly["week_end_date"], extension_weekly["BB_bp"], label="BB")
axes[0].plot(extension_weekly["week_end_date"], extension_weekly["B_bp"], label="B")
axes[0].set_title("Rating-bucket OAS levels, weekly average, basis points")
axes[0].set_ylabel("OAS (bp)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(extension_model_data["week_end_date"], extension_model_data["dBBB"], label="ΔBBB")
axes[1].plot(extension_model_data["week_end_date"], extension_model_data["dBB"], label="ΔBB")
axes[1].plot(extension_model_data["week_end_date"], extension_model_data["dB"], label="ΔB")
axes[1].set_title("Weekly changes in rating-bucket OAS")
axes[1].set_ylabel("Change (bp)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(extension_weekly["week_end_date"], extension_weekly["BBB_BB_gap_bp"], label="BB - BBB")
axes[2].plot(extension_weekly["week_end_date"], extension_weekly["BB_B_gap_bp"], label="B - BB")
axes[2].plot(extension_weekly["week_end_date"], extension_weekly["BBB_B_gap_bp"], label="B - BBB")
axes[2].set_title("Cross-rating OAS gaps")
axes[2].set_ylabel("Gap (bp)")
axes[2].set_xlabel("Date")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


def descriptive_stats(series):
    return pd.Series({
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "p25": series.quantile(0.25),
        "median": series.median(),
        "p75": series.quantile(0.75),
        "max": series.max(),
        "skewness": series.skew(),
        "kurtosis": series.kurt()
    })

desc_levels = pd.concat([
    descriptive_stats(extension_weekly["BBB_bp"]).rename("BBB_bp"),
    descriptive_stats(extension_weekly["BB_bp"]).rename("BB_bp"),
    descriptive_stats(extension_weekly["B_bp"]).rename("B_bp")
], axis=1).T

desc_changes = pd.concat([
    descriptive_stats(extension_model_data["dBBB"]).rename("dBBB"),
    descriptive_stats(extension_model_data["dBB"]).rename("dBB"),
    descriptive_stats(extension_model_data["dB"]).rename("dB")
], axis=1).T

print("Descriptive statistics - levels:")
try:
    display(desc_levels.round(3))
except Exception:
    print(desc_levels.round(3))

print("Descriptive statistics - weekly changes:")
try:
    display(desc_changes.round(3))
except Exception:
    print(desc_changes.round(3))

# COMMAND ----------

# ==========================================================
# 9. PRELIMINARY CHECKS: CORRELATION AND STATIONARITY
# ==========================================================

y_data = extension_model_data.set_index("week_end_date")[y_cols].copy()

print("Correlation matrix of weekly changes:")
try:
    display(y_data.corr().round(3))
except Exception:
    print(y_data.corr().round(3))

plt.figure(figsize=(7, 5))
plt.imshow(y_data.corr(), aspect="auto")
plt.xticks(range(len(y_cols)), y_cols)
plt.yticks(range(len(y_cols)), y_cols)
plt.colorbar(label="Correlation")
plt.title("Correlation matrix: ΔBBB, ΔBB, ΔB")
plt.tight_layout()
plt.show()

# ADF tests
adf_rows = []
for col in y_cols:
    x = y_data[col].dropna()
    result = adfuller(x, autolag="AIC")
    adf_rows.append({
        "series": col,
        "ADF_stat": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3]
    })

adf_table = pd.DataFrame(adf_rows)
print("ADF tests on weekly changes:")
try:
    display(adf_table.round(4))
except Exception:
    print(adf_table.round(4))

# ACF plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, y_cols):
    plot_acf(y_data[col].dropna(), lags=20, ax=ax)
    ax.set_title(f"ACF: {col}")
plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# 10. TRAIN / TEST SPLIT
# ==========================================================

split_idx = int(len(y_data) * TRAIN_RATIO)

y_train = y_data.iloc[:split_idx].copy()
y_test = y_data.iloc[split_idx:].copy()

print("Train observations:", len(y_train))
print("Test observations:", len(y_test))
print("Train period:", y_train.index.min().date(), "to", y_train.index.max().date())
print("Test period: ", y_test.index.min().date(), "to", y_test.index.max().date())

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, col in zip(axes, y_cols):
    ax.plot(y_train.index, y_train[col], label="Train")
    ax.plot(y_test.index, y_test[col], label="Test")
    ax.axvline(y_test.index[0], linestyle="--")
    ax.set_title(f"{col}: train/test split")
    ax.set_ylabel("Weekly change (bp)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# 11. EVALUATION HELPERS
# ==========================================================

def evaluate_multivariate_forecast(y_true, y_pred, model_name):
    y_true = y_true.copy()
    y_pred = y_pred.copy()
    y_pred = y_pred.reindex(y_true.index)

    rows = []
    for col in y_true.columns:
        rmse = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
        mae = mean_absolute_error(y_true[col], y_pred[col])
        rows.append({
            "model": model_name,
            "series": col,
            "RMSE": rmse,
            "MAE": mae
        })

    all_errors = (y_true.values - y_pred.values)
    avg_rmse = np.sqrt(np.mean(all_errors**2))
    avg_mae = np.mean(np.abs(all_errors))

    rows.append({
        "model": model_name,
        "series": "average",
        "RMSE": avg_rmse,
        "MAE": avg_mae
    })

    return pd.DataFrame(rows)


def diebold_mariano_multivariate(y_true, pred_1, pred_2, power=2, h=1):
    # Multivariate Diebold-Mariano test using average loss across buckets at each time t.
    # Positive mean_loss_diff means model 2 has lower loss than model 1.
    y_true = y_true.copy()
    pred_1 = pred_1.reindex(y_true.index)
    pred_2 = pred_2.reindex(y_true.index)

    e1 = y_true.values - pred_1.values
    e2 = y_true.values - pred_2.values

    loss1 = np.mean(np.abs(e1)**power, axis=1)
    loss2 = np.mean(np.abs(e2)**power, axis=1)

    d = loss1 - loss2
    d = pd.Series(d).dropna().values

    T = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)

    if h > 1:
        gammas = []
        for lag in range(1, h):
            cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
            gammas.append(cov)
        var_d = gamma0 + 2 * np.sum(gammas)
    else:
        var_d = gamma0

    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "DM_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_bar
    }


def make_zero_forecast(index, columns):
    # Random Walk / no-change benchmark for weekly changes.
    return pd.DataFrame(0.0, index=index, columns=columns)

# COMMAND ----------

# ==========================================================
# 12. VAR MODEL: LAG SELECTION AND IN-SAMPLE FIT
# ==========================================================

var_selector = VAR(y_train)
lag_selection = var_selector.select_order(maxlags=MAX_VAR_LAGS)

print("VAR lag-order selection:")
print(lag_selection.summary())

if FORCE_VAR_LAG is not None:
    selected_var_lag = FORCE_VAR_LAG
else:
    selected_var_lag = lag_selection.aic
    if selected_var_lag is None or selected_var_lag < 1:
        selected_var_lag = 1

print(f"Selected VAR lag: {selected_var_lag}")

var_model = VAR(y_train).fit(selected_var_lag)
print(var_model.summary())

# Residual diagnostics
var_resid = var_model.resid

print("VAR residual correlation:")
try:
    display(var_resid.corr().round(3))
except Exception:
    print(var_resid.corr().round(3))

for col in var_resid.columns:
    print(f"\nLjung-Box test on VAR residuals: {col}")
    lb = acorr_ljungbox(var_resid[col].dropna(), lags=[4, 8, 12], return_df=True).round(4)
    try:
        display(lb)
    except Exception:
        print(lb)

    jb_stat, jb_pvalue = jarque_bera(var_resid[col].dropna())
    print(f"Jarque-Bera {col}: stat={jb_stat:.4f}, p-value={jb_pvalue:.6f}")

# COMMAND ----------

# ==========================================================
# 13. VAR STATIC OUT-OF-SAMPLE FORECAST
# ==========================================================

h = len(y_test)

rw_static_pred = make_zero_forecast(y_test.index, y_cols)

last_train_values = y_train.values[-selected_var_lag:]
var_static_values = var_model.forecast(y=last_train_values, steps=h)

var_static_pred = pd.DataFrame(
    var_static_values,
    index=y_test.index,
    columns=y_cols
)

eval_rw_static = evaluate_multivariate_forecast(y_test, rw_static_pred, "Random Walk")
eval_var_static = evaluate_multivariate_forecast(y_test, var_static_pred, f"VAR({selected_var_lag})")

var_static_eval = pd.concat([eval_rw_static, eval_var_static], ignore_index=True)

print("VAR static OOS evaluation:")
try:
    display(var_static_eval.round(4))
except Exception:
    print(var_static_eval.round(4))

dm_var_static = diebold_mariano_multivariate(
    y_true=y_test,
    pred_1=rw_static_pred,
    pred_2=var_static_pred,
    power=2,
    h=1
)

print("Multivariate DM test: Random Walk vs VAR static forecast")
print(pd.Series(dm_var_static).round(4))

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, col in zip(axes, y_cols):
    ax.plot(y_test.index, y_test[col], label="Actual")
    ax.plot(rw_static_pred.index, rw_static_pred[col], label="Random Walk")
    ax.plot(var_static_pred.index, var_static_pred[col], label=f"VAR({selected_var_lag})")
    ax.set_title(f"Static OOS forecast: {col}")
    ax.set_ylabel("Weekly change (bp)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# 14. VAR ROLLING ONE-STEP-AHEAD FORECAST
# ==========================================================

if RUN_ROLLING_VAR:
    var_rolling_preds = []

    for i, current_date in enumerate(y_test.index):
        y_hist = y_data.iloc[:split_idx + i].copy()

        try:
            model_i = VAR(y_hist).fit(selected_var_lag)
            forecast_i = model_i.forecast(y=y_hist.values[-selected_var_lag:], steps=1)[0]
        except Exception as e:
            forecast_i = np.zeros(len(y_cols))

        var_rolling_preds.append(forecast_i)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(y_test)} rolling VAR forecasts")

    var_rolling_pred = pd.DataFrame(
        var_rolling_preds,
        index=y_test.index,
        columns=y_cols
    )

    rw_rolling_pred = make_zero_forecast(y_test.index, y_cols)

    eval_rw_rolling = evaluate_multivariate_forecast(y_test, rw_rolling_pred, "Random Walk")
    eval_var_rolling = evaluate_multivariate_forecast(y_test, var_rolling_pred, f"Rolling VAR({selected_var_lag})")

    var_rolling_eval = pd.concat([eval_rw_rolling, eval_var_rolling], ignore_index=True)

    print("VAR rolling 1-step evaluation:")
    try:
        display(var_rolling_eval.round(4))
    except Exception:
        print(var_rolling_eval.round(4))

    dm_var_rolling = diebold_mariano_multivariate(
        y_true=y_test,
        pred_1=rw_rolling_pred,
        pred_2=var_rolling_pred,
        power=2,
        h=1
    )

    print("Multivariate DM test: Random Walk vs rolling VAR")
    print(pd.Series(dm_var_rolling).round(4))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, col in zip(axes, y_cols):
        ax.plot(y_test.index, y_test[col], label="Actual")
        ax.plot(rw_rolling_pred.index, rw_rolling_pred[col], label="Random Walk")
        ax.plot(var_rolling_pred.index, var_rolling_pred[col], label=f"Rolling VAR({selected_var_lag})")
        ax.set_title(f"Rolling one-step forecast: {col}")
        ax.set_ylabel("Weekly change (bp)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Rolling VAR skipped because RUN_ROLLING_VAR=False")

# COMMAND ----------

# ==========================================================
# 15. REGIME FACTOR FOR MS-VAR-STYLE APPROXIMATION
# ==========================================================

scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)

pca = PCA(n_components=1)
factor_train = pd.Series(
    pca.fit_transform(y_train_scaled).ravel(),
    index=y_train.index,
    name="credit_factor"
)

# Orient the factor so high values roughly correspond to wider spreads / stress.
avg_change_train = y_train.mean(axis=1)
flip_factor_sign = False

if factor_train.corr(avg_change_train) < 0:
    factor_train = -factor_train
    flip_factor_sign = True

print("Explained variance by first credit factor:", round(pca.explained_variance_ratio_[0], 4))
print("Factor sign flipped:", flip_factor_sign)

plt.figure(figsize=(14, 4))
plt.plot(factor_train.index, factor_train)
plt.title("Common credit factor from ΔBBB, ΔBB and ΔB")
plt.ylabel("Standardized factor")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

regime_model = MarkovRegression(
    factor_train,
    k_regimes=2,
    trend="c",
    switching_variance=True
).fit(disp=False)

print(regime_model.summary())

regime_probs = pd.DataFrame(
    regime_model.smoothed_marginal_probabilities,
    index=factor_train.index
)

regime_probs.columns = [f"Regime_{i}" for i in regime_probs.columns]

regime_labels = regime_probs.idxmax(axis=1).str.replace("Regime_", "").astype(int)
regime_means = factor_train.groupby(regime_labels).mean()
stress_regime = int(regime_means.idxmax())
calm_regime = 1 - stress_regime

print("Calm regime:", calm_regime)
print("Stress regime:", stress_regime)
print("Regime factor means:")
print(regime_means)

plt.figure(figsize=(14, 4))
plt.plot(regime_probs.index, regime_probs[f"Regime_{calm_regime}"], label="Calm regime probability")
plt.plot(regime_probs.index, regime_probs[f"Regime_{stress_regime}"], label="Stress regime probability")
plt.title("Estimated regime probabilities from common credit factor")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# 16. FIT REGIME-SPECIFIC VAR MODELS
# ==========================================================
# Practical MS-VAR proxy:
# - assign each training observation to the most likely regime;
# - fit one VAR for calm observations and one VAR for stress observations.
#
# This is not a full MLE MS-VAR, but it captures regime-dependent multivariate dynamics.

train_with_regime = y_train.copy()
train_with_regime["regime"] = regime_labels.reindex(y_train.index)

def fit_var_safe(data, lag, fallback_data):
    data = data[y_cols].dropna()
    min_obs = max(30, lag * len(y_cols) + 10)

    if len(data) < min_obs:
        print(f"Not enough observations for regime-specific VAR. Using global VAR. n={len(data)}")
        return VAR(fallback_data).fit(lag)

    try:
        return VAR(data).fit(lag)
    except Exception as e:
        print("Regime VAR failed. Using global VAR. Error:", str(e)[:200])
        return VAR(fallback_data).fit(lag)

calm_data = train_with_regime[train_with_regime["regime"] == calm_regime][y_cols]
stress_data = train_with_regime[train_with_regime["regime"] == stress_regime][y_cols]

print("Calm observations:", len(calm_data))
print("Stress observations:", len(stress_data))

var_calm = fit_var_safe(calm_data, selected_var_lag, y_train)
var_stress = fit_var_safe(stress_data, selected_var_lag, y_train)

print("Calm VAR summary:")
print(var_calm.summary())

print("Stress VAR summary:")
print(var_stress.summary())

# COMMAND ----------

# ==========================================================
# 17. MS-VAR PROXY STATIC FORECAST
# ==========================================================

last_regime = int(regime_labels.iloc[-1])
selected_regime_model = var_stress if last_regime == stress_regime else var_calm

print("Last in-sample regime:", last_regime)
print("Using:", "stress VAR" if last_regime == stress_regime else "calm VAR")

msvar_static_values = selected_regime_model.forecast(
    y=y_train.values[-selected_var_lag:],
    steps=len(y_test)
)

msvar_static_pred = pd.DataFrame(
    msvar_static_values,
    index=y_test.index,
    columns=y_cols
)

eval_msvar_static = evaluate_multivariate_forecast(
    y_test,
    msvar_static_pred,
    "MS-VAR proxy static"
)

msvar_static_eval = pd.concat([eval_rw_static, eval_var_static, eval_msvar_static], ignore_index=True)

print("Static OOS evaluation: RW vs VAR vs MS-VAR proxy")
try:
    display(msvar_static_eval.round(4))
except Exception:
    print(msvar_static_eval.round(4))

dm_msvar_static = diebold_mariano_multivariate(
    y_true=y_test,
    pred_1=rw_static_pred,
    pred_2=msvar_static_pred,
    power=2,
    h=1
)

print("Multivariate DM test: Random Walk vs MS-VAR proxy static forecast")
print(pd.Series(dm_msvar_static).round(4))

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, col in zip(axes, y_cols):
    ax.plot(y_test.index, y_test[col], label="Actual")
    ax.plot(rw_static_pred.index, rw_static_pred[col], label="Random Walk")
    ax.plot(var_static_pred.index, var_static_pred[col], label=f"VAR({selected_var_lag})")
    ax.plot(msvar_static_pred.index, msvar_static_pred[col], label="MS-VAR proxy")
    ax.set_title(f"Static OOS forecast with MS-VAR proxy: {col}")
    ax.set_ylabel("Weekly change (bp)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# 18. MS-VAR PROXY ROLLING ONE-STEP FORECAST
# ==========================================================

def classify_latest_regime_from_factor(factor_value, regime_means, calm_regime, stress_regime):
    dist_calm = abs(factor_value - regime_means.loc[calm_regime])
    dist_stress = abs(factor_value - regime_means.loc[stress_regime])
    return stress_regime if dist_stress < dist_calm else calm_regime


if RUN_ROLLING_MSVAR_PROXY:
    msvar_rolling_preds = []
    selected_regimes_rolling = []

    for i, current_date in enumerate(y_test.index):
        y_hist = y_data.iloc[:split_idx + i].copy()

        latest_scaled = scaler.transform(y_hist.iloc[[-1]])
        latest_factor = float(pca.transform(latest_scaled).ravel()[0])

        if flip_factor_sign:
            latest_factor = -latest_factor

        current_regime = classify_latest_regime_from_factor(
            latest_factor,
            regime_means,
            calm_regime,
            stress_regime
        )

        model_i = var_stress if current_regime == stress_regime else var_calm

        try:
            forecast_i = model_i.forecast(y=y_hist.values[-selected_var_lag:], steps=1)[0]
        except Exception:
            forecast_i = var_model.forecast(y=y_hist.values[-selected_var_lag:], steps=1)[0]

        msvar_rolling_preds.append(forecast_i)
        selected_regimes_rolling.append(current_regime)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(y_test)} MS-VAR proxy rolling forecasts")

    msvar_rolling_pred = pd.DataFrame(
        msvar_rolling_preds,
        index=y_test.index,
        columns=y_cols
    )

    msvar_rolling_regimes = pd.Series(
        selected_regimes_rolling,
        index=y_test.index,
        name="selected_regime"
    )

    eval_msvar_rolling = evaluate_multivariate_forecast(
        y_test,
        msvar_rolling_pred,
        "Rolling MS-VAR proxy"
    )

    if RUN_ROLLING_VAR:
        msvar_rolling_eval = pd.concat([eval_rw_rolling, eval_var_rolling, eval_msvar_rolling], ignore_index=True)
    else:
        msvar_rolling_eval = pd.concat([
            evaluate_multivariate_forecast(y_test, make_zero_forecast(y_test.index, y_cols), "Random Walk"),
            eval_msvar_rolling
        ], ignore_index=True)

    print("Rolling 1-step evaluation: RW vs VAR vs MS-VAR proxy")
    try:
        display(msvar_rolling_eval.round(4))
    except Exception:
        print(msvar_rolling_eval.round(4))

    dm_msvar_rolling = diebold_mariano_multivariate(
        y_true=y_test,
        pred_1=make_zero_forecast(y_test.index, y_cols),
        pred_2=msvar_rolling_pred,
        power=2,
        h=1
    )

    print("Multivariate DM test: Random Walk vs rolling MS-VAR proxy")
    print(pd.Series(dm_msvar_rolling).round(4))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, col in zip(axes, y_cols):
        ax.plot(y_test.index, y_test[col], label="Actual")
        ax.plot(msvar_rolling_pred.index, msvar_rolling_pred[col], label="MS-VAR proxy")
        if RUN_ROLLING_VAR:
            ax.plot(var_rolling_pred.index, var_rolling_pred[col], label=f"Rolling VAR({selected_var_lag})", alpha=0.8)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"Rolling one-step forecast with MS-VAR proxy: {col}")
        ax.set_ylabel("Weekly change (bp)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 3))
    plt.plot(msvar_rolling_regimes.index, msvar_rolling_regimes.values, drawstyle="steps-post")
    plt.title("Rolling selected regime for MS-VAR proxy")
    plt.ylabel("Regime")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

else:
    print("Rolling MS-VAR proxy skipped because RUN_ROLLING_MSVAR_PROXY=False")

# COMMAND ----------

# ==========================================================
# 19. FINAL COMPARISON TABLES
# ==========================================================

final_static_eval = msvar_static_eval.copy()
final_static_eval["method"] = "Static OOS"

if RUN_ROLLING_MSVAR_PROXY:
    final_rolling_eval = msvar_rolling_eval.copy()
    final_rolling_eval["method"] = "Rolling 1-step"
else:
    final_rolling_eval = pd.DataFrame()

final_extension_eval = pd.concat([final_static_eval, final_rolling_eval], ignore_index=True)
final_extension_eval = final_extension_eval[["method", "model", "series", "RMSE", "MAE"]]

print("Final extension evaluation table:")
try:
    display(final_extension_eval.round(4))
except Exception:
    print(final_extension_eval.round(4))

avg_summary = final_extension_eval[final_extension_eval["series"] == "average"].copy()
print("Average performance across BBB, BB and B:")
try:
    display(avg_summary.round(4))
except Exception:
    print(avg_summary.round(4))

extension_weekly.to_csv(f"{OUTPUT_PREFIX}_weekly_data.csv", index=False)
final_extension_eval.to_csv(f"{OUTPUT_PREFIX}_forecast_evaluation.csv", index=False)

print("Saved:")
print(f"- {OUTPUT_PREFIX}_weekly_data.csv")
print(f"- {OUTPUT_PREFIX}_forecast_evaluation.csv")

# COMMAND ----------

# ==========================================================
# 20. OPTIONAL: MACRO-FINANCIAL BLOCK FOR FUTURE VARX / ARX
# ==========================================================
# This cell does not estimate a VARX model, because the core requested extension is BBB/BB/B + VAR/MS-VAR.
# However, it prepares a clean macro block for a later forecasting-performance extension.

if macro_weekly is not None:
    macro_candidate_cols = [
        c for c in [
            "sp500_ret",
            "vix",
            "vix_diff",
            "treasury_10y",
            "treasury_2y",
            "term_spread_10y_2y",
            "d_term_spread_10y_2y"
        ]
        if c in extension_model_data.columns
    ]

    if len(macro_candidate_cols) > 0:
        macro_block = extension_model_data[["week_end_date"] + macro_candidate_cols].copy()
        print("Prepared macro-financial variables for future VARX/ARX extension:")
        try:
            display(macro_block.head())
            display(macro_block.describe().T.round(4))
        except Exception:
            print(macro_block.head())
            print(macro_block.describe().T.round(4))

        corr_block = extension_model_data[y_cols + macro_candidate_cols].corr()
        print("Correlation between rating-bucket changes and macro-financial variables:")
        try:
            display(corr_block.loc[y_cols, macro_candidate_cols].round(3))
        except Exception:
            print(corr_block.loc[y_cols, macro_candidate_cols].round(3))
    else:
        print("No macro candidate columns found after transformation.")
else:
    print("No macro data available.")