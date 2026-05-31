# Databricks notebook source
# MAGIC %md
# MAGIC # IG + AAA / AA / A investment-grade rating buckets extension — Base univariate models
# MAGIC
# MAGIC This notebook reproduces the base modelling logic of the TFG for a more granular decomposition inside the investment-grade segment:
# MAGIC
# MAGIC - IG aggregate
# MAGIC - AAA
# MAGIC - AA
# MAGIC - A
# MAGIC
# MAGIC The target variable remains the same as in the base project: **weekly changes in OAS, expressed in basis points**.
# MAGIC
# MAGIC Models included:
# MAGIC
# MAGIC 1. Random Walk / no-change benchmark
# MAGIC 2. AR(1)
# MAGIC 3. ARMA(p,q), selected by AIC on the training sample
# MAGIC 4. ARMA-GARCH(1,1)
# MAGIC 5. ARMA-EGARCH(1,1)
# MAGIC 6. ARMA-GJR-GARCH(1,1)
# MAGIC 7. Markov-Switching as an optional extension
# MAGIC
# MAGIC Additional diagnostic:
# MAGIC
# MAGIC - ADF and KPSS stationarity tests for all four series, both in OAS levels and weekly spread changes.
# MAGIC
# MAGIC Forecasting is implemented for the relevant simple comparison: Random Walk, selected ARMA(p,q), and ARMA(p,q)-GJR-GARCH(1,1). In the two-step ARMA-GJR-GARCH setup, the point forecast comes from the ARMA mean equation, while GJR-GARCH provides a conditional volatility forecast.
# MAGIC
# MAGIC Data note: the AAA, AA and A series are loaded directly from FRED by default. FRED currently notes that, starting in April 2026, these ICE BofA series will only include three years of observations. If you already have longer historical data saved in Databricks, set `USE_FRED_DIRECT_DOWNLOAD = False` and adapt the loading cell accordingly.
# MAGIC

# COMMAND ----------

# MAGIC %pip install statsmodels

# COMMAND ----------

# MAGIC %pip install arch

# COMMAND ----------

# ==========================================================
# 1. IMPORTS AND CONFIGURATION
# ==========================================================

# In Databricks, uncomment if needed:
# %pip install statsmodels arch scikit-learn scipy matplotlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import jarque_bera

from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot

from arch import arch_model

plt.style.use("default")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

CATALOG = "tfg_data"
SCHEMA = "original_data"

# Existing aggregate IG table in your Databricks environment.
# The notebook loads IG from FRED by default too, so this is only a fallback.
IG_TABLE = "ig_aggregate_oas"

# Direct FRED download is the cleanest way to test AAA / AA / A quickly.
USE_FRED_DIRECT_DOWNLOAD = True

FRED_SERIES = {
    "IG":  "BAMLC0A0CM",      # ICE BofA US Corporate Index OAS
    "AAA": "BAMLC0A1CAAA",    # ICE BofA AAA US Corporate Index OAS
    "AA":  "BAMLC0A2CAA",     # ICE BofA AA US Corporate Index OAS
    "A":   "BAMLC0A3CA"       # ICE BofA Single-A US Corporate Index OAS
}

WEEK_RULE = "W-FRI"
TRAIN_RATIO = 0.80
N_LAGS = 4

MAX_ARMA_P = 3
MAX_ARMA_Q = 3
GARCH_DIST = "normal"
RUN_MARKOV_SWITCHING = True
RUN_ROLLING_FORECASTS = True
AUTO_DETECT_BP_SCALE = True


# COMMAND ----------

# ==========================================================
# 2. GENERAL HELPERS
# ==========================================================

def display_or_print(obj, name=None):
    if name is not None:
        print(name)
    try:
        display(obj)
    except Exception:
        print(obj)


def load_table_from_databricks_or_csv(table_name, csv_path=None):
    # Load a Databricks table into pandas. If Spark is not available, fall back to CSV.
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
    # Find a column using exact and soft matching.
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


def convert_to_bp(series, name="series"):
    # FRED/ICE BofA OAS series are usually in percentage points, so 3.5 means 350 bps.
    # If values already look like bps, keep them unchanged.
    x = pd.to_numeric(series, errors="coerce")
    med = x.dropna().abs().median()
    if AUTO_DETECT_BP_SCALE and med > 50:
        print(f"{name}: values look already expressed in bps. No multiplication applied.")
        return x
    print(f"{name}: values look expressed in percentage points. Multiplying by 100 to get bps.")
    return x * 100.0


def load_fred_oas_series(series_id, name):
    """
    Loads a FRED series directly from the public CSV endpoint.
    FRED OAS series are usually in percentage points and daily frequency.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df = df.rename(columns={df.columns[0]: "date", series_id: name})
    df["date"] = pd.to_datetime(df["date"])
    df[name] = pd.to_numeric(df[name].replace(".", np.nan), errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    print(f"Loaded from FRED: {name} ({series_id}), shape={df.shape}")
    return df


# COMMAND ----------

# ==========================================================
# 3. LOAD AND STANDARDIZE IG + AAA / AA / A BUCKETS
# ==========================================================

if USE_FRED_DIRECT_DOWNLOAD:
    raw_series = {}
    for name, series_id in FRED_SERIES.items():
        raw_series[name] = load_fred_oas_series(series_id=series_id, name=name)

    ig_raw = raw_series["IG"]
    aaa_raw = raw_series["AAA"]
    aa_raw = raw_series["AA"]
    a_raw = raw_series["A"]

else:
    # Fallback structure if you have the series already stored as Databricks tables.
    # Adapt table names if needed.
    raw_ig = load_table_from_databricks_or_csv(IG_TABLE)

    print("IG raw columns:")
    print(list(raw_ig.columns))

    ig_date_col = find_date_column(raw_ig)
    ig_value_col = find_series_column(
        raw_ig,
        candidates=["BAMLC0A0CM", "IG_OAS", "IG", "investment_grade", "investmentgrade", "value", "oas"],
        exclude_cols=[ig_date_col]
    )
    if ig_value_col is None:
        raise ValueError("Could not identify IG OAS column. Please check raw_ig columns and update candidates.")

    ig_raw = raw_ig[[ig_date_col, ig_value_col]].copy()
    ig_raw = ig_raw.rename(columns={ig_date_col: "date", ig_value_col: "IG"})
    ig_raw["date"] = pd.to_datetime(ig_raw["date"])
    ig_raw["IG"] = pd.to_numeric(ig_raw["IG"], errors="coerce")
    ig_raw = ig_raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # If you do not use FRED, define AAA / AA / A dataframes here with columns: date, AAA / AA / A.
    raise ValueError(
        "USE_FRED_DIRECT_DOWNLOAD=False requires you to provide local Databricks tables for AAA, AA and A. "
        "For quick testing, keep USE_FRED_DIRECT_DOWNLOAD=True."
    )

print("Standardized raw series preview:")
display_or_print(ig_raw.head(), "IG raw standardized preview:")
display_or_print(aaa_raw.head(), "AAA raw standardized preview:")
display_or_print(aa_raw.head(), "AA raw standardized preview:")
display_or_print(a_raw.head(), "A raw standardized preview:")


# COMMAND ----------

# ==========================================================
# 4. WEEKLY TRANSFORMATION
# ==========================================================

from functools import reduce

def build_weekly_multi_series(df, series_cols, week_rule="W-FRI", n_lags=4):
    data = df.copy()
    data = data.set_index("date").asfreq("D")
    for col in series_cols:
        data[col] = data[col].interpolate(method="time", limit_direction="both").ffill().bfill()

    weekly = data.resample(week_rule).mean(numeric_only=True)

    for col in series_cols:
        weekly[f"{col}_bp"] = convert_to_bp(weekly[col], name=col)
        weekly[f"d{col}"] = weekly[f"{col}_bp"].diff()
        for lag in range(1, n_lags + 1):
            weekly[f"{col}_bp_lag{lag}"] = weekly[f"{col}_bp"].shift(lag)
            weekly[f"d{col}_lag{lag}"] = weekly[f"d{col}"].shift(lag)

    return weekly.reset_index().rename(columns={"date": "week_end_date"})

# Merge before weekly transformation to guarantee a common calendar across all series.
# This keeps the empirical treatment consistent across IG, AAA, AA and A.
raw_all = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="outer"),
    [ig_raw, aaa_raw, aa_raw, a_raw]
).sort_values("date").reset_index(drop=True)

series_names = ["IG", "AAA", "AA", "A"]
rating_weekly = build_weekly_multi_series(raw_all, series_cols=series_names, week_rule=WEEK_RULE, n_lags=N_LAGS)

series_config = {
    "IG": {"level_col": "IG_bp", "diff_col": "dIG"},
    "AAA": {"level_col": "AAA_bp", "diff_col": "dAAA"},
    "AA": {"level_col": "AA_bp", "diff_col": "dAA"},
    "A": {"level_col": "A_bp", "diff_col": "dA"}
}

model_data = rating_weekly.dropna(subset=[v["diff_col"] for v in series_config.values()]).copy()
model_data = model_data.sort_values("week_end_date").reset_index(drop=True)

print("Weekly data shape:", rating_weekly.shape)
print("Model data shape:", model_data.shape)
display_or_print(model_data.head(), "Model data preview:")


# COMMAND ----------

# ==========================================================
# 5. DESCRIPTIVE COMPARISON: IG / AAA / AA / A
# ==========================================================

plt.figure(figsize=(14, 6))
for name, cfg in series_config.items():
    plt.plot(rating_weekly["week_end_date"], rating_weekly[cfg["level_col"]], label=name)
plt.title("Weekly OAS levels by credit segment")
plt.xlabel("Date")
plt.ylabel("OAS (bps)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

n_series = len(series_config)
fig, axes = plt.subplots(n_series, 1, figsize=(14, 2.7 * n_series), sharex=True)
if n_series == 1:
    axes = [axes]

for ax, (name, cfg) in zip(axes, series_config.items()):
    ax.plot(model_data["week_end_date"], model_data[cfg["diff_col"]])
    ax.set_title(f"{name}: weekly change in OAS")
    ax.set_ylabel("bp")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Date")
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

level_desc = pd.concat([descriptive_stats(model_data[cfg["level_col"]]).rename(name) for name, cfg in series_config.items()], axis=1).T
change_desc = pd.concat([descriptive_stats(model_data[cfg["diff_col"]]).rename(name) for name, cfg in series_config.items()], axis=1).T

print("Descriptive statistics: OAS levels in bps")
display_or_print(level_desc.round(3))
print("Descriptive statistics: weekly OAS changes in bps")
display_or_print(change_desc.round(3))


# COMMAND ----------

# ============================================
# EDA extension before model estimation
# Credit segments: IG, AAA, AA and A
#
# Distribution, normality, autocorrelation,
# volatility clustering and ARCH effects
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import jarque_bera, gaussian_kde
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# --------------------------------------------
# 1) Prepare weekly first-difference series
# --------------------------------------------

bucket_names = list(series_config.keys())

# Preferred option: use series_dict if it already exists in the notebook
# series_dict should contain the weekly spread changes for each segment.
bucket_series_dict = {}

for name in bucket_names:
    if "series_dict" in globals() and name in series_dict:
        bucket_series_dict[name] = series_dict[name].dropna().copy()
    else:
        # Fallback: construct from model_data and series_config
        diff_col = series_config[name]["diff_col"]
        tmp = model_data[["week_end_date", diff_col]].copy()
        tmp["week_end_date"] = pd.to_datetime(tmp["week_end_date"])
        tmp = tmp.dropna(subset=[diff_col]).sort_values("week_end_date")
        bucket_series_dict[name] = tmp.set_index("week_end_date")[diff_col].astype(float)

# --------------------------------------------
# 2) Set analysis parameters
# --------------------------------------------

acf_lags = 20
ljungbox_lags = [4, 8, 12]
arch_lags = 8
rolling_window = 26   # 26 weeks ~ half year

# --------------------------------------------
# 3) Helper function to run tests and plots
# --------------------------------------------

def run_bucket_eda_checks(series, name):
    x = pd.Series(series).dropna().copy()

    print("=" * 90)
    print(f"{name} weekly spread changes (first differences in basis points)")
    print("=" * 90)

    # -------------------------
    # Basic distribution stats
    # -------------------------
    desc = pd.Series({
        "n_obs": x.shape[0],
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "p25": x.quantile(0.25),
        "median": x.median(),
        "p75": x.quantile(0.75),
        "max": x.max(),
        "skewness": x.skew(),
        "kurtosis": x.kurt()
    })

    print("\nDescriptive statistics:")
    print(desc.round(4))

    # -------------------------
    # Jarque-Bera normality test
    # -------------------------
    jb_result = jarque_bera(x)

    try:
        jb_stat = jb_result.statistic
        jb_pvalue = jb_result.pvalue
    except AttributeError:
        jb_stat, jb_pvalue = jb_result

    print("\nJarque-Bera test for normality:")
    print(f"JB statistic = {jb_stat:.4f}")
    print(f"p-value      = {jb_pvalue:.6f}")

    if jb_pvalue < 0.05:
        print("Interpretation: reject normality at the 5% level.")
    else:
        print("Interpretation: cannot reject normality at the 5% level.")

    # -------------------------
    # Ljung-Box on spread changes
    # -------------------------
    lb = acorr_ljungbox(x, lags=ljungbox_lags, return_df=True)

    print("\nLjung-Box test on spread changes:")
    print(lb.round(4))

    # -------------------------
    # Ljung-Box on squared changes
    # Useful for volatility clustering
    # -------------------------
    lb_sq = acorr_ljungbox(x**2, lags=ljungbox_lags, return_df=True)

    print("\nLjung-Box test on squared spread changes:")
    print(lb_sq.round(4))

    # -------------------------
    # ARCH-LM test
    # -------------------------
    try:
        arch_lm_stat, arch_lm_pvalue, arch_f_stat, arch_f_pvalue = het_arch(x, nlags=arch_lags)

        print("\nARCH-LM test:")
        print(f"LM statistic = {arch_lm_stat:.4f}")
        print(f"LM p-value   = {arch_lm_pvalue:.6f}")
        print(f"F statistic  = {arch_f_stat:.4f}")
        print(f"F p-value    = {arch_f_pvalue:.6f}")

        if arch_lm_pvalue < 0.05:
            print("Interpretation: evidence of conditional heteroskedasticity / ARCH effects.")
        else:
            print("Interpretation: no strong evidence of ARCH effects at the 5% level.")

    except Exception as e:
        print("\nARCH-LM test could not be computed:")
        print(e)

    # -----------------------------------------
    # 4) Distribution plots + rolling volatility
    # -----------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} – Distribution and volatility diagnostics", fontsize=14)

    # Histogram + KDE
    axes[0, 0].hist(x, bins=35, density=True, alpha=0.7, edgecolor="black")

    try:
        kde = gaussian_kde(x)
        x_grid = np.linspace(x.min(), x.max(), 300)
        axes[0, 0].plot(x_grid, kde(x_grid), linewidth=2)
    except Exception:
        pass

    axes[0, 0].set_title(f"{name}: histogram and KDE of weekly changes")
    axes[0, 0].set_xlabel("Weekly change (bp)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].grid(True, alpha=0.3)

    # Q-Q plot
    qqplot(x, line="s", ax=axes[0, 1])
    axes[0, 1].set_title(f"{name}: Q-Q plot")

    # Time series of weekly changes
    axes[1, 0].plot(x.index, x.values)
    axes[1, 0].set_title(f"{name}: weekly spread changes")
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Change (bp)")
    axes[1, 0].grid(True, alpha=0.3)

    # Rolling volatility
    rolling_vol = x.rolling(rolling_window).std()
    axes[1, 1].plot(rolling_vol.index, rolling_vol.values)
    axes[1, 1].set_title(f"{name}: rolling volatility ({rolling_window}-week std)")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Std. dev.")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # -------------------------
    # 5) ACF and PACF
    # -------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{name} – Serial dependence diagnostics", fontsize=14)

    plot_acf(x, lags=acf_lags, ax=axes[0])
    axes[0].set_title(f"{name}: ACF of weekly changes")

    plot_pacf(x, lags=acf_lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{name}: PACF of weekly changes")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # 6) Squared changes over time
    # -------------------------

    plt.figure(figsize=(14, 4))
    plt.plot((x**2).index, (x**2).values)
    plt.title(f"{name}: squared weekly changes")
    plt.xlabel("Date")
    plt.ylabel("Squared change")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Return summary row for comparison table
    return {
        "series": name,
        "n_obs": x.shape[0],
        "mean": x.mean(),
        "std": x.std(),
        "skewness": x.skew(),
        "kurtosis": x.kurt(),
        "JB_pvalue": jb_pvalue,
        "LB_change_pvalue_lag12": lb.loc[12, "lb_pvalue"] if 12 in lb.index else np.nan,
        "LB_squared_change_pvalue_lag12": lb_sq.loc[12, "lb_pvalue"] if 12 in lb_sq.index else np.nan,
        "ARCH_LM_pvalue": arch_lm_pvalue if "arch_lm_pvalue" in locals() else np.nan
    }

# --------------------------------------------
# 4) Run diagnostics for IG, AAA, AA and A
# --------------------------------------------

eda_bucket_rows = []

for name, s in bucket_series_dict.items():
    row = run_bucket_eda_checks(s, name)
    eda_bucket_rows.append(row)

eda_bucket_summary = pd.DataFrame(eda_bucket_rows)

print("\nSummary table: IG / AAA / AA / A preliminary diagnostics")
display_or_print(eda_bucket_summary.round(4))


# COMMAND ----------

# ==========================================================
# 5B. STATIONARITY TESTS: ADF AND KPSS
# ==========================================================
# The empirical models are estimated on weekly spread changes.
# This cell reports ADF and KPSS tests for both:
# 1) OAS levels in bps
# 2) weekly OAS changes in bps
#
# Interpretation:
# - ADF null hypothesis: unit root / non-stationarity.
#   Small p-value -> evidence of stationarity.
# - KPSS null hypothesis: stationarity.
#   Small p-value -> evidence against stationarity.
#
# Therefore, the cleanest stationarity evidence is:
# - ADF p-value < 0.05
# - KPSS p-value > 0.05
# ==========================================================

from statsmodels.tsa.stattools import adfuller, kpss


def run_adf_kpss(x, series_name, variable_type):
    x = pd.Series(x).dropna().astype(float)

    row = {
        "series": series_name,
        "variable": variable_type,
        "n_obs": len(x),
        "ADF_stat": np.nan,
        "ADF_pvalue": np.nan,
        "KPSS_stat": np.nan,
        "KPSS_pvalue": np.nan,
        "stationarity_reading": ""
    }

    if len(x) < 20:
        row["stationarity_reading"] = "Too few observations"
        return row

    # ADF test
    try:
        adf_res = adfuller(x, autolag="AIC")
        row["ADF_stat"] = adf_res[0]
        row["ADF_pvalue"] = adf_res[1]
    except Exception as e:
        row["ADF_error"] = str(e)[:120]

    # KPSS test
    try:
        # regression='c' tests stationarity around a constant mean.
        # This is appropriate for weekly changes; for levels it is still a useful warning check.
        kpss_res = kpss(x, regression="c", nlags="auto")
        row["KPSS_stat"] = kpss_res[0]
        row["KPSS_pvalue"] = kpss_res[1]
    except Exception as e:
        row["KPSS_error"] = str(e)[:120]

    adf_stationary = row["ADF_pvalue"] < 0.05 if pd.notna(row["ADF_pvalue"]) else False
    kpss_stationary = row["KPSS_pvalue"] > 0.05 if pd.notna(row["KPSS_pvalue"]) else False

    if adf_stationary and kpss_stationary:
        row["stationarity_reading"] = "Stationary by both ADF and KPSS"
    elif adf_stationary and not kpss_stationary:
        row["stationarity_reading"] = "Mixed: ADF rejects unit root, KPSS rejects stationarity"
    elif (not adf_stationary) and kpss_stationary:
        row["stationarity_reading"] = "Mixed: ADF does not reject unit root, KPSS does not reject stationarity"
    else:
        row["stationarity_reading"] = "Likely non-stationary / weak evidence of stationarity"

    return row


stationarity_rows = []

for name, cfg in series_config.items():
    level_col = cfg["level_col"]
    diff_col = cfg["diff_col"]

    level_series = model_data[level_col]
    diff_series = model_data[diff_col]

    stationarity_rows.append(run_adf_kpss(level_series, name, "OAS level (bps)"))
    stationarity_rows.append(run_adf_kpss(diff_series, name, "Weekly change (bps)"))

stationarity_tests_table = pd.DataFrame(stationarity_rows)

print("ADF and KPSS stationarity tests: IG / AAA / AA / A")
display_or_print(stationarity_tests_table.round(4))

# A compact version focused on the modelling target
stationarity_target_table = stationarity_tests_table[stationarity_tests_table["variable"] == "Weekly change (bps)"].copy()
print("\nStationarity tests for the modelling target: weekly OAS changes")
display_or_print(stationarity_target_table[[
    "series", "n_obs", "ADF_stat", "ADF_pvalue", "KPSS_stat", "KPSS_pvalue", "stationarity_reading"
]].round(4))


# COMMAND ----------

# ==========================================================
# 6. TRAIN / TEST SPLIT
# ==========================================================

series_dict = {}
for name, cfg in series_config.items():
    tmp = model_data[["week_end_date", cfg["diff_col"]]].dropna().copy()
    tmp["week_end_date"] = pd.to_datetime(tmp["week_end_date"])
    series_dict[name] = tmp.set_index("week_end_date")[cfg["diff_col"]].astype(float)

common_index = None
for y in series_dict.values():
    common_index = y.index if common_index is None else common_index.intersection(y.index)
series_dict = {name: y.loc[common_index].copy() for name, y in series_dict.items()}

split_data = {}
for name, y in series_dict.items():
    split_idx = int(len(y) * TRAIN_RATIO)
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    split_data[name] = {"full": y, "train": y_train, "test": y_test}
    print(f"{name}")
    print(f"Full sample:  {len(y)} observations")
    print(f"Train sample: {len(y_train)} observations")
    print(f"Test sample:  {len(y_test)} observations")
    print(f"Train period: {y_train.index.min().date()} to {y_train.index.max().date()}")
    print(f"Test period:  {y_test.index.min().date()} to {y_test.index.max().date()}")
    print("-" * 80)

for name, parts in split_data.items():
    plt.figure(figsize=(14, 3.5))
    plt.plot(parts["train"].index, parts["train"].values, label="Train")
    plt.plot(parts["test"].index, parts["test"].values, label="Test")
    plt.axvline(parts["test"].index[0], linestyle="--")
    plt.title(f"{name}: train/test split")
    plt.xlabel("Date")
    plt.ylabel("Weekly OAS change (bps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# ==========================================================
# 7. MODEL AND DIAGNOSTIC HELPERS
# ==========================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_forecasts(df, actual_col, pred_cols):
    rows = []
    for col in pred_cols:
        rows.append({"model": col, "RMSE": rmse(df[actual_col], df[col]), "MAE": mean_absolute_error(df[actual_col], df[col])})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

def select_best_arma(y, max_p=3, max_q=3):
    best_aic, best_order, best_model = np.inf, None, None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(y, order=(p, 0, q)).fit()
                if model.aic < best_aic:
                    best_aic, best_order, best_model = model.aic, (p, q), model
            except Exception:
                continue
    if best_model is None:
        raise RuntimeError("No ARMA model could be estimated. Check the data or reduce max_p/max_q.")
    return best_order, best_model

def safe_jarque_bera(x):
    x = pd.Series(x).dropna()
    try:
        res = jarque_bera(x)
        stat = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        pval = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
        return stat, pval
    except Exception:
        return np.nan, np.nan

def lb_pvalue(x, lags=12):
    x = pd.Series(x).dropna()
    try:
        return float(acorr_ljungbox(x, lags=[lags], return_df=True)["lb_pvalue"].iloc[-1])
    except Exception:
        return np.nan

def arch_lm_pvalue(x, nlags=8):
    x = pd.Series(x).dropna()
    try:
        return float(het_arch(x, nlags=nlags)[1])
    except Exception:
        return np.nan

def summarize_model_residuals(series_name, model_name, resid, aic=np.nan, bic=np.nan, params_note=""):
    resid = pd.Series(resid).dropna()
    _, jb_p = safe_jarque_bera(resid)
    return {
        "series": series_name,
        "model": model_name,
        "AIC": aic,
        "BIC": bic,
        "resid_mean": resid.mean(),
        "resid_std": resid.std(),
        "JB_pvalue": jb_p,
        "LB_resid_pvalue_lag12": lb_pvalue(resid, lags=12),
        "LB_sq_resid_pvalue_lag12": lb_pvalue(resid**2, lags=12),
        "ARCH_LM_pvalue": arch_lm_pvalue(resid, nlags=8),
        "note": params_note
    }

def residual_diagnostic_plots(resid, title, acf_lags=20):
    resid = pd.Series(resid).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(resid, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"{title}: histogram")
    axes[0].grid(True, alpha=0.3)
    qqplot(resid, line="s", ax=axes[1])
    axes[1].set_title(f"{title}: Q-Q plot")
    plot_acf(resid, lags=acf_lags, ax=axes[2])
    axes[2].set_title(f"{title}: ACF")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# ==========================================================
# 8. IN-SAMPLE CALIBRATION OF BASE MODELS
# ==========================================================

calibration_results = {}
diagnostics_rows = []

for name, parts in split_data.items():
    y_train = parts["train"]
    print("=" * 100)
    print(f"{name}: in-sample calibration")
    print("=" * 100)
    calibration_results[name] = {}

    # Random Walk / no-change benchmark for changes
    rw_resid = y_train.copy()
    calibration_results[name]["Random Walk"] = {"resid": rw_resid}
    diagnostics_rows.append(summarize_model_residuals(name, "Random Walk", rw_resid, params_note="No-change benchmark on weekly changes"))

    # AR(1)
    try:
        ar1 = ARIMA(y_train, order=(1, 0, 0)).fit()
        calibration_results[name]["AR(1)"] = ar1
        diagnostics_rows.append(summarize_model_residuals(name, "AR(1)", ar1.resid, aic=ar1.aic, bic=ar1.bic))
    except Exception as e:
        print(f"AR(1) failed for {name}: {e}")

    # ARMA(p,q), selected by AIC
    best_order, arma_model = select_best_arma(y_train, max_p=MAX_ARMA_P, max_q=MAX_ARMA_Q)
    calibration_results[name]["ARMA"] = {"order": best_order, "model": arma_model}
    print(f"Selected ARMA order for {name}: {best_order}")
    diagnostics_rows.append(summarize_model_residuals(name, f"ARMA{best_order}", arma_model.resid, aic=arma_model.aic, bic=arma_model.bic, params_note="Selected by AIC"))

    arma_resid = arma_model.resid.dropna()

    # ARMA-GARCH(1,1)
    try:
        garch = arch_model(arma_resid, mean="Zero", vol="GARCH", p=1, q=1, dist=GARCH_DIST).fit(disp="off")
        calibration_results[name]["ARMA-GARCH"] = {"arma_order": best_order, "arma_model": arma_model, "garch_model": garch}
        diagnostics_rows.append(summarize_model_residuals(name, f"ARMA{best_order}-GARCH(1,1)", pd.Series(garch.std_resid).dropna(), aic=garch.aic, bic=garch.bic, params_note="Standardized residuals"))
    except Exception as e:
        print(f"GARCH failed for {name}: {e}")

    # ARMA-EGARCH(1,1)
    try:
        egarch = arch_model(arma_resid, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist=GARCH_DIST).fit(disp="off")
        calibration_results[name]["ARMA-EGARCH"] = {"arma_order": best_order, "arma_model": arma_model, "egarch_model": egarch}
        diagnostics_rows.append(summarize_model_residuals(name, f"ARMA{best_order}-EGARCH(1,1)", pd.Series(egarch.std_resid).dropna(), aic=egarch.aic, bic=egarch.bic, params_note="Standardized residuals"))
    except Exception as e:
        print(f"EGARCH failed for {name}: {e}")

    # ARMA-GJR-GARCH(1,1)
    try:
        gjr = arch_model(arma_resid, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=GARCH_DIST).fit(disp="off")
        calibration_results[name]["ARMA-GJR-GARCH"] = {"arma_order": best_order, "arma_model": arma_model, "gjr_model": gjr}
        diagnostics_rows.append(summarize_model_residuals(name, f"ARMA{best_order}-GJR-GARCH(1,1)", pd.Series(gjr.std_resid).dropna(), aic=gjr.aic, bic=gjr.bic, params_note="Standardized residuals"))
    except Exception as e:
        print(f"GJR-GARCH failed for {name}: {e}")

    # Markov-Switching optional extension
    if RUN_MARKOV_SWITCHING:
        try:
            ms = MarkovRegression(y_train, k_regimes=2, trend="c", switching_variance=True).fit(disp=False)
            calibration_results[name]["Markov-Switching"] = ms
            ms_resid = y_train - pd.Series(ms.fittedvalues, index=y_train.index)
            diagnostics_rows.append(summarize_model_residuals(name, "Markov-Switching(2 regimes)", ms_resid, aic=ms.aic, bic=ms.bic, params_note="Residual = actual - fitted value"))
        except Exception as e:
            print(f"Markov-Switching failed for {name}: {e}")

model_diagnostics_table = pd.DataFrame(diagnostics_rows)
print("Model diagnostics summary:")
display_or_print(model_diagnostics_table.round(4))

# COMMAND ----------

# ==========================================================
# 9. COMPACT MODEL-SELECTION VIEW
# ==========================================================

selection_view = model_diagnostics_table[[
    "series", "model", "AIC", "BIC", "JB_pvalue", "LB_resid_pvalue_lag12", "LB_sq_resid_pvalue_lag12", "ARCH_LM_pvalue", "note"
]].copy()
selection_view["AIC_sort"] = selection_view["AIC"].fillna(np.inf)
selection_view = selection_view.sort_values(["series", "AIC_sort"]).drop(columns="AIC_sort")
display_or_print(selection_view.round(4), "Compact model selection view:")

arma_order_summary = pd.DataFrame([
    {"series": name, "selected_ARMA_order": calibration_results[name]["ARMA"]["order"]}
    for name in split_data.keys()
])
display_or_print(arma_order_summary, "Selected ARMA orders:")

# COMMAND ----------

# ==========================================================
# 10. RESIDUAL DIAGNOSTIC PLOTS FOR SELECTED ARMA AND GJR-GARCH
# ==========================================================

for name in split_data.keys():
    order = calibration_results[name]["ARMA"]["order"]
    arma_model = calibration_results[name]["ARMA"]["model"]
    residual_diagnostic_plots(arma_model.resid, title=f"{name} ARMA{order} residuals")

    if "ARMA-GJR-GARCH" in calibration_results[name]:
        gjr_resid = pd.Series(calibration_results[name]["ARMA-GJR-GARCH"]["gjr_model"].std_resid).dropna()
        residual_diagnostic_plots(gjr_resid, title=f"{name} ARMA{order}-GJR standardized residuals")

# COMMAND ----------

# ==========================================================
# 11. STATIC OUT-OF-SAMPLE FORECASTS
# ==========================================================

static_forecasts = {}
static_eval_rows = []

for name, parts in split_data.items():
    y_train = parts["train"]
    y_test = parts["test"]
    h = len(y_test)

    order = calibration_results[name]["ARMA"]["order"]
    p, q = order

    rw_pred = pd.Series(0.0, index=y_test.index)
    arma_model = ARIMA(y_train, order=(p, 0, q)).fit()
    arma_pred = pd.Series(arma_model.forecast(steps=h).values, index=y_test.index)

    arma_resid = arma_model.resid.dropna()
    gjr_model = arch_model(arma_resid, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=GARCH_DIST).fit(disp="off")
    try:
        gjr_var = gjr_model.forecast(horizon=h, reindex=False).variance.iloc[-1].values
        gjr_sigma = np.sqrt(gjr_var)
    except Exception as e:
        print(f"Static GJR sigma forecast failed for {name}: {e}")
        gjr_sigma = np.repeat(np.nan, h)

    fcst = pd.DataFrame({
        "actual": y_test,
        "RW_forecast": rw_pred,
        "ARMA_forecast": arma_pred,
        "ARMA_GJR_forecast": arma_pred.copy(),
        "ARMA_GJR_sigma": gjr_sigma
    }, index=y_test.index)

    static_forecasts[name] = fcst
    ev = evaluate_forecasts(fcst, "actual", ["RW_forecast", "ARMA_forecast", "ARMA_GJR_forecast"])
    ev["series"] = name
    ev["method"] = "Static OOS"
    ev["ARMA_order"] = str(order)
    static_eval_rows.append(ev)

static_eval_table = pd.concat(static_eval_rows, ignore_index=True)
static_eval_table = static_eval_table[["method", "series", "model", "ARMA_order", "RMSE", "MAE"]]
static_eval_table["model"] = static_eval_table["model"].replace({
    "RW_forecast": "Random Walk",
    "ARMA_forecast": "Selected ARMA",
    "ARMA_GJR_forecast": "Selected ARMA-GJR-GARCH(1,1)"
})
display_or_print(static_eval_table.round(4), "Static OOS forecast evaluation:")

# COMMAND ----------

# ==========================================================
# 12. STATIC FORECAST PLOTS
# ==========================================================

for name, fcst in static_forecasts.items():
    order = calibration_results[name]["ARMA"]["order"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(fcst.index, fcst["actual"], label="Actual")
    axes[0].plot(fcst.index, fcst["RW_forecast"], label="Random Walk")
    axes[0].plot(fcst.index, fcst["ARMA_forecast"], label=f"ARMA{order}")
    axes[0].plot(fcst.index, fcst["ARMA_GJR_forecast"], label=f"ARMA{order}-GJR-GARCH", linestyle="--")
    axes[0].set_title(f"{name}: static out-of-sample forecasts")
    axes[0].set_ylabel("Weekly OAS change (bps)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(fcst.index, fcst["ARMA_GJR_sigma"])
    axes[1].set_title(f"{name}: static GJR-GARCH conditional volatility forecast")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Forecast sigma")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# ==========================================================
# 13. FAST ROLLING ONE-STEP-AHEAD FORECASTS - FIXED VERSION
# ==========================================================
# ARMA parameters are estimated once on the training sample.
# Then the ARMA state is updated recursively with realised OOS observations
# using a clean RangeIndex to avoid Databricks/statsmodels date-index issues.
# GJR-GARCH parameters are estimated once on training ARMA residuals.
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

rolling_forecasts = {}
rolling_eval_rows = []

def get_gjr_params(gjr_res):
    params = gjr_res.params
    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", 0.0))
    gamma = float(params.get("gamma[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    return omega, alpha, gamma, beta


def gjr_one_step_variance(last_eps, last_h, omega, alpha, gamma, beta):
    if not np.isfinite(last_eps) or not np.isfinite(last_h):
        return np.nan
    indicator = 1.0 if last_eps < 0 else 0.0
    h_next = omega + alpha * last_eps**2 + gamma * indicator * last_eps**2 + beta * last_h
    if not np.isfinite(h_next) or h_next <= 0:
        return np.nan
    return h_next


def fit_arma_with_range_index(y, order):
    y_range = pd.Series(y.astype(float).values, index=pd.RangeIndex(start=0, stop=len(y), step=1))
    p, q = order
    model = ARIMA(y_range, order=(p, 0, q), enforce_stationarity=False, enforce_invertibility=False)
    return model.fit()


if RUN_ROLLING_FORECASTS:
    for name, parts in split_data.items():
        print("=" * 100)
        print(f"{name}: fast rolling one-step-ahead forecasts - fixed version")
        print("=" * 100)

        y_train = parts["train"].dropna().astype(float).copy()
        y_test = parts["test"].dropna().astype(float).copy()
        rolling_dates = y_test.index

        order = calibration_results[name]["ARMA"]["order"]
        p, q = order

        print(f"Selected ARMA order for {name}: {order}")
        print(f"Train observations: {len(y_train)}")
        print(f"Test observations:  {len(y_test)}")

        actuals, rw_preds, arma_preds, gjr_preds, gjr_sigmas = [], [], [], [], []

        try:
            arma_current = fit_arma_with_range_index(y_train, order)
            arma_available = True
            print("Initial ARMA fit completed.")
        except Exception as e:
            print(f"Initial ARMA fit failed for {name}: {e}")
            arma_current = None
            arma_available = False

        if arma_available:
            try:
                train_resid = pd.Series(arma_current.resid).dropna()
                gjr_res = arch_model(train_resid, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist=GARCH_DIST).fit(disp="off")
                omega, alpha, gamma, beta = get_gjr_params(gjr_res)
                last_eps = float(train_resid.iloc[-1])
                last_h = float(gjr_res.conditional_volatility.iloc[-1] ** 2)
                gjr_available = True
                print("Initial GJR-GARCH fit completed.")
                print(f"omega={omega:.6f}, alpha={alpha:.6f}, gamma={gamma:.6f}, beta={beta:.6f}")
            except Exception as e:
                print(f"Initial GJR-GARCH fit failed for {name}: {e}")
                gjr_available = False
                omega, alpha, gamma, beta = np.nan, np.nan, np.nan, np.nan
                last_eps, last_h = np.nan, np.nan
        else:
            gjr_available = False
            omega, alpha, gamma, beta = np.nan, np.nan, np.nan, np.nan
            last_eps, last_h = np.nan, np.nan

        for i, current_date in enumerate(rolling_dates):
            y_true = float(y_test.loc[current_date])
            rw_pred = 0.0

            if arma_available and arma_current is not None:
                try:
                    arma_pred = float(arma_current.forecast(steps=1).iloc[0])
                except Exception as e:
                    print(f"{name}: ARMA forecast failed at step {i}: {e}")
                    arma_pred = np.nan
            else:
                arma_pred = np.nan

            if gjr_available:
                h_1 = gjr_one_step_variance(last_eps, last_h, omega, alpha, gamma, beta)
                sigma_1 = np.sqrt(h_1) if np.isfinite(h_1) and h_1 > 0 else np.nan
            else:
                h_1 = np.nan
                sigma_1 = np.nan

            actuals.append(y_true)
            rw_preds.append(rw_pred)
            arma_preds.append(arma_pred)
            gjr_preds.append(arma_pred)
            gjr_sigmas.append(sigma_1)

            if arma_available and arma_current is not None and np.isfinite(y_true):
                try:
                    next_idx = len(y_train) + i
                    new_obs = pd.Series([y_true], index=pd.RangeIndex(start=next_idx, stop=next_idx + 1, step=1))
                    arma_current = arma_current.append(new_obs, refit=False)
                except Exception as e:
                    print(f"{name}: ARMA append failed at step {i}: {e}")

            if gjr_available and np.isfinite(arma_pred):
                realized_eps = y_true - arma_pred
                if np.isfinite(realized_eps) and np.isfinite(h_1):
                    last_eps = realized_eps
                    last_h = h_1

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(rolling_dates)} rolling forecasts for {name}")

        fcst = pd.DataFrame({"actual": actuals, "RW_forecast": rw_preds, "ARMA_forecast": arma_preds, "ARMA_GJR_forecast": gjr_preds, "ARMA_GJR_sigma": gjr_sigmas}, index=rolling_dates)
        rolling_forecasts[name] = fcst

        print("Forecast sanity check:")
        print(fcst[["actual", "ARMA_forecast", "ARMA_GJR_sigma"]].describe().round(4))
        arma_std = fcst["ARMA_forecast"].std()
        if arma_std < 1e-6:
            print("WARNING: ARMA forecast is almost flat. Check append/update behaviour.")
        else:
            print(f"ARMA forecast variation looks OK. Std = {arma_std:.4f}")

        fcst_eval = fcst.dropna(subset=["actual", "RW_forecast", "ARMA_forecast", "ARMA_GJR_forecast"]).copy()
        ev = evaluate_forecasts(fcst_eval, "actual", ["RW_forecast", "ARMA_forecast", "ARMA_GJR_forecast"])
        ev["series"] = name
        ev["method"] = "Rolling 1-step"
        ev["ARMA_order"] = str(order)
        rolling_eval_rows.append(ev)

    rolling_eval_table = pd.concat(rolling_eval_rows, ignore_index=True)
    rolling_eval_table = rolling_eval_table[["method", "series", "model", "ARMA_order", "RMSE", "MAE"]]
    rolling_eval_table["model"] = rolling_eval_table["model"].replace({"RW_forecast": "Random Walk", "ARMA_forecast": "Selected ARMA", "ARMA_GJR_forecast": "Selected ARMA-GJR-GARCH(1,1)"})
    display_or_print(rolling_eval_table.round(4), "Fast rolling one-step forecast evaluation:")
else:
    rolling_eval_table = pd.DataFrame()
    print("Rolling forecasts skipped. Set RUN_ROLLING_FORECASTS = True to run this section.")


# COMMAND ----------

# ==========================================================
# 14. ROLLING FORECAST PLOTS
# ==========================================================

if RUN_ROLLING_FORECASTS:
    for name, fcst in rolling_forecasts.items():
        order = calibration_results[name]["ARMA"]["order"]
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(fcst.index, fcst["actual"], label="Actual")
        axes[0].plot(fcst.index, fcst["RW_forecast"], label="Random Walk")
        axes[0].plot(fcst.index, fcst["ARMA_forecast"], label=f"ARMA{order}")
        axes[0].plot(fcst.index, fcst["ARMA_GJR_forecast"], label=f"ARMA{order}-GJR-GARCH", linestyle="--")
        axes[0].set_title(f"{name}: rolling one-step-ahead forecasts")
        axes[0].set_ylabel("Weekly OAS change (bps)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(fcst.index, fcst["ARMA_GJR_sigma"])
        axes[1].set_title(f"{name}: rolling GJR-GARCH conditional volatility forecast")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Forecast sigma")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# ==========================================================
# 15. DIEBOLD-MARIANO TESTS AGAINST RANDOM WALK
# ==========================================================
# Positive mean_loss_diff means the alternative model has lower loss than Random Walk.

def diebold_mariano_test(y_true, pred_1, pred_2, power=2, h=1):
    y_true, pred_1, pred_2 = np.asarray(y_true), np.asarray(pred_1), np.asarray(pred_2)
    e1 = y_true - pred_1
    e2 = y_true - pred_2
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d = pd.Series(d).dropna().values
    T = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    if h > 1:
        gamma = [np.cov(d[lag:], d[:-lag], ddof=1)[0, 1] for lag in range(1, h)]
        var_d = gamma0 + 2 * np.sum(gamma)
    else:
        var_d = gamma0
    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return {"DM_stat": dm_stat, "p_value": p_value, "mean_loss_diff": d_bar}

dm_rows = []
for method_name, fcst_dict in [("Static OOS", static_forecasts), ("Rolling 1-step", rolling_forecasts if RUN_ROLLING_FORECASTS else {})]:
    for name, fcst in fcst_dict.items():
        for alternative_col, alternative_name in [("ARMA_forecast", "Selected ARMA"), ("ARMA_GJR_forecast", "Selected ARMA-GJR-GARCH(1,1)")]:
            res = diebold_mariano_test(fcst["actual"], fcst["RW_forecast"], fcst[alternative_col], power=2, h=1)
            dm_rows.append({"method": method_name, "series": name, "comparison": f"Random Walk vs {alternative_name}", **res})

dm_results_table = pd.DataFrame(dm_rows)
display_or_print(dm_results_table.round(4), "Diebold-Mariano tests:")

# COMMAND ----------

# ==========================================================
# 16. FINAL COMPARISON TABLES
# Including Directional Accuracy
# ==========================================================

import numpy as np
import pandas as pd


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))


def model_name_to_forecast_col(model_name):
    model_name = str(model_name).lower()
    if "random walk" in model_name:
        return "RW_forecast"
    if "gjr" in model_name or "garch" in model_name:
        return "ARMA_GJR_forecast"
    if "arma" in model_name:
        return "ARMA_forecast"
    return None


def add_directional_accuracy(eval_table, forecast_dict, method_name):
    out = eval_table.copy()
    directional_values = []
    for _, row in out.iterrows():
        series_name = row["series"]
        model_name = row["model"]
        forecast_col = model_name_to_forecast_col(model_name)
        if series_name not in forecast_dict or forecast_col is None:
            directional_values.append(np.nan)
            continue
        fcst = forecast_dict[series_name].copy()
        if "actual" not in fcst.columns or forecast_col not in fcst.columns:
            directional_values.append(np.nan)
            continue
        directional_values.append(directional_accuracy(fcst["actual"], fcst[forecast_col]))
    out["Directional_Accuracy"] = directional_values
    out["method"] = method_name
    return out


static_eval_table_da = add_directional_accuracy(static_eval_table, static_forecasts, "Static OOS")
final_eval_tables = [static_eval_table_da]

if RUN_ROLLING_FORECASTS and not rolling_eval_table.empty:
    rolling_eval_table_da = add_directional_accuracy(rolling_eval_table, rolling_forecasts, "Rolling 1-step")
    final_eval_tables.append(rolling_eval_table_da)

final_forecast_eval = pd.concat(final_eval_tables, ignore_index=True)
preferred_cols = ["method", "series", "model", "ARMA_order", "RMSE", "MAE", "Directional_Accuracy"]
existing_preferred_cols = [c for c in preferred_cols if c in final_forecast_eval.columns]
remaining_cols = [c for c in final_forecast_eval.columns if c not in existing_preferred_cols]
final_forecast_eval = final_forecast_eval[existing_preferred_cols + remaining_cols]

print("Final forecast evaluation table:")
display_or_print(final_forecast_eval.round(4))

rmse_pivot = final_forecast_eval.pivot_table(index=["method", "model"], columns="series", values="RMSE")
mae_pivot = final_forecast_eval.pivot_table(index=["method", "model"], columns="series", values="MAE")
directional_accuracy_pivot = final_forecast_eval.pivot_table(index=["method", "model"], columns="series", values="Directional_Accuracy")

print("RMSE comparison across IG / AAA / AA / A:")
display_or_print(rmse_pivot.round(4))
print("MAE comparison across IG / AAA / AA / A:")
display_or_print(mae_pivot.round(4))
print("Directional accuracy comparison across IG / AAA / AA / A:")
display_or_print(directional_accuracy_pivot.round(4))




# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretation guide
# MAGIC
# MAGIC Use this notebook to compare the base-model results inside the investment-grade segment: IG aggregate, AAA, AA and A.
# MAGIC
# MAGIC A useful way to read the output is:
# MAGIC
# MAGIC 1. **Descriptive statistics**: check whether lower investment-grade ratings have higher spread levels and higher volatility.
# MAGIC 2. **Stationarity tests**: use ADF and KPSS to verify that weekly spread changes are suitable for ARMA/GARCH-style modelling.
# MAGIC 3. **ARMA order selection**: check whether A or AA require richer dynamics than AAA.
# MAGIC 4. **Residual diagnostics**: check whether ARMA removes autocorrelation and whether GARCH-type models remove volatility clustering.
# MAGIC 5. **Forecast evaluation**: compare whether ARMA-based models improve over the Random Walk benchmark.
# MAGIC 6. **Diebold-Mariano tests**: check whether any RMSE/MAE improvement is statistically meaningful.
# MAGIC
# MAGIC For the written TFG, the question is whether predictive dynamics differ not only between IG and HY, but also within the investment-grade segment itself.
# MAGIC
