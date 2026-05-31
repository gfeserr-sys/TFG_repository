# Databricks notebook source
# MAGIC %md
# MAGIC CREATE SCHEMAS AND CATALOGS TO STORE DATA

# COMMAND ----------

# DBTITLE 1,Create tfg_data catalog
# MAGIC %sql
# MAGIC -- Drop the schema first
# MAGIC DROP SCHEMA IF EXISTS tfg_data;
# MAGIC
# MAGIC -- Create tfg_data as a catalog
# MAGIC CREATE CATALOG IF NOT EXISTS tfg_data;

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS tfg_data.original_data;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS tfg_data.bronze_data;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS tfg_data.silver_data;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS tfg_data.gold_data;
# MAGIC

# COMMAND ----------

# MAGIC %pip install pyspark databricks-connect delta-spark

# COMMAND ----------

# MAGIC %md
# MAGIC DATA IMPORT AND TRANSFORMATION

# COMMAND ----------

# DBTITLE 1,Cell 7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("default")

from pyspark.sql import functions as F
from pyspark.sql import Window

ig_s = (
    spark.sql("SELECT * FROM tfg_data.original_data.ig_aggregate_oas")
    .withColumnRenamed("observation_date", "date")
    .withColumnRenamed("BAMLC0A0CM", "value")
)

hy_s = (
    spark.sql("SELECT * FROM tfg_data.original_data.hy_aggregate_oas")
    .withColumnRenamed("observation_date", "date")
    .withColumnRenamed("BAMLH0A0HYM2", "value")
)

ig=ig_s.toPandas()
hy=hy_s.toPandas()

# COMMAND ----------

def build_weekly_series(df, prefix, n_lags=4, week_rule="W-FRI"):
    """
    Build interpolated daily and weekly time series from a raw OAS dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with columns ['date', 'value'].
    prefix : str
        Prefix used to identify the series (e.g. 'IG' or 'HY').
    n_lags : int
        Number of lags to create.
    week_rule : str
        Weekly resampling rule. 'W-FRI' means weeks ending on Friday.

    Returns
    -------
    daily : pandas.DataFrame
        Daily interpolated series.
    weekly : pandas.DataFrame
        Weekly averaged series with transformations and lags.
    """

    data = df.copy()

    # Create a complete daily calendar
    data = data.set_index("date").asfreq("D")

    # Interpolate missing values using time information
    # This uses information from both previous and next observations
    data["value"] = (
        data["value"]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )

    # Store daily interpolated version
    daily = data.copy()

    # Convert daily series into weekly series using weekly averages
    weekly = data.resample(week_rule).mean(numeric_only=True)

    # Convert from percentage points to basis points
    weekly["value_bp"] = weekly["value"] * 100

    # First difference in basis points
    weekly["diff_1"] = weekly["value_bp"].diff()

    # Create lags for level and first difference
    for lag in range(1, n_lags + 1):
        weekly[f"value_bp_lag{lag}"] = weekly["value_bp"].shift(lag)
        weekly[f"diff_1_lag{lag}"] = weekly["diff_1"].shift(lag)

    # Format and rename daily output
    daily = daily.reset_index().rename(columns={
        "value": f"{prefix}_value_interp_pct"
    })

    # Format and rename weekly output
    weekly = weekly.reset_index().rename(columns={
        "date": "week_end_date",
        "value": f"{prefix}_value_interp_pct",
        "value_bp": f"{prefix}_value_bp",
        "diff_1": f"{prefix}_diff_1"
    })

    for lag in range(1, n_lags + 1):
        weekly = weekly.rename(columns={
            f"value_bp_lag{lag}": f"{prefix}_value_bp_lag{lag}",
            f"diff_1_lag{lag}": f"{prefix}_diff_1_lag{lag}"
        })

    return daily, weekly

# COMMAND ----------

ig_daily, ig_weekly = build_weekly_series(ig, prefix="IG", n_lags=4, week_rule="W-FRI")

ig_weekly.head()

# COMMAND ----------

hy_daily, hy_weekly = build_weekly_series(hy, prefix="HY", n_lags=4, week_rule="W-FRI")

hy_weekly.head()

# COMMAND ----------

def add_subsamples(df, date_col="week_end_date"):
    """
    Add economically meaningful subsamples to the weekly dataset.

    Subsamples:
    - pre_GFC
    - GFC
    - post_GFC_pre_COVID
    - COVID
    - post_COVID
    """

    out = df.copy()

    out["subsample"] = np.select(
        [
            out[date_col] < pd.Timestamp("2007-07-01"),
            (out[date_col] >= pd.Timestamp("2007-07-01")) & (out[date_col] <= pd.Timestamp("2009-06-30")),
            (out[date_col] >= pd.Timestamp("2009-07-01")) & (out[date_col] <= pd.Timestamp("2020-02-15")),
            (out[date_col] >= pd.Timestamp("2020-02-16")) & (out[date_col] <= pd.Timestamp("2020-12-31"))
        ],
        [
            "pre_GFC",
            "GFC",
            "post_GFC_pre_COVID",
            "COVID"
        ],
        default="post_COVID"
    )

    # Dummy variables for each subsample
    out["pre_GFC"] = (out["subsample"] == "pre_GFC").astype(int)
    out["GFC"] = (out["subsample"] == "GFC").astype(int)
    out["post_GFC_pre_COVID"] = (out["subsample"] == "post_GFC_pre_COVID").astype(int)
    out["COVID"] = (out["subsample"] == "COVID").astype(int)
    out["post_COVID"] = (out["subsample"] == "post_COVID").astype(int)

    return out


# Add subsample structure to both weekly datasets
ig_weekly = add_subsamples(ig_weekly)
hy_weekly = add_subsamples(hy_weekly)

print("IG weekly with subsamples:")
display(ig_weekly.head())

print("\nHY weekly with subsamples:")
display(hy_weekly.head())

# COMMAND ----------

# Merge IG and HY weekly datasets only for comparative analysis
# and for constructing the HY-IG spread gap

credit_weekly = pd.merge(
    ig_weekly,
    hy_weekly.drop(columns=["subsample", "pre_GFC", "GFC", "post_GFC_pre_COVID", "COVID", "post_COVID"]),
    on="week_end_date",
    how="inner"
)

# Create the HY minus IG gap in basis points
credit_weekly["HY_IG_gap_bp"] = credit_weekly["HY_value_bp"] - credit_weekly["IG_value_bp"]

# Create lags of the gap
for lag in range(1, 5):
    credit_weekly[f"HY_IG_gap_bp_lag{lag}"] = credit_weekly["HY_IG_gap_bp"].shift(lag)

print("Merged weekly shape:", credit_weekly.shape)

print("\nMerged weekly preview:")
print(credit_weekly.head())

# COMMAND ----------

# MAGIC %md
# MAGIC EDA 

# COMMAND ----------

# Keep only the columns needed for EDA
ig_eda = ig_weekly[["week_end_date", "IG_value_bp","IG_diff_1"]].copy()
hy_eda = hy_weekly[["week_end_date", "HY_value_bp","HY_diff_1"]].copy()
gap_eda = credit_weekly[["week_end_date", "HY_IG_gap_bp"]].copy()

# Make sure dates are in datetime format
ig_eda["week_end_date"] = pd.to_datetime(ig_eda["week_end_date"])
hy_eda["week_end_date"] = pd.to_datetime(hy_eda["week_end_date"])
gap_eda["week_end_date"] = pd.to_datetime(gap_eda["week_end_date"])


# COMMAND ----------

display(ig_eda)

# COMMAND ----------

# DBTITLE 1,Install statsmodels
# MAGIC %pip install statsmodels

# COMMAND ----------

# Plot IG and HY separately

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(ig_eda["week_end_date"], ig_eda["IG_value_bp"])
axes[0].set_title("Investment Grade OAS (weekly, basis points)")
axes[0].set_ylabel("Basis points")
axes[0].grid(True, alpha=0.3)

axes[1].plot(hy_eda["week_end_date"], hy_eda["HY_value_bp"])
axes[1].set_title("High Yield OAS (weekly, basis points)")
axes[1].set_ylabel("Basis points")
axes[1].set_xlabel("Date")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Visual comparison in levels: IG, HY and the HY-IG gap

plt.figure(figsize=(14, 6))
plt.plot(credit_weekly["week_end_date"], credit_weekly["IG_value_bp"], label="IG OAS")
plt.plot(credit_weekly["week_end_date"], credit_weekly["HY_value_bp"], label="HY OAS")
plt.plot(credit_weekly["week_end_date"], credit_weekly["HY_IG_gap_bp"], label="HY - IG gap", linestyle="--")
plt.title("IG vs HY corporate spreads and HY-IG gap (weekly, basis points)")
plt.xlabel("Date")
plt.ylabel("Basis points")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Very useful extra: standardized comparison
# This allows comparison of dynamics without the scale problem

credit_eda_std = credit_weekly[["week_end_date", "IG_value_bp", "HY_value_bp"]].copy()

credit_eda_std["IG_z"] = (
    (credit_eda_std["IG_value_bp"] - credit_eda_std["IG_value_bp"].mean()) /
    credit_eda_std["IG_value_bp"].std()
)

credit_eda_std["HY_z"] = (
    (credit_eda_std["HY_value_bp"] - credit_eda_std["HY_value_bp"].mean()) /
    credit_eda_std["HY_value_bp"].std()
)

plt.figure(figsize=(14, 6))
plt.plot(credit_eda_std["week_end_date"], credit_eda_std["IG_z"], label="IG standardized")
plt.plot(credit_eda_std["week_end_date"], credit_eda_std["HY_z"], label="HY standardized")

plt.title("Standardized comparison of IG and HY spread dynamics")
plt.xlabel("Date")
plt.ylabel("Z-score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Plot IG and HY differences

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(ig_eda["week_end_date"], ig_eda["IG_diff_1"])
axes[0].set_title("Investment Grade Change OAS (weekly, basis points)")
axes[0].set_ylabel("Basis points")
axes[0].grid(True, alpha=0.3)

axes[1].plot(hy_eda["week_end_date"], hy_eda["HY_diff_1"])
axes[1].set_title("High Yield Change OAS (weekly, basis points)")
axes[1].set_ylabel("Basis points")
axes[1].set_xlabel("Date")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Descriptive statistics table

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

desc_table = pd.concat([
    descriptive_stats(credit_weekly["IG_value_bp"]).rename("IG"),
    descriptive_stats(credit_weekly["HY_value_bp"]).rename("HY"),
    descriptive_stats(credit_weekly["HY_IG_gap_bp"]).rename("HY_IG_gap")
], axis=1).T

print("Descriptive statistics (weekly OAS in basis points):")
display(desc_table.round(2))

# COMMAND ----------

# ============================================
# EDA extension before model estimation
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

ig_series = ig_weekly[["week_end_date", "IG_diff_1"]].copy()
hy_series = hy_weekly[["week_end_date", "HY_diff_1"]].copy()

ig_series["week_end_date"] = pd.to_datetime(ig_series["week_end_date"])
hy_series["week_end_date"] = pd.to_datetime(hy_series["week_end_date"])

ig_series = ig_series.dropna(subset=["IG_diff_1"]).reset_index(drop=True)
hy_series = hy_series.dropna(subset=["HY_diff_1"]).reset_index(drop=True)

series_dict = {
    "IG": ig_series.set_index("week_end_date")["IG_diff_1"],
    "HY": hy_series.set_index("week_end_date")["HY_diff_1"]
}

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

def run_eda_checks(series, name):
    x = series.dropna().copy()

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
    jb_stat, jb_pvalue = jarque_bera(x)

    print("\nJarque-Bera test for normality:")
    print(f"JB statistic = {jb_stat:.4f}")
    print(f"p-value      = {jb_pvalue:.6f}")

    if jb_pvalue < 0.05:
        print("Interpretation: reject normality at the 5% level.")
    else:
        print("Interpretation: cannot reject normality at the 5% level.")

    # -------------------------
    # Ljung-Box on levels of diff_1
    # -------------------------
    lb = acorr_ljungbox(x, lags=ljungbox_lags, return_df=True)

    print("\nLjung-Box test on spread changes:")
    print(lb.round(4))

    # -------------------------
    # Ljung-Box on squared diff_1
    # Useful for volatility clustering
    # -------------------------
    lb_sq = acorr_ljungbox(x**2, lags=ljungbox_lags, return_df=True)

    print("\nLjung-Box test on squared spread changes:")
    print(lb_sq.round(4))

    # -------------------------
    # ARCH-LM test
    # -------------------------
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

    # Time series of changes
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
    # 6) Extra visual check:
    #    squared changes over time
    # -------------------------

    plt.figure(figsize=(14, 4))
    plt.plot((x**2).index, (x**2).values)
    plt.title(f"{name}: squared weekly changes")
    plt.xlabel("Date")
    plt.ylabel("Squared change")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --------------------------------------------
# 4) Run all diagnostics for IG and HY
# --------------------------------------------

for name, s in series_dict.items():
    run_eda_checks(s, name)

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller, kpss

# ejemplo: sustituye 'series' por tu variable, por ejemplo df["IG_OAS"] o df["dIG_OAS"]
x = ig_weekly["IG_diff_1"].dropna()

# -------------------------
# ADF test
# H0: unit root -> no estacionaria
# H1: estacionaria
# -------------------------
adf_result = adfuller(x, autolag="AIC")

print("ADF test")
print(f"Test statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print(f"Lags used: {adf_result[2]}")
print(f"Number of observations: {adf_result[3]}")
print("Critical values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print("=> Reject H0: the series looks stationary")
else:
    print("=> Fail to reject H0: the series may be non-stationary")


# -------------------------
# KPSS test
# H0: estacionaria
# H1: no estacionaria
# regression='c' -> estacionaria alrededor de una constante
# regression='ct' -> estacionaria alrededor de tendencia
# -------------------------
kpss_result = kpss(x, regression="c", nlags="auto")

print("\nKPSS test")
print(f"Test statistic: {kpss_result[0]:.4f}")
print(f"p-value: {kpss_result[1]:.4f}")
print(f"Lags used: {kpss_result[2]}")
print("Critical values:")
for key, value in kpss_result[3].items():
    print(f"   {key}: {value:.4f}")

if kpss_result[1] < 0.05:
    print("=> Reject H0: the series may be non-stationary")
else:
    print("=> Fail to reject H0: the series looks stationary")

# COMMAND ----------

from statsmodels.tsa.stattools import adfuller, kpss

# ejemplo: sustituye 'series' por tu variable, por ejemplo df["IG_OAS"] o df["dIG_OAS"]
x = hy_weekly["HY_diff_1"].dropna()

# -------------------------
# ADF test
# H0: unit root -> no estacionaria
# H1: estacionaria
# -------------------------
adf_result = adfuller(x, autolag="AIC")

print("ADF test")
print(f"Test statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print(f"Lags used: {adf_result[2]}")
print(f"Number of observations: {adf_result[3]}")
print("Critical values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value:.4f}")

if adf_result[1] < 0.05:
    print("=> Reject H0: the series looks stationary")
else:
    print("=> Fail to reject H0: the series may be non-stationary")


# -------------------------
# KPSS test
# H0: estacionaria
# H1: no estacionaria
# regression='c' -> estacionaria alrededor de una constante
# regression='ct' -> estacionaria alrededor de tendencia
# -------------------------
kpss_result = kpss(x, regression="c", nlags="auto")

print("\nKPSS test")
print(f"Test statistic: {kpss_result[0]:.4f}")
print(f"p-value: {kpss_result[1]:.4f}")
print(f"Lags used: {kpss_result[2]}")
print("Critical values:")
for key, value in kpss_result[3].items():
    print(f"   {key}: {value:.4f}")

if kpss_result[1] < 0.05:
    print("=> Reject H0: the series may be non-stationary")
else:
    print("=> Fail to reject H0: the series looks stationary")

# COMMAND ----------

# MAGIC %md
# MAGIC MODEL CALIBRATION
# MAGIC

# COMMAND ----------

# ==========================================================
# TRAIN / TEST SPLIT FOR OUT-OF-SAMPLE FORECASTING
# We keep the last 20% of the sample as test
# and use the first 80% as in-sample / training data
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from arch import arch_model


# ----------------------------------------------------------
# 1) Build clean weekly target series
# ----------------------------------------------------------

ig_y = (
    ig_weekly[["week_end_date", "IG_diff_1"]]
    .dropna()
    .assign(week_end_date=lambda df: pd.to_datetime(df["week_end_date"]))
    .set_index("week_end_date")["IG_diff_1"]
)

hy_y = (
    hy_weekly[["week_end_date", "HY_diff_1"]]
    .dropna()
    .assign(week_end_date=lambda df: pd.to_datetime(df["week_end_date"]))
    .set_index("week_end_date")["HY_diff_1"]
)

series_dict = {
    "IG": ig_y,
    "HY": hy_y
}


# ----------------------------------------------------------
# 2) Time-series split: first 80% train, last 20% test
# ----------------------------------------------------------

train_ratio = 0.80
split_data = {}

for name, y in series_dict.items():
    split_idx = int(len(y) * train_ratio)

    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    split_data[name] = {
        "full": y,
        "train": y_train,
        "test": y_test
    }

    print(f"{name}")
    print(f"Full sample:  {len(y)} observations")
    print(f"Train sample: {len(y_train)} observations")
    print(f"Test sample:  {len(y_test)} observations")
    print(f"Train period: {y_train.index.min().date()} to {y_train.index.max().date()}")
    print(f"Test period:  {y_test.index.min().date()} to {y_test.index.max().date()}")
    print("-" * 80)


# ----------------------------------------------------------
# 3) Optional visual check of the split
# ----------------------------------------------------------

for name, parts in split_data.items():
    plt.figure(figsize=(14, 4))
    plt.plot(parts["train"].index, parts["train"].values, label="Train")
    plt.plot(parts["test"].index, parts["test"].values, label="Test")
    plt.axvline(parts["test"].index[0], linestyle="--")
    plt.title(f"{name} - In-sample / Out-of-sample split")
    plt.xlabel("Date")
    plt.ylabel("Weekly spread change (bp)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 4) Residual diagnostics helper
# ----------------------------------------------------------

def residual_diagnostics(resid, title, lb_lags=(4, 8, 12), acf_lags=20):
    resid = pd.Series(resid).dropna()

    print("=" * 95)
    print(title)
    print("=" * 95)

    print("\nResidual summary:")
    print(resid.describe().round(4))
    print("\nSkewness:", round(resid.skew(), 4))
    print("Kurtosis:", round(resid.kurt(), 4))

    jb_stat, jb_pvalue = jarque_bera(resid)
    print("\nJarque-Bera test:")
    print(f"JB statistic = {jb_stat:.4f}")
    print(f"p-value      = {jb_pvalue:.6f}")

    lb = acorr_ljungbox(resid, lags=list(lb_lags), return_df=True)
    print("\nLjung-Box test on residuals:")
    print(lb.round(4))

    lb_sq = acorr_ljungbox(resid**2, lags=list(lb_lags), return_df=True)
    print("\nLjung-Box test on squared residuals:")
    print(lb_sq.round(4))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(resid, bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"{title} - Histogram")
    axes[0].grid(True, alpha=0.3)

    qqplot(resid, line="s", ax=axes[1])
    axes[1].set_title(f"{title} - Q-Q plot")

    plot_acf(resid, lags=acf_lags, ax=axes[2])
    axes[2].set_title(f"{title} - ACF of residuals")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 5) Small ARMA order-selection helper
# ----------------------------------------------------------

def select_best_arma(y, max_p=3, max_q=3):
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(y, order=(p, 0, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
                    best_model = model
            except:
                continue

    return best_order, best_model

# COMMAND ----------

# ==========================================================
# MODEL 1: RANDOM WALK / NO-CHANGE BENCHMARK
# Estimated only on the in-sample period
# ==========================================================

rw_results_in_sample = {}

for name, parts in split_data.items():
    y_train = parts["train"]

    print(f"\n{name} - Random Walk / no-change benchmark (train only)")

    rw_fitted_train = pd.Series(0.0, index=y_train.index)
    rw_resid_train = y_train - rw_fitted_train

    rw_results_in_sample[name] = {
        "train_series": y_train,
        "fitted_train": rw_fitted_train,
        "resid_train": rw_resid_train
    }

    residual_diagnostics(
        rw_resid_train,
        title=f"{name} - Random Walk residuals (train sample)"
    )

# COMMAND ----------

# ==========================================================
# MODEL 2: AR(1)
# Estimated only on the training sample
# ==========================================================

ar1_results_in_sample = {}

for name, parts in split_data.items():
    y_train = parts["train"]

    print(f"\n{name} - AR(1) estimated on train sample")

    ar1 = ARIMA(y_train, order=(1, 0, 0)).fit()

    ar1_results_in_sample[name] = ar1

    print(ar1.summary())

    ar1_resid_train = ar1.resid

    residual_diagnostics(
        ar1_resid_train,
        title=f"{name} - AR(1) residuals (train sample)"
    )

# COMMAND ----------

# ==========================================================
# MODEL 3: ARMA(p,q)
# Order selected by AIC using only the training sample
# ==========================================================

arma_results_in_sample = {}
arma_orders_in_sample = {}

for name, parts in split_data.items():
    y_train = parts["train"]

    print(f"\n{name} - ARMA order selection on train sample")

    best_order, best_model = select_best_arma(y_train, max_p=3, max_q=3)

    arma_orders_in_sample[name] = best_order
    arma_results_in_sample[name] = best_model

    print(f"Selected ARMA order for {name}: {best_order}")
    print(best_model.summary())

    arma_resid_train = best_model.resid

    residual_diagnostics(
        arma_resid_train,
        title=f"{name} - ARMA{best_order} residuals (train sample)"
    )

# COMMAND ----------

# ==========================================================
# MODEL 4: ARMA-GARCH(1,1)
# Practical implementation in two steps:
# 1) fit ARMA mean
# 2) fit GARCH(1,1) on ARMA residuals
#
# Residual diagnostics are run on standardized residuals
# ==========================================================

arma_garch_results = {}

for name, y in series_dict.items():
    print(f"\n{name} - ARMA-GARCH(1,1)")

    # Step 1: ARMA mean
    best_order, arma_model = select_best_arma(y, max_p=3, max_q=3)
    arma_resid = arma_model.resid.dropna()

    print(f"Selected ARMA mean for {name}: {best_order}")

    # Step 2: GARCH on ARMA residuals
    garch = arch_model(
        arma_resid,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    ).fit(disp="off")

    arma_garch_results[name] = {
        "arma_order": best_order,
        "arma_model": arma_model,
        "garch_model": garch
    }

    print(garch.summary())

    std_resid = pd.Series(garch.std_resid).dropna()

    residual_diagnostics(
        std_resid,
        title=f"{name} - ARMA-GARCH standardized residuals"
    )

# COMMAND ----------

# ==========================================================
# MODEL 5: ARMA-EGARCH(1,1)
# Estimated only on the training sample
# ==========================================================

egarch_results_in_sample = {}

for name, parts in split_data.items():
    y_train = parts["train"]

    print(f"\n{name} - ARMA-EGARCH(1,1) estimated on train sample")

    # Step 1: ARMA mean on train
    best_order, arma_model = select_best_arma(y_train, max_p=3, max_q=3)
    arma_resid_train = arma_model.resid.dropna()

    print(f"Selected ARMA mean for {name}: {best_order}")

    # Step 2: EGARCH on train residuals
    egarch = arch_model(
        arma_resid_train,
        mean="Zero",
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist="normal"
    ).fit(disp="off")

    egarch_results_in_sample[name] = {
        "train_series": y_train,
        "arma_order": best_order,
        "arma_model": arma_model,
        "egarch_model": egarch
    }

    print(egarch.summary())

    std_resid_train = pd.Series(egarch.std_resid).dropna()

    residual_diagnostics(
        std_resid_train,
        title=f"{name} - ARMA-EGARCH standardized residuals (train sample)"
    )

# COMMAND ----------

# ==========================================================
# MODEL 6: ARMA-GJR-GARCH(1,1)
# Estimated only on the training sample
# ==========================================================

gjr_results_in_sample = {}

for name, parts in split_data.items():
    y_train = parts["train"]

    print(f"\n{name} - ARMA-GJR-GARCH(1,1) estimated on train sample")

    # Step 1: ARMA mean on train
    best_order, arma_model = select_best_arma(y_train, max_p=3, max_q=3)
    arma_resid_train = arma_model.resid.dropna()

    print(f"Selected ARMA mean for {name}: {best_order}")

    # Step 2: GJR-GARCH on train residuals
    gjr = arch_model(
        arma_resid_train,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal"
    ).fit(disp="off")

    gjr_results_in_sample[name] = {
        "train_series": y_train,
        "arma_order": best_order,
        "arma_model": arma_model,
        "gjr_model": gjr
    }

    print(gjr.summary())

    std_resid_train = pd.Series(gjr.std_resid).dropna()

    residual_diagnostics(
        std_resid_train,
        title=f"{name} - ARMA-GJR-GARCH standardized residuals (train sample)"
    )

# COMMAND ----------

# ==========================================================
# MODEL 7: MARKOV-SWITCHING (2 regimes)
# Estimated only on the training sample
# ==========================================================

ms_results_in_sample = {}

for name, parts in split_data.items():
    y_train = parts["train"]

    print(f"\n{name} - Markov-Switching (2 regimes) estimated on train sample")

    ms = MarkovRegression(
        y_train,
        k_regimes=2,
        trend="c",
        switching_variance=True
    ).fit(disp=False)

    ms_results_in_sample[name] = ms

    print(ms.summary())

    ms_fitted_train = pd.Series(ms.fittedvalues, index=y_train.index)
    ms_resid_train = y_train - ms_fitted_train

    residual_diagnostics(
        ms_resid_train,
        title=f"{name} - Markov-Switching residuals (train sample)"
    )

    regime_probs = pd.DataFrame(
        ms.smoothed_marginal_probabilities,
        index=y_train.index
    )
    regime_probs.columns = [f"Regime_{i}" for i in regime_probs.columns]

    regime_probs.plot(figsize=(12, 4), title=f"{name} - Smoothed regime probabilities (train sample)")
    plt.grid(True, alpha=0.3)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC FORECASTING IG SERIES
# MAGIC

# COMMAND ----------

# ==========================================================
# IG FORECASTING SETUP
# Winners:
# 1) Random Walk / no-change
# 2) ARMA(1,2)
# 3) ARMA(1,2)-GJR-GARCH(1,1)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


# ----------------------------------------------------------
# 1) Recover IG train/test from previous split
# ----------------------------------------------------------

ig_train = split_data["IG"]["train"].copy()
ig_test = split_data["IG"]["test"].copy()
ig_full = split_data["IG"]["full"].copy()

print("IG train observations:", len(ig_train))
print("IG test observations: ", len(ig_test))
print("Train period:", ig_train.index.min().date(), "to", ig_train.index.max().date())
print("Test period: ", ig_test.index.min().date(), "to", ig_test.index.max().date())


# ----------------------------------------------------------
# 2) Small helper functions
# ----------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_forecasts(df, actual_col, pred_cols):
    rows = []
    for col in pred_cols:
        rows.append({
            "model": col,
            "RMSE": rmse(df[actual_col], df[col]),
            "MAE": mean_absolute_error(df[actual_col], df[col])
        })
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

# COMMAND ----------

# ==========================================================
# STATIC OUT-OF-SAMPLE FORECAST
# Train once on ig_train and forecast the whole ig_test block
# Models:
# 1) Random Walk
# 2) ARMA(1,2)
# 3) ARMA(1,2)-GJR-GARCH(1,1)
# ==========================================================

h = len(ig_test)

# ----------------------------------------------------------
# 1) Random Walk / no-change benchmark
# ----------------------------------------------------------
rw_static_forecast = pd.Series(0.0, index=ig_test.index)

# ----------------------------------------------------------
# 2) ARMA(1,2)
# ----------------------------------------------------------
arma_static_model = ARIMA(ig_train, order=(1, 0, 2)).fit()
arma_static_forecast = pd.Series(
    arma_static_model.forecast(steps=h).values,
    index=ig_test.index
)

# ----------------------------------------------------------
# 3) ARMA(1,2)-GJR-GARCH(1,1)
# Mean forecast is still the ARMA mean forecast
# GJR-GARCH adds the multi-step conditional volatility forecast
# ----------------------------------------------------------
arma_static_resid = arma_static_model.resid.dropna()

gjr_static_model = arch_model(
    arma_static_resid,
    mean="Zero",
    vol="GARCH",
    p=1,
    o=1,
    q=1,
    dist="normal"
).fit(disp="off")

gjr_static_var_fcst = gjr_static_model.forecast(horizon=h, reindex=False).variance.iloc[-1].values
gjr_static_sigma_fcst = np.sqrt(gjr_static_var_fcst)

arma_gjr_static_forecast = arma_static_forecast.copy()

# ----------------------------------------------------------
# 4) Store static forecasts
# ----------------------------------------------------------
ig_static_fcst = pd.DataFrame({
    "actual": ig_test,
    "RW_forecast": rw_static_forecast,
    "ARMA_12_forecast": arma_static_forecast,
    "ARMA_12_GJR_forecast": arma_gjr_static_forecast,
    "ARMA_12_GJR_sigma": gjr_static_sigma_fcst
}, index=ig_test.index)

print("Static forecast preview:")
print(ig_static_fcst.head())

# ----------------------------------------------------------
# 5) Static forecast evaluation
# ----------------------------------------------------------
static_eval = evaluate_forecasts(
    ig_static_fcst,
    actual_col="actual",
    pred_cols=["RW_forecast", "ARMA_12_forecast", "ARMA_12_GJR_forecast"]
)

print("\nStatic out-of-sample forecast evaluation:")
print(static_eval.round(4))

# COMMAND ----------

# ==========================================================
# ROLLING ONE-STEP-AHEAD FORECAST
# Expanding window:
# at each step, re-estimate the model using all data
# available up to t-1 and forecast t
#
# Models:
# 1) Random Walk
# 2) ARMA(1,2)
# 3) ARMA(1,2)-GJR-GARCH(1,1)
# ==========================================================

rolling_dates = ig_test.index
train_size = len(ig_train)

rw_roll_preds = []
arma_roll_preds = []
gjr_roll_preds = []
gjr_roll_sigmas = []
actuals_roll = []

for i, current_date in enumerate(rolling_dates):
    # Expanding window: all observations available before the current test point
    y_hist = ig_full.iloc[:train_size + i].copy()
    y_true = ig_full.iloc[train_size + i]

    # ------------------------------------------------------
    # 1) Random Walk / no-change benchmark
    # ------------------------------------------------------
    rw_pred = 0.0

    # ------------------------------------------------------
    # 2) ARMA(1,2)
    # ------------------------------------------------------
    arma_roll_model = ARIMA(y_hist, order=(1, 0, 2)).fit()
    arma_pred = float(arma_roll_model.forecast(steps=1).iloc[0])

    # ------------------------------------------------------
    # 3) ARMA(1,2)-GJR-GARCH(1,1)
    # Two-step setup:
    # - ARMA models the conditional mean
    # - GJR-GARCH models conditional volatility
    #
    # Important:
    # In this setup, the point forecast for the mean is still
    # the ARMA forecast. GJR-GARCH adds the 1-step-ahead sigma.
    # ------------------------------------------------------
    arma_roll_resid = arma_roll_model.resid.dropna()

    gjr_roll_model = arch_model(
        arma_roll_resid,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal"
    ).fit(disp="off")

    gjr_var_1 = float(gjr_roll_model.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    gjr_sigma_1 = np.sqrt(gjr_var_1)

    gjr_pred = arma_pred

    # ------------------------------------------------------
    # Store results
    # ------------------------------------------------------
    actuals_roll.append(y_true)
    rw_roll_preds.append(rw_pred)
    arma_roll_preds.append(arma_pred)
    gjr_roll_preds.append(gjr_pred)
    gjr_roll_sigmas.append(gjr_sigma_1)

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(rolling_dates)} rolling forecasts")

ig_rolling_fcst = pd.DataFrame({
    "actual": actuals_roll,
    "RW_forecast": rw_roll_preds,
    "ARMA_12_forecast": arma_roll_preds,
    "ARMA_12_GJR_forecast": gjr_roll_preds,
    "ARMA_12_GJR_sigma": gjr_roll_sigmas
}, index=rolling_dates)

print("Rolling forecast preview:")
print(ig_rolling_fcst.head())

# ----------------------------------------------------------
# Rolling forecast evaluation
# ----------------------------------------------------------
rolling_eval = evaluate_forecasts(
    ig_rolling_fcst,
    actual_col="actual",
    pred_cols=["RW_forecast", "ARMA_12_forecast", "ARMA_12_GJR_forecast"]
)

print("\nRolling one-step-ahead forecast evaluation:")
print(rolling_eval.round(4))

# COMMAND ----------

# ==========================================================
# PLOTS: static and rolling forecasts
# Static and rolling both use GJR-GARCH
# ==========================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# -----------------------
# Static forecast plot
# -----------------------
axes[0].plot(ig_static_fcst.index, ig_static_fcst["actual"], label="Actual")
axes[0].plot(ig_static_fcst.index, ig_static_fcst["RW_forecast"], label="Random Walk")
axes[0].plot(ig_static_fcst.index, ig_static_fcst["ARMA_12_forecast"], label="ARMA(1,2)")
axes[0].plot(ig_static_fcst.index, ig_static_fcst["ARMA_12_GJR_forecast"], label="ARMA(1,2)-GJR-GARCH(1,1)", linestyle="--")
axes[0].set_title("IG - Static out-of-sample forecasts")
axes[0].set_ylabel("Weekly spread change (bp)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# -----------------------
# Rolling forecast plot
# -----------------------
axes[1].plot(ig_rolling_fcst.index, ig_rolling_fcst["actual"], label="Actual")
axes[1].plot(ig_rolling_fcst.index, ig_rolling_fcst["RW_forecast"], label="Random Walk")
axes[1].plot(ig_rolling_fcst.index, ig_rolling_fcst["ARMA_12_forecast"], label="ARMA(1,2)")
axes[1].plot(ig_rolling_fcst.index, ig_rolling_fcst["ARMA_12_GJR_forecast"], label="ARMA(1,2)-GJR-GARCH(1,1)", linestyle="--")
axes[1].set_title("IG - Rolling one-step-ahead forecasts")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Weekly spread change (bp)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# VOLATILITY FORECASTS
# Static: GJR-GARCH sigma
# Rolling: GJR-GARCH sigma
# ==========================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(ig_static_fcst.index, ig_static_fcst["ARMA_12_GJR_sigma"])
axes[0].set_title("IG - Static GJR-GARCH conditional volatility forecast")
axes[0].set_ylabel("Forecast sigma")
axes[0].grid(True, alpha=0.3)

axes[1].plot(ig_rolling_fcst.index, ig_rolling_fcst["ARMA_12_GJR_sigma"])
axes[1].set_title("IG - Rolling one-step-ahead GJR-GARCH conditional volatility forecast")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Forecast sigma")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# COMBINED EVALUATION TABLE
# Static and rolling both use GJR-GARCH
# ==========================================================

static_eval["method"] = "Static OOS"
rolling_eval["method"] = "Rolling 1-step"

ig_forecast_eval = pd.concat([static_eval, rolling_eval], axis=0, ignore_index=True)
ig_forecast_eval = ig_forecast_eval[["method", "model", "RMSE", "MAE"]]

ig_forecast_eval["model"] = ig_forecast_eval["model"].replace({
    "RW_forecast": "Random Walk",
    "ARMA_12_forecast": "ARMA(1,2)",
    "ARMA_12_GJR_forecast": "ARMA(1,2)-GJR-GARCH(1,1)"
})

print("IG forecast evaluation summary:")
print(ig_forecast_eval.round(4))

# COMMAND ----------

# ==========================================================
# DIEBOLD-MARIANO TEST FOR IG
# Compares forecast accuracy against Random Walk
# Loss function: squared error
# ==========================================================

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(y_true, pred_1, pred_2, power=2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    H0: both models have equal expected forecast loss
    H1: forecast losses are different

    Parameters
    ----------
    y_true : array-like
        Actual values
    pred_1 : array-like
        Forecasts from model 1
    pred_2 : array-like
        Forecasts from model 2
    power : int
        Loss power. 2 = squared error
    h : int
        Forecast horizon. For rolling 1-step, use h=1

    Returns
    -------
    dict with DM statistic, p-value and mean loss differential
    """

    y_true = np.asarray(y_true)
    pred_1 = np.asarray(pred_1)
    pred_2 = np.asarray(pred_2)

    e1 = y_true - pred_1
    e2 = y_true - pred_2

    d = np.abs(e1)**power - np.abs(e2)**power
    d = pd.Series(d).dropna().values

    T = len(d)
    d_bar = np.mean(d)

    # HAC variance estimate with truncation lag h-1
    gamma0 = np.var(d, ddof=1)

    if h > 1:
        gamma = []
        for lag in range(1, h):
            cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
            gamma.append(cov)
        var_d = gamma0 + 2 * np.sum(gamma)
    else:
        var_d = gamma0

    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "DM_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_bar
    }


# ----------------------------------------------------------
# 1) STATIC OOS
# ----------------------------------------------------------

ig_dm_static_rw_vs_arma = diebold_mariano_test(
    y_true=ig_static_fcst["actual"],
    pred_1=ig_static_fcst["RW_forecast"],
    pred_2=ig_static_fcst["ARMA_12_forecast"],
    power=2,
    h=1
)

ig_dm_static_rw_vs_gjr = diebold_mariano_test(
    y_true=ig_static_fcst["actual"],
    pred_1=ig_static_fcst["RW_forecast"],
    pred_2=ig_static_fcst["ARMA_12_GJR_forecast"],
    power=2,
    h=1
)

# ----------------------------------------------------------
# 2) ROLLING 1-STEP
# ----------------------------------------------------------

ig_dm_rolling_rw_vs_arma = diebold_mariano_test(
    y_true=ig_rolling_fcst["actual"],
    pred_1=ig_rolling_fcst["RW_forecast"],
    pred_2=ig_rolling_fcst["ARMA_12_forecast"],
    power=2,
    h=1
)

ig_dm_rolling_rw_vs_gjr = diebold_mariano_test(
    y_true=ig_rolling_fcst["actual"],
    pred_1=ig_rolling_fcst["RW_forecast"],
    pred_2=ig_rolling_fcst["ARMA_12_GJR_forecast"],
    power=2,
    h=1
)

# ----------------------------------------------------------
# 3) Results table
# Positive mean_loss_diff means model 2 improves on model 1
# because RW loss - alternative model loss > 0
# ----------------------------------------------------------

ig_dm_results = pd.DataFrame([
    {
        "method": "Static OOS",
        "comparison": "RW vs ARMA(1,2)",
        **ig_dm_static_rw_vs_arma
    },
    {
        "method": "Static OOS",
        "comparison": "RW vs ARMA(1,2)-GJR-GARCH(1,1)",
        **ig_dm_static_rw_vs_gjr
    },
    {
        "method": "Rolling 1-step",
        "comparison": "RW vs ARMA(1,2)",
        **ig_dm_rolling_rw_vs_arma
    },
    {
        "method": "Rolling 1-step",
        "comparison": "RW vs ARMA(1,2)-GJR-GARCH(1,1)",
        **ig_dm_rolling_rw_vs_gjr
    }
])

print("IG - Diebold-Mariano test results")
print(ig_dm_results.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC FORECASTING HY SERIES

# COMMAND ----------

# ==========================================================
# HY FORECASTING SETUP
# Winners:
# 1) Random Walk / no-change
# 2) ARMA(3,2)
# 3) ARMA(3,2)-GJR-GARCH(1,1)
# ==========================================================

hy_train = split_data["HY"]["train"].copy()
hy_test = split_data["HY"]["test"].copy()
hy_full = split_data["HY"]["full"].copy()

print("HY train observations:", len(hy_train))
print("HY test observations: ", len(hy_test))
print("Train period:", hy_train.index.min().date(), "to", hy_train.index.max().date())
print("Test period: ", hy_test.index.min().date(), "to", hy_test.index.max().date())

# COMMAND ----------

# ==========================================================
# STATIC OUT-OF-SAMPLE FORECAST
# Train once on hy_train and forecast the whole hy_test block
# Models:
# 1) Random Walk
# 2) ARMA(3,2)
# 3) ARMA(3,2)-GJR-GARCH(1,1)
# ==========================================================

h = len(hy_test)

# 1) Random Walk / no-change benchmark
rw_static_forecast = pd.Series(0.0, index=hy_test.index)

# 2) ARMA(3,2)
arma_static_model = ARIMA(hy_train, order=(3, 0, 2)).fit()
arma_static_forecast = pd.Series(
    arma_static_model.forecast(steps=h).values,
    index=hy_test.index
)

# 3) ARMA(3,2)-GJR-GARCH(1,1)
arma_static_resid = arma_static_model.resid.dropna()

gjr_static_model = arch_model(
    arma_static_resid,
    mean="Zero",
    vol="GARCH",
    p=1,
    o=1,
    q=1,
    dist="normal"
).fit(disp="off")

gjr_static_var_fcst = gjr_static_model.forecast(horizon=h, reindex=False).variance.iloc[-1].values
gjr_static_sigma_fcst = np.sqrt(gjr_static_var_fcst)

arma_gjr_static_forecast = arma_static_forecast.copy()

# Store static forecasts
hy_static_fcst = pd.DataFrame({
    "actual": hy_test,
    "RW_forecast": rw_static_forecast,
    "ARMA_32_forecast": arma_static_forecast,
    "ARMA_32_GJR_forecast": arma_gjr_static_forecast,
    "ARMA_32_GJR_sigma": gjr_static_sigma_fcst
}, index=hy_test.index)

print("Static forecast preview:")
print(hy_static_fcst.head())

# Static forecast evaluation
static_eval_hy = evaluate_forecasts(
    hy_static_fcst,
    actual_col="actual",
    pred_cols=["RW_forecast", "ARMA_32_forecast", "ARMA_32_GJR_forecast"]
)

print("\nStatic out-of-sample forecast evaluation:")
print(static_eval_hy.round(4))

# COMMAND ----------

# ==========================================================
# ROLLING ONE-STEP-AHEAD FORECAST
# Expanding window:
# at each step, re-estimate the model using all data
# available up to t-1 and forecast t
#
# Models:
# 1) Random Walk
# 2) ARMA(3,2)
# 3) ARMA(3,2)-GJR-GARCH(1,1)
# ==========================================================

rolling_dates = hy_test.index
train_size = len(hy_train)

rw_roll_preds = []
arma_roll_preds = []
gjr_roll_preds = []
gjr_roll_sigmas = []
actuals_roll = []

for i, current_date in enumerate(rolling_dates):
    y_hist = hy_full.iloc[:train_size + i].copy()
    y_true = hy_full.iloc[train_size + i]

    # 1) Random Walk
    rw_pred = 0.0

    # 2) ARMA(3,2)
    arma_roll_model = ARIMA(y_hist, order=(3, 0, 2)).fit()
    arma_pred = float(arma_roll_model.forecast(steps=1).iloc[0])

    # 3) ARMA(3,2)-GJR-GARCH(1,1)
    arma_roll_resid = arma_roll_model.resid.dropna()

    gjr_roll_model = arch_model(
        arma_roll_resid,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal"
    ).fit(disp="off")

    gjr_var_1 = float(gjr_roll_model.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    gjr_sigma_1 = np.sqrt(gjr_var_1)

    gjr_pred = arma_pred

    actuals_roll.append(y_true)
    rw_roll_preds.append(rw_pred)
    arma_roll_preds.append(arma_pred)
    gjr_roll_preds.append(gjr_pred)
    gjr_roll_sigmas.append(gjr_sigma_1)

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(rolling_dates)} rolling forecasts")

hy_rolling_fcst = pd.DataFrame({
    "actual": actuals_roll,
    "RW_forecast": rw_roll_preds,
    "ARMA_32_forecast": arma_roll_preds,
    "ARMA_32_GJR_forecast": gjr_roll_preds,
    "ARMA_32_GJR_sigma": gjr_roll_sigmas
}, index=rolling_dates)

print("Rolling forecast preview:")
print(hy_rolling_fcst.head())

# Rolling forecast evaluation
rolling_eval_hy = evaluate_forecasts(
    hy_rolling_fcst,
    actual_col="actual",
    pred_cols=["RW_forecast", "ARMA_32_forecast", "ARMA_32_GJR_forecast"]
)

print("\nRolling one-step-ahead forecast evaluation:")
print(rolling_eval_hy.round(4))

# COMMAND ----------

# ==========================================================
# PLOTS: static and rolling forecasts
# Static and rolling both use GJR-GARCH
# ==========================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Static
axes[0].plot(hy_static_fcst.index, hy_static_fcst["actual"], label="Actual")
axes[0].plot(hy_static_fcst.index, hy_static_fcst["RW_forecast"], label="Random Walk")
axes[0].plot(hy_static_fcst.index, hy_static_fcst["ARMA_32_forecast"], label="ARMA(3,2)")
axes[0].plot(hy_static_fcst.index, hy_static_fcst["ARMA_32_GJR_forecast"], label="ARMA(3,2)-GJR-GARCH(1,1)", linestyle="--")
axes[0].set_title("HY - Static out-of-sample forecasts")
axes[0].set_ylabel("Weekly spread change (bp)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Rolling
axes[1].plot(hy_rolling_fcst.index, hy_rolling_fcst["actual"], label="Actual")
axes[1].plot(hy_rolling_fcst.index, hy_rolling_fcst["RW_forecast"], label="Random Walk")
axes[1].plot(hy_rolling_fcst.index, hy_rolling_fcst["ARMA_32_forecast"], label="ARMA(3,2)")
axes[1].plot(hy_rolling_fcst.index, hy_rolling_fcst["ARMA_32_GJR_forecast"], label="ARMA(3,2)-GJR-GARCH(1,1)", linestyle="--")
axes[1].set_title("HY - Rolling one-step-ahead forecasts")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Weekly spread change (bp)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# VOLATILITY FORECASTS
# Static: GJR-GARCH sigma
# Rolling: GJR-GARCH sigma
# ==========================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(hy_static_fcst.index, hy_static_fcst["ARMA_32_GJR_sigma"])
axes[0].set_title("HY - Static GJR-GARCH conditional volatility forecast")
axes[0].set_ylabel("Forecast sigma")
axes[0].grid(True, alpha=0.3)

axes[1].plot(hy_rolling_fcst.index, hy_rolling_fcst["ARMA_32_GJR_sigma"])
axes[1].set_title("HY - Rolling one-step-ahead GJR-GARCH conditional volatility forecast")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Forecast sigma")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# ==========================================================
# COMBINED EVALUATION TABLE
# Static and rolling both use GJR-GARCH
# ==========================================================

static_eval_hy["method"] = "Static OOS"
rolling_eval_hy["method"] = "Rolling 1-step"

hy_forecast_eval = pd.concat([static_eval_hy, rolling_eval_hy], axis=0, ignore_index=True)
hy_forecast_eval = hy_forecast_eval[["method", "model", "RMSE", "MAE"]]

hy_forecast_eval["model"] = hy_forecast_eval["model"].replace({
    "RW_forecast": "Random Walk",
    "ARMA_32_forecast": "ARMA(3,2)",
    "ARMA_32_GJR_forecast": "ARMA(3,2)-GJR-GARCH(1,1)"
})

print("HY forecast evaluation summary:")
print(hy_forecast_eval.round(4))

# COMMAND ----------

# ==========================================================
# DIEBOLD-MARIANO TEST FOR HY
# Compares forecast accuracy against Random Walk
# Loss function: squared error
# ==========================================================

# ----------------------------------------------------------
# 1) STATIC OOS
# ----------------------------------------------------------

hy_dm_static_rw_vs_arma = diebold_mariano_test(
    y_true=hy_static_fcst["actual"],
    pred_1=hy_static_fcst["RW_forecast"],
    pred_2=hy_static_fcst["ARMA_32_forecast"],
    power=2,
    h=1
)

hy_dm_static_rw_vs_gjr = diebold_mariano_test(
    y_true=hy_static_fcst["actual"],
    pred_1=hy_static_fcst["RW_forecast"],
    pred_2=hy_static_fcst["ARMA_32_GJR_forecast"],
    power=2,
    h=1
)

# ----------------------------------------------------------
# 2) ROLLING 1-STEP
# ----------------------------------------------------------

hy_dm_rolling_rw_vs_arma = diebold_mariano_test(
    y_true=hy_rolling_fcst["actual"],
    pred_1=hy_rolling_fcst["RW_forecast"],
    pred_2=hy_rolling_fcst["ARMA_32_forecast"],
    power=2,
    h=1
)

hy_dm_rolling_rw_vs_gjr = diebold_mariano_test(
    y_true=hy_rolling_fcst["actual"],
    pred_1=hy_rolling_fcst["RW_forecast"],
    pred_2=hy_rolling_fcst["ARMA_32_GJR_forecast"],
    power=2,
    h=1
)

# ----------------------------------------------------------
# 3) Results table
# Positive mean_loss_diff means model 2 improves on model 1
# ----------------------------------------------------------

hy_dm_results = pd.DataFrame([
    {
        "method": "Static OOS",
        "comparison": "RW vs ARMA(3,2)",
        **hy_dm_static_rw_vs_arma
    },
    {
        "method": "Static OOS",
        "comparison": "RW vs ARMA(3,2)-GJR-GARCH(1,1)",
        **hy_dm_static_rw_vs_gjr
    },
    {
        "method": "Rolling 1-step",
        "comparison": "RW vs ARMA(3,2)",
        **hy_dm_rolling_rw_vs_arma
    },
    {
        "method": "Rolling 1-step",
        "comparison": "RW vs ARMA(3,2)-GJR-GARCH(1,1)",
        **hy_dm_rolling_rw_vs_gjr
    }
])

print("HY - Diebold-Mariano test results")
print(hy_dm_results.round(4))

# COMMAND ----------

# ==========================================================
# VOLATILITY CLUSTERING AND ASYMMETRY DIAGNOSTICS
# Purpose:
# Check whether ARMA-GJR-GARCH adds value beyond point forecasting
# by modelling conditional volatility and asymmetric volatility effects.
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.api import OLS, add_constant
from arch import arch_model


# ----------------------------------------------------------
# 1) Series and ARMA orders
# ----------------------------------------------------------

series_to_check = {
    "IG": {
        "series": split_data["IG"]["train"],
        "arma_order": (1, 0, 2)
    },
    "HY": {
        "series": split_data["HY"]["train"],
        "arma_order": (3, 0, 2)
    }
}

# If you also want rating buckets later, add them here:
# "BBB": {"series": bbb_train, "arma_order": (1, 0, 2)}
# "BB": {"series": bb_train, "arma_order": (3, 0, 2)}
# "B": {"series": b_train, "arma_order": (3, 0, 2)}


# ----------------------------------------------------------
# 2) Diagnostic function
# ----------------------------------------------------------

def volatility_asymmetry_diagnostics(y, name, arma_order):
    print("=" * 100)
    print(f"{name} - Volatility clustering and asymmetry diagnostics")
    print("=" * 100)

    y = y.dropna().copy()

    # ------------------------------------------------------
    # ARMA mean model
    # ------------------------------------------------------
    arma_model = ARIMA(y, order=arma_order).fit()
    resid = arma_model.resid.dropna()

    print("\nARMA model:")
    print(arma_model.summary())

    # ------------------------------------------------------
    # Ljung-Box on residuals and squared residuals
    # ------------------------------------------------------
    lb_resid = acorr_ljungbox(resid, lags=[4, 8, 12], return_df=True)
    lb_sq_resid = acorr_ljungbox(resid**2, lags=[4, 8, 12], return_df=True)

    print("\nLjung-Box on ARMA residuals:")
    print(lb_resid.round(4))

    print("\nLjung-Box on squared ARMA residuals:")
    print(lb_sq_resid.round(4))

    # ------------------------------------------------------
    # ARCH-LM test
    # ------------------------------------------------------
    arch_lm_stat, arch_lm_pvalue, arch_f_stat, arch_f_pvalue = het_arch(resid, nlags=8)

    print("\nARCH-LM test on ARMA residuals:")
    print(f"LM statistic = {arch_lm_stat:.4f}")
    print(f"LM p-value   = {arch_lm_pvalue:.6f}")
    print(f"F statistic  = {arch_f_stat:.4f}")
    print(f"F p-value    = {arch_f_pvalue:.6f}")

    if arch_lm_pvalue < 0.05:
        print("Interpretation: evidence of ARCH effects / conditional heteroskedasticity.")
    else:
        print("Interpretation: no strong evidence of ARCH effects.")

    # ------------------------------------------------------
    # Simple asymmetry / sign-bias style regression
    # ------------------------------------------------------
    # If negative residuals have a different effect on future squared residuals,
    # this supports asymmetric volatility models such as GJR-GARCH.
    asym_df = pd.DataFrame({
        "resid": resid
    })

    asym_df["resid_sq"] = asym_df["resid"] ** 2
    asym_df["resid_lag"] = asym_df["resid"].shift(1)
    asym_df["neg_lag"] = (asym_df["resid_lag"] < 0).astype(int)
    asym_df["neg_size_lag"] = asym_df["neg_lag"] * asym_df["resid_lag"]
    asym_df["pos_size_lag"] = (1 - asym_df["neg_lag"]) * asym_df["resid_lag"]

    asym_df = asym_df.dropna()

    X = asym_df[["neg_lag", "neg_size_lag", "pos_size_lag"]]
    X = add_constant(X)
    y_asym = asym_df["resid_sq"]

    asym_model = OLS(y_asym, X).fit()

    print("\nSign-bias / asymmetry regression:")
    print(asym_model.summary())

    # ------------------------------------------------------
    # Fit GJR-GARCH on ARMA residuals
    # ------------------------------------------------------
    gjr_model = arch_model(
        resid,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal"
    ).fit(disp="off")

    print("\nGJR-GARCH model on ARMA residuals:")
    print(gjr_model.summary())

    std_resid = pd.Series(gjr_model.std_resid, index=resid.index).dropna()

    # ------------------------------------------------------
    # Post-GJR diagnostics
    # ------------------------------------------------------
    lb_std = acorr_ljungbox(std_resid, lags=[4, 8, 12], return_df=True)
    lb_std_sq = acorr_ljungbox(std_resid**2, lags=[4, 8, 12], return_df=True)

    print("\nLjung-Box on standardized GJR residuals:")
    print(lb_std.round(4))

    print("\nLjung-Box on squared standardized GJR residuals:")
    print(lb_std_sq.round(4))

    # ------------------------------------------------------
    # Plots
    # ------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(resid.index, resid)
    axes[0].set_title(f"{name} - ARMA residuals")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(resid.index, resid**2)
    axes[1].set_title(f"{name} - Squared ARMA residuals")
    axes[1].set_ylabel("Squared residual")
    axes[1].grid(True, alpha=0.3)

    conditional_vol = gjr_model.conditional_volatility
    axes[2].plot(conditional_vol.index, conditional_vol)
    axes[2].set_title(f"{name} - GJR-GARCH conditional volatility")
    axes[2].set_ylabel("Conditional sigma")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "arma_model": arma_model,
        "arma_resid": resid,
        "gjr_model": gjr_model,
        "std_resid": std_resid,
        "lb_resid": lb_resid,
        "lb_sq_resid": lb_sq_resid,
        "arch_lm_pvalue": arch_lm_pvalue,
        "asymmetry_model": asym_model,
        "lb_std": lb_std,
        "lb_std_sq": lb_std_sq
    }


# ----------------------------------------------------------
# 3) Run diagnostics
# ----------------------------------------------------------

volatility_diagnostics_results = {}

for name, info in series_to_check.items():
    volatility_diagnostics_results[name] = volatility_asymmetry_diagnostics(
        y=info["series"],
        name=name,
        arma_order=info["arma_order"]
    )