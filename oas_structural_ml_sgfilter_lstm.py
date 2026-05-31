# Databricks notebook source
# MAGIC %md
# MAGIC # Structural ML extension for corporate bond OAS
# MAGIC
# MAGIC This notebook adapts the structure of the `ACI_Finance_SGFilter.ipynb` notebook to the corporate bond OAS project.
# MAGIC
# MAGIC The idea is not to replace the econometric core of the TFG. It is a machine-learning extension: a small structural LSTM model that uses both the smoothed OAS level and a Savitzky-Golay derivative as inputs.
# MAGIC
# MAGIC Main steps:
# MAGIC 1. Load IG, HY and rating-bucket OAS series from Databricks tables.
# MAGIC 2. Convert daily OAS data to weekly `W-FRI` frequency.
# MAGIC 3. Apply a Savitzky-Golay filter to obtain a smoothed OAS series and a derivative signal.
# MAGIC 4. Build supervised learning windows similar to the ACI notebook: recent levels + recent derivatives as inputs, future levels + future derivatives as outputs.
# MAGIC 5. Train a twin-input LSTM model.
# MAGIC 6. Report RMSE tables and plots.
# MAGIC
# MAGIC Important interpretation note: this is a nonlinear ML extension. Its RMSE is reported mainly on scaled series, following the ACI notebook logic. For direct comparison with the baseline ARMA/GARCH models, the main econometric results remain the primary benchmark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Optional package installation
# MAGIC
# MAGIC Run this only if your environment does not already have TensorFlow or SciPy installed. In Databricks, after installing TensorFlow, you may need to restart Python.

# COMMAND ----------

# MAGIC %pip install tensorflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports and configuration

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

pd.set_option("display.max_columns", 140)
pd.set_option("display.width", 180)
plt.style.use("default")

# ----------------------------------------------------------
# Databricks source configuration
# ----------------------------------------------------------
CATALOG = "tfg_data"
SCHEMA = "original_data"

IG_TABLE = "ig_aggregate_oas"
HY_TABLE = "hy_aggregate_oas"
RATING_BUCKET_TABLE = "bbb_bb_b_extensiondata"

# Candidate columns. The loader tries these first and then falls back to fuzzy matching.
SERIES_SPECS = {
    "IG":  {"table": IG_TABLE, "candidates": ["BAMLC0A0CM", "IG", "ig", "value"]},
    "HY":  {"table": HY_TABLE, "candidates": ["BAMLH0A0HYM2", "HY", "hy", "value"]},
    "BBB": {"table": RATING_BUCKET_TABLE, "candidates": ["BAMLC0A4CBBB", "BBB", "bbb"]},
    "BB":  {"table": RATING_BUCKET_TABLE, "candidates": ["BAMLH0A1HYBB", "BB", "bb"]},
    "B":   {"table": RATING_BUCKET_TABLE, "candidates": ["BAMLH0A2HYB", "B", "b", "Single_B", "single_b"]},
}

DATE_CANDIDATES = ["observation_date", "date", "DATE", "Date"]

# Choose which series to run. Start with one or two if computation is slow.
SERIES_TO_RUN = ["IG", "HY", "BBB", "BB", "B"]

# ----------------------------------------------------------
# Time-series preprocessing
# ----------------------------------------------------------
WEEK_RULE = "W-FRI"
TRAIN_RATIO = 0.80
AUTO_DETECT_BP_SCALE = True

# ----------------------------------------------------------
# Savitzky-Golay / supervised-window design
# This mirrors the ACI notebook structure.
# ----------------------------------------------------------
WINDOW_LENGTH = 5      # must be odd
POLY_ORDER = 3
DESIRED_RANGE = 0.9

N_STEPS_IN_ORIGINAL = 5
N_STEPS_IN_DIFF = 4
N_STEPS_OUT = 4       # aligns better with the TFG horizons than 10; change to 10 to mimic ACI exactly
EDGE_DROP = 10        # drops early observations where filter/window alignment is less stable

# ----------------------------------------------------------
# LSTM training setup
# Keep small for a first run. Increase NUM_EXP/EPOCHS for final robustness.
# ----------------------------------------------------------
NUM_EXP = 5
EPOCHS = 40
HIDDEN = 32
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.10
PATIENCE = 8

print("Configuration loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and prepare weekly OAS series
# MAGIC
# MAGIC This cell loads the raw Databricks tables and converts the daily OAS observations into weekly Friday averages, following the same preprocessing logic used in the baseline notebooks.

# COMMAND ----------

def table_name(table):
    return f"{CATALOG}.{SCHEMA}.{table}"


def read_databricks_table(table):
    """Read a Databricks table into pandas."""
    return spark.table(table_name(table)).toPandas()


def find_date_col(df):
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: first column that looks like a date
    for c in df.columns:
        if "date" in c.lower():
            return c
    raise ValueError(f"No date column found. Columns available: {list(df.columns)}")


def find_value_col(df, candidates, series_name):
    # exact match
    for c in candidates:
        if c in df.columns:
            return c

    # case-insensitive exact match
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    # fuzzy match
    if series_name == "BBB":
        patterns = ["bbb"]
    elif series_name == "BB":
        patterns = ["hybb", "bb"]
    elif series_name == "B":
        patterns = ["hyb", "single", "bamlh0a2", "_b"]
    elif series_name == "IG":
        patterns = ["bamlc0a0cm", "ig"]
    elif series_name == "HY":
        patterns = ["bamlh0a0hym2", "hy"]
    else:
        patterns = [series_name.lower()]

    for c in df.columns:
        cl = c.lower()
        if any(p in cl for p in patterns) and "date" not in cl:
            return c

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"No value column found for {series_name}. Columns available: {list(df.columns)}. "
        f"Please update SERIES_SPECS candidates."
    )


def to_weekly_oas(raw_df, value_col, series_name):
    """Clean, interpolate and resample a raw OAS series to weekly frequency."""
    date_col = find_date_col(raw_df)

    df = raw_df[[date_col, value_col]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    # Daily calendar + interpolation. This avoids zero imputation.
    daily = df.set_index("date").asfreq("D")
    daily["value"] = daily["value"].interpolate(method="time", limit_direction="both").ffill().bfill()

    weekly = daily.resample(WEEK_RULE).mean(numeric_only=True)

    # Convert to basis points if the raw series is in percentage points.
    med_abs = weekly["value"].abs().median()
    if AUTO_DETECT_BP_SCALE and med_abs < 50:
        weekly[f"{series_name}_oas_bp"] = weekly["value"] * 100
        scale_note = "raw values appear to be percent; multiplied by 100 to obtain bps"
    else:
        weekly[f"{series_name}_oas_bp"] = weekly["value"]
        scale_note = "raw values appear to already be in bps; no multiplication applied"

    weekly = weekly[[f"{series_name}_oas_bp"]].dropna().reset_index().rename(columns={"date": "week_end_date"})
    return weekly, scale_note


def load_all_series(series_names):
    weekly_dict = {}
    load_notes = []

    table_cache = {}

    for name in series_names:
        spec = SERIES_SPECS[name]
        table = spec["table"]

        if table not in table_cache:
            table_cache[table] = read_databricks_table(table)

        raw = table_cache[table]
        value_col = find_value_col(raw, spec["candidates"], name)
        weekly, scale_note = to_weekly_oas(raw, value_col, name)
        weekly_dict[name] = weekly

        load_notes.append({
            "series": name,
            "table": table_name(table),
            "value_col_used": value_col,
            "n_weekly_obs": len(weekly),
            "start": weekly["week_end_date"].min(),
            "end": weekly["week_end_date"].max(),
            "scale_note": scale_note
        })

    return weekly_dict, pd.DataFrame(load_notes)


weekly_series, load_notes = load_all_series(SERIES_TO_RUN)
print("Loaded series:")
display(load_notes)

for name, df in weekly_series.items():
    print(f"{name} preview:")
    display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Visual check of the OAS series

# COMMAND ----------

plt.figure(figsize=(14, 6))
for name, df in weekly_series.items():
    plt.plot(df["week_end_date"], df[f"{name}_oas_bp"], label=name)

plt.title("Weekly OAS series by segment / rating bucket")
plt.xlabel("Date")
plt.ylabel("OAS (basis points)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Savitzky-Golay smoothing and derivative construction
# MAGIC
# MAGIC This is the key structural step taken from the ACI notebook. The model is not fed only the raw series. It receives:
# MAGIC
# MAGIC - a smoothed version of the OAS level,
# MAGIC - a derivative signal that approximates the local rate of change.
# MAGIC
# MAGIC This is the ML equivalent of trying to give the network both the level state and a short-run momentum/slope signal.

# COMMAND ----------

def make_odd_window(length, requested_window):
    """Return a valid odd Savitzky-Golay window length."""
    w = min(requested_window, length - 1 if (length - 1) % 2 == 1 else length - 2)
    if w < 3:
        raise ValueError("Series is too short for Savitzky-Golay filtering.")
    if w % 2 == 0:
        w -= 1
    return w


def apply_savgol_structure(weekly_df, series_name):
    """Apply SG smoothing, derivative extraction and scaling."""
    y = weekly_df[f"{series_name}_oas_bp"].astype(float).values
    dates = pd.to_datetime(weekly_df["week_end_date"]).values

    window = make_odd_window(len(y), WINDOW_LENGTH)
    poly = min(POLY_ORDER, window - 1)

    smoothed = savgol_filter(y, window_length=window, polyorder=poly)
    derivative = savgol_filter(y, window_length=window, polyorder=poly, deriv=1)

    max_abs_value = max(np.max(np.abs(smoothed)), np.max(np.abs(derivative)))
    if max_abs_value == 0:
        max_abs_value = 1.0

    scale_factor = DESIRED_RANGE / max_abs_value
    scaled_original = smoothed * scale_factor
    scaled_derivative = derivative * scale_factor

    out = pd.DataFrame({
        "week_end_date": dates,
        "oas_bp": y,
        "sg_smoothed_oas_bp": smoothed,
        "sg_derivative_bp": derivative,
        "scaled_original": scaled_original,
        "scaled_derivative": scaled_derivative
    })

    info = {
        "series": series_name,
        "window_length_used": window,
        "poly_order_used": poly,
        "max_abs_value": max_abs_value,
        "scale_factor": scale_factor,
        "scaled_min": min(scaled_original.min(), scaled_derivative.min()),
        "scaled_max": max(scaled_original.max(), scaled_derivative.max())
    }

    return out, info


structured_series = {}
sg_info = []

for name, df in weekly_series.items():
    structured, info = apply_savgol_structure(df, name)
    structured_series[name] = structured
    sg_info.append(info)

sg_info = pd.DataFrame(sg_info)
display(sg_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Plot original, smoothed and derivative series

# COMMAND ----------

def plot_sg_series(structured_df, series_name):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(structured_df["week_end_date"], structured_df["oas_bp"], label="Original weekly OAS", linestyle="--", alpha=0.6)
    axes[0].plot(structured_df["week_end_date"], structured_df["sg_smoothed_oas_bp"], label="SG smoothed OAS")
    axes[0].set_title(f"{series_name}: original and Savitzky-Golay smoothed OAS")
    axes[0].set_ylabel("OAS (bp)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(structured_df["week_end_date"], structured_df["sg_derivative_bp"], label="SG derivative", color="orange")
    axes[1].set_title(f"{series_name}: Savitzky-Golay derivative")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Approx. local change (bp/week)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


for name in SERIES_TO_RUN:
    plot_sg_series(structured_series[name], name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sign check between level changes and derivative
# MAGIC
# MAGIC This mirrors the diagnostic check in the ACI notebook. The derivative is not expected to match every one-period change perfectly, especially after smoothing, but many sign disagreements would suggest that the derivative is not a useful local slope signal.

# COMMAND ----------

def verify_differential_signs(original_series, derivative_series):
    discrepancies = []
    for i in range(1, len(original_series)):
        delta = original_series[i] - original_series[i - 1]
        derivative = derivative_series[i - 1]
        if (delta < 0 and derivative > 0) or (delta > 0 and derivative < 0):
            discrepancies.append((i, delta, derivative))
    return discrepancies

sign_rows = []
for name, df in structured_series.items():
    discrepancies = verify_differential_signs(df["scaled_original"].values, df["scaled_derivative"].values)
    total = len(df) - 1
    sign_rows.append({
        "series": name,
        "n_discrepancies": len(discrepancies),
        "n_comparisons": total,
        "discrepancy_rate": len(discrepancies) / total
    })

sign_check = pd.DataFrame(sign_rows)
display(sign_check)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Supervised learning dataset construction
# MAGIC
# MAGIC This reproduces the ACI notebook logic:
# MAGIC
# MAGIC - `Input1` to `Input5`: recent scaled OAS level observations.
# MAGIC - `Input_Differential1` to `Input_Differential4`: recent scaled derivative observations.
# MAGIC - `Output1` to `Output4`: future scaled OAS levels.
# MAGIC - `Output_Differential1` to `Output_Differential4`: future scaled derivatives.
# MAGIC
# MAGIC The number of output steps is set to 4 to match the 1-week and 4-week horizon logic in the TFG. You can set it to 10 if you want to mimic the ACI notebook exactly.

# COMMAND ----------

def build_supervised_dataframe(structured_df):
    original = structured_df["scaled_original"].values
    derivative = structured_df["scaled_derivative"].values
    dates = pd.to_datetime(structured_df["week_end_date"])

    max_required = max(
        N_STEPS_IN_ORIGINAL + N_STEPS_OUT,
        N_STEPS_IN_DIFF + N_STEPS_OUT
    )
    final_length = len(original) - max_required

    if final_length <= EDGE_DROP:
        raise ValueError("Not enough observations to build supervised dataset.")

    formatted = pd.DataFrame()

    # Original inputs
    for i in range(N_STEPS_IN_ORIGINAL):
        formatted[f"Input{i+1}"] = original[i:final_length + i]

    # Derivative inputs
    for i in range(N_STEPS_IN_DIFF):
        formatted[f"Input_Differential{i+1}"] = derivative[i:final_length + i]

    # Future original outputs
    for i in range(N_STEPS_OUT):
        formatted[f"Output{i+1}"] = original[(N_STEPS_IN_ORIGINAL + i):(final_length + N_STEPS_IN_ORIGINAL + i)]

    # Future derivative outputs
    for i in range(N_STEPS_OUT):
        formatted[f"Output_Differential{i+1}"] = derivative[(N_STEPS_IN_DIFF + i):(final_length + N_STEPS_IN_DIFF + i)]

    # Forecast date: date of the first output step
    formatted["forecast_start_date"] = dates.iloc[N_STEPS_IN_ORIGINAL:final_length + N_STEPS_IN_ORIGINAL].values

    # Drop early edge observations, following the spirit of the ACI notebook.
    formatted = formatted.iloc[EDGE_DROP:].reset_index(drop=True)

    return formatted


supervised_data = {}
for name, df in structured_series.items():
    supervised_data[name] = build_supervised_dataframe(df)
    print(name, supervised_data[name].shape)
    display(supervised_data[name].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Train/test split and tensor reshaping

# COMMAND ----------

def split_and_reshape(formatted_df):
    split_idx = int(len(formatted_df) * TRAIN_RATIO)

    train_df = formatted_df.iloc[:split_idx].reset_index(drop=True)
    test_df = formatted_df.iloc[split_idx:].reset_index(drop=True)

    input_original_cols = [f"Input{i+1}" for i in range(N_STEPS_IN_ORIGINAL)]
    input_diff_cols = [f"Input_Differential{i+1}" for i in range(N_STEPS_IN_DIFF)]
    output_original_cols = [f"Output{i+1}" for i in range(N_STEPS_OUT)]
    output_diff_cols = [f"Output_Differential{i+1}" for i in range(N_STEPS_OUT)]

    X_train_original = train_df[input_original_cols].values.reshape(-1, N_STEPS_IN_ORIGINAL, 1)
    X_train_diff = train_df[input_diff_cols].values.reshape(-1, N_STEPS_IN_DIFF, 1)
    Y_train_original = train_df[output_original_cols].values
    Y_train_diff = train_df[output_diff_cols].values

    X_test_original = test_df[input_original_cols].values.reshape(-1, N_STEPS_IN_ORIGINAL, 1)
    X_test_diff = test_df[input_diff_cols].values.reshape(-1, N_STEPS_IN_DIFF, 1)
    Y_test_original = test_df[output_original_cols].values
    Y_test_diff = test_df[output_diff_cols].values

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train_original": X_train_original,
        "X_train_diff": X_train_diff,
        "Y_train_original": Y_train_original,
        "Y_train_diff": Y_train_diff,
        "X_test_original": X_test_original,
        "X_test_diff": X_test_diff,
        "Y_test_original": Y_test_original,
        "Y_test_diff": Y_test_diff,
    }


ml_datasets = {}
for name, df in supervised_data.items():
    ml_datasets[name] = split_and_reshape(df)
    print(f"{name}: train={ml_datasets[name]['train_df'].shape}, test={ml_datasets[name]['test_df'].shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. TensorFlow check

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Twin-input LSTM model
# MAGIC
# MAGIC The architecture follows the ACI notebook:
# MAGIC
# MAGIC - one input branch for the smoothed OAS level window,
# MAGIC - one input branch for the derivative window,
# MAGIC - a shared LSTM transformation,
# MAGIC - concatenation of both representations,
# MAGIC - two outputs: future OAS level and future derivative.

# COMMAND ----------

def rmse_np(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def step_rmse_np(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2, axis=0))


def build_twin_lstm(hidden=HIDDEN, dropout_rate=0.0):
    n_features = 1

    input_original = Input(shape=(N_STEPS_IN_ORIGINAL, n_features), name="input_original")
    input_diff = Input(shape=(N_STEPS_IN_DIFF, n_features), name="input_diff")

    shared_lstm = LSTM(units=hidden, activation="tanh", return_sequences=False, name="shared_lstm")

    original_lstm_output = shared_lstm(input_original)
    diff_lstm_output = shared_lstm(input_diff)

    combined = Concatenate(name="combined_state")([original_lstm_output, diff_lstm_output])

    if dropout_rate > 0:
        combined = Dropout(dropout_rate)(combined)

    output_original = Dense(units=N_STEPS_OUT, activation="linear", name="output_original")(combined)
    output_diff = Dense(units=N_STEPS_OUT, activation="linear", name="output_diff")(combined)

    model = Model(inputs=[input_original, input_diff], outputs=[output_original, output_diff])
    model.compile(
        optimizer="adam",
        loss={"output_original": "mse", "output_diff": "mse"},
        loss_weights={"output_original": 1.0, "output_diff": 0.5}
    )

    return model


def run_twin_lstm_experiments(data, series_name, num_exp=NUM_EXP, epochs=EPOCHS, hidden=HIDDEN):
    X_train_original = data["X_train_original"]
    X_train_diff = data["X_train_diff"]
    Y_train_original = data["Y_train_original"]
    Y_train_diff = data["Y_train_diff"]
    X_test_original = data["X_test_original"]
    X_test_diff = data["X_test_diff"]
    Y_test_original = data["Y_test_original"]
    Y_test_diff = data["Y_test_diff"]

    train_acc_original = np.zeros(num_exp)
    train_acc_diff = np.zeros(num_exp)
    test_acc_original = np.zeros(num_exp)
    test_acc_diff = np.zeros(num_exp)
    step_rmse_original = np.zeros((num_exp, N_STEPS_OUT))
    step_rmse_diff = np.zeros((num_exp, N_STEPS_OUT))

    best_rmse = np.inf
    best_predict_test = None
    best_model = None
    histories = []

    for exp in range(num_exp):
        tf.keras.backend.clear_session()
        seed = 1234 + exp
        np.random.seed(seed)
        tf.random.set_seed(seed)

        model = build_twin_lstm(hidden=hidden)
        early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        history = model.fit(
            [X_train_original, X_train_diff],
            [Y_train_original, Y_train_diff],
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[early_stop],
            verbose=0
        )
        histories.append(history.history)

        pred_train_original, pred_train_diff = model.predict([X_train_original, X_train_diff], verbose=0)
        pred_test_original, pred_test_diff = model.predict([X_test_original, X_test_diff], verbose=0)

        train_acc_original[exp] = rmse_np(Y_train_original, pred_train_original)
        train_acc_diff[exp] = rmse_np(Y_train_diff, pred_train_diff)
        test_acc_original[exp] = rmse_np(Y_test_original, pred_test_original)
        test_acc_diff[exp] = rmse_np(Y_test_diff, pred_test_diff)
        step_rmse_original[exp, :] = step_rmse_np(Y_test_original, pred_test_original)
        step_rmse_diff[exp, :] = step_rmse_np(Y_test_diff, pred_test_diff)

        print(
            f"{series_name} | Exp {exp+1}/{num_exp} | "
            f"Train RMSE original={train_acc_original[exp]:.5f}, "
            f"Test RMSE original={test_acc_original[exp]:.5f}, "
            f"Test RMSE diff={test_acc_diff[exp]:.5f}"
        )

        if test_acc_original[exp] < best_rmse:
            best_rmse = test_acc_original[exp]
            best_predict_test = (pred_test_original, pred_test_diff)
            best_model = model

    return {
        "train_acc_original": train_acc_original,
        "train_acc_diff": train_acc_diff,
        "test_acc_original": test_acc_original,
        "test_acc_diff": test_acc_diff,
        "step_rmse_original": step_rmse_original,
        "step_rmse_diff": step_rmse_diff,
        "best_predict_test": best_predict_test,
        "best_model": best_model,
        "histories": histories
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Result tables and plots

# COMMAND ----------

def prepare_results_table(results, target="original"):
    if target == "original":
        train_acc = results["train_acc_original"]
        test_acc = results["test_acc_original"]
        step_rmse = results["step_rmse_original"]
    else:
        train_acc = results["train_acc_diff"]
        test_acc = results["test_acc_diff"]
        step_rmse = results["step_rmse_diff"]

    num_exp = len(train_acc)
    arr = np.column_stack([train_acc, test_acc, step_rmse])
    index = [f"Exp{j+1}" for j in range(num_exp)]
    columns = ["TrainRMSE", "TestRMSE"] + [f"Step{j+1}" for j in range(N_STEPS_OUT)]
    return pd.DataFrame(np.round(arr, 5), index=index, columns=columns)


def calculate_statistics_table(results, target="original"):
    if target == "original":
        train_acc = results["train_acc_original"]
        test_acc = results["test_acc_original"]
        step_rmse = results["step_rmse_original"]
    else:
        train_acc = results["train_acc_diff"]
        test_acc = results["test_acc_diff"]
        step_rmse = results["step_rmse_diff"]

    rows = []
    labels = ["TrainRMSE", "TestRMSE"] + [f"Step{j+1}" for j in range(N_STEPS_OUT)]
    arrays = [train_acc, test_acc] + [step_rmse[:, j] for j in range(N_STEPS_OUT)]

    for label, arr in zip(labels, arrays):
        arr = np.asarray(arr)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        ci_lb = mean - 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else mean
        ci_ub = mean + 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else mean
        rows.append({
            "Metric": label,
            "Mean": mean,
            "Standard Deviation": std,
            "CI_LB": ci_lb,
            "CI_UB": ci_ub,
            "Min": arr.min(),
            "Max": arr.max()
        })

    return pd.DataFrame(rows).set_index("Metric").round(5)


def plot_actual_vs_predicted(data, results, series_name):
    Y_test_original = data["Y_test_original"]
    Y_test_diff = data["Y_test_diff"]
    pred_original, pred_diff = results["best_predict_test"]
    dates = pd.to_datetime(data["test_df"]["forecast_start_date"])

    # Plot selected steps only to keep output manageable.
    steps_to_plot = sorted(set([0, min(3, N_STEPS_OUT - 1)]))

    for j in steps_to_plot:
        plt.figure(figsize=(14, 4))
        plt.plot(dates, Y_test_original[:, j], label="actual")
        plt.plot(dates, pred_original[:, j], label="predicted")
        plt.title(f"{series_name}: scaled OAS level actual vs predicted - Step {j+1}")
        plt.xlabel("Forecast start date")
        plt.ylabel("Scaled value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    for j in steps_to_plot:
        plt.figure(figsize=(14, 4))
        plt.plot(dates, Y_test_diff[:, j], label="actual")
        plt.plot(dates, pred_diff[:, j], label="predicted")
        plt.title(f"{series_name}: scaled derivative actual vs predicted - Step {j+1}")
        plt.xlabel("Forecast start date")
        plt.ylabel("Scaled derivative")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_rmse_means(stats_original_df, stats_diff_df, series_name):
    plt.figure(figsize=(8, 5))
    plt.bar(["TrainRMSE", "TestRMSE"], stats_original_df.loc[["TrainRMSE", "TestRMSE"], "Mean"],
            yerr=stats_original_df.loc[["TrainRMSE", "TestRMSE"], "Standard Deviation"], capsize=5)
    plt.title(f"{series_name}: original-series RMSE mean")
    plt.ylabel("RMSE (scaled)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(["TrainRMSE", "TestRMSE"], stats_diff_df.loc[["TrainRMSE", "TestRMSE"], "Mean"],
            yerr=stats_diff_df.loc[["TrainRMSE", "TestRMSE"], "Standard Deviation"], capsize=5)
    plt.title(f"{series_name}: derivative-series RMSE mean")
    plt.ylabel("RMSE (scaled)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    step_labels = [f"Step{j+1}" for j in range(N_STEPS_OUT)]
    plt.bar(step_labels, stats_original_df.loc[step_labels, "Mean"],
            yerr=stats_original_df.loc[step_labels, "Standard Deviation"], capsize=5)
    plt.title(f"{series_name}: step-wise RMSE mean for OAS level")
    plt.ylabel("RMSE (scaled)")
    plt.xlabel("Forecast step")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(step_labels, stats_diff_df.loc[step_labels, "Mean"],
            yerr=stats_diff_df.loc[step_labels, "Standard Deviation"], capsize=5)
    plt.title(f"{series_name}: step-wise RMSE mean for derivative")
    plt.ylabel("RMSE (scaled)")
    plt.xlabel("Forecast step")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Naive benchmark for the structural ML setup
# MAGIC
# MAGIC This benchmark is not the same as the Random Walk used in the econometric part, because here the model predicts scaled smoothed levels and derivatives. Still, it is useful:
# MAGIC
# MAGIC - Level benchmark: future OAS level equals the last observed input level.
# MAGIC - Derivative benchmark: future derivative equals zero.

# COMMAND ----------

def structural_naive_benchmark(data):
    X_test_original = data["X_test_original"]
    Y_test_original = data["Y_test_original"]
    Y_test_diff = data["Y_test_diff"]

    last_level = X_test_original[:, -1, 0].reshape(-1, 1)
    naive_original = np.repeat(last_level, N_STEPS_OUT, axis=1)
    naive_diff = np.zeros_like(Y_test_diff)

    return {
        "naive_test_rmse_original": rmse_np(Y_test_original, naive_original),
        "naive_test_rmse_diff": rmse_np(Y_test_diff, naive_diff),
        "naive_step_rmse_original": step_rmse_np(Y_test_original, naive_original),
        "naive_step_rmse_diff": step_rmse_np(Y_test_diff, naive_diff)
    }


def benchmark_table(series_name, data, results):
    naive = structural_naive_benchmark(data)
    ml_original = results["test_acc_original"].mean()
    ml_diff = results["test_acc_diff"].mean()

    out = pd.DataFrame([
        {"series": series_name, "target": "scaled_oas_level", "model": "Naive last-level", "TestRMSE": naive["naive_test_rmse_original"]},
        {"series": series_name, "target": "scaled_oas_level", "model": "Twin LSTM", "TestRMSE": ml_original},
        {"series": series_name, "target": "scaled_derivative", "model": "Naive zero-derivative", "TestRMSE": naive["naive_test_rmse_diff"]},
        {"series": series_name, "target": "scaled_derivative", "model": "Twin LSTM", "TestRMSE": ml_diff},
    ])
    return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Run the structural ML model
# MAGIC
# MAGIC This cell trains the model for each selected OAS series. If the notebook is slow, reduce `SERIES_TO_RUN`, `NUM_EXP` or `EPOCHS` in the configuration cell.

# COMMAND ----------

all_ml_results = {}
all_result_tables = []
all_stats_tables = []
all_benchmarks = []

for name in SERIES_TO_RUN:
    print("=" * 100)
    print(f"Running structural ML model for {name}")
    print("=" * 100)

    data = ml_datasets[name]
    results = run_twin_lstm_experiments(data, series_name=name, num_exp=NUM_EXP, epochs=EPOCHS, hidden=HIDDEN)
    all_ml_results[name] = results

    arr_original_df = prepare_results_table(results, target="original")
    arr_diff_df = prepare_results_table(results, target="diff")
    stats_original_df = calculate_statistics_table(results, target="original")
    stats_diff_df = calculate_statistics_table(results, target="diff")

    print(f"\n{name} - Original OAS level results")
    display(arr_original_df)
    print(f"\n{name} - Derivative results")
    display(arr_diff_df)

    print(f"\n{name} - Original OAS level summary")
    display(stats_original_df)
    print(f"\n{name} - Derivative summary")
    display(stats_diff_df)

    bmk = benchmark_table(name, data, results)
    print(f"\n{name} - Structural ML vs naive benchmark")
    display(bmk)

    all_result_tables.append(arr_original_df.assign(series=name, target="original"))
    all_result_tables.append(arr_diff_df.assign(series=name, target="derivative"))
    all_stats_tables.append(stats_original_df.assign(series=name, target="original"))
    all_stats_tables.append(stats_diff_df.assign(series=name, target="derivative"))
    all_benchmarks.append(bmk)

    plot_actual_vs_predicted(data, results, name)
    plot_rmse_means(stats_original_df, stats_diff_df, name)

combined_benchmarks = pd.concat(all_benchmarks, axis=0, ignore_index=True)
print("Combined structural benchmark table:")
display(combined_benchmarks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Compact comparison across OAS segments

# COMMAND ----------

def compact_summary_for_series(name, results):
    stats_original = calculate_statistics_table(results, target="original")
    stats_diff = calculate_statistics_table(results, target="diff")

    return pd.DataFrame([
        {
            "series": name,
            "target": "scaled_oas_level",
            "train_rmse_mean": stats_original.loc["TrainRMSE", "Mean"],
            "test_rmse_mean": stats_original.loc["TestRMSE", "Mean"],
            "step1_rmse_mean": stats_original.loc["Step1", "Mean"],
            f"step{N_STEPS_OUT}_rmse_mean": stats_original.loc[f"Step{N_STEPS_OUT}", "Mean"],
        },
        {
            "series": name,
            "target": "scaled_derivative",
            "train_rmse_mean": stats_diff.loc["TrainRMSE", "Mean"],
            "test_rmse_mean": stats_diff.loc["TestRMSE", "Mean"],
            "step1_rmse_mean": stats_diff.loc["Step1", "Mean"],
            f"step{N_STEPS_OUT}_rmse_mean": stats_diff.loc[f"Step{N_STEPS_OUT}", "Mean"],
        }
    ])

compact_summary = pd.concat(
    [compact_summary_for_series(name, res) for name, res in all_ml_results.items()],
    ignore_index=True
)

display(compact_summary.round(5))

# COMMAND ----------

# ==========================================================
# FAST APPLES-TO-APPLES COMPARISON
# Structural LSTM vs Random Walk vs ARMA vs ARMA-GJR-GARCH
#
# This version avoids refitting ARMA at every rolling step.
# ARMA is fitted once before the test period and then updated
# recursively with observed data, without parameter re-estimation.
# This is much faster and more comparable to the LSTM setup.
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

HORIZONS_TO_COMPARE = [1, 4]

ARMA_ORDER_MAP = {
    "IG": (1, 2),
    "HY": (3, 2),
    "BBB": (1, 2),
    "BB": (3, 2),
    "B": (3, 2)
}

# If it is still slow, start only with ["IG", "HY"]
SERIES_FOR_COMPARISON = list(all_ml_results.keys())

print("Series included:", SERIES_FOR_COMPARISON)


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def rmse_bp(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae_bp(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def diebold_mariano_test(y_true, pred_1, pred_2, power=2, h=1):
    y_true = np.asarray(y_true, dtype=float)
    pred_1 = np.asarray(pred_1, dtype=float)
    pred_2 = np.asarray(pred_2, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(pred_1) & np.isfinite(pred_2)
    y_true = y_true[mask]
    pred_1 = pred_1[mask]
    pred_2 = pred_2[mask]

    e1 = y_true - pred_1
    e2 = y_true - pred_2

    d = np.abs(e1) ** power - np.abs(e2) ** power
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
        cov_sum = 0
        for lag in range(1, h):
            if lag < T:
                cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
                cov_sum += cov
        var_d = gamma0 + 2 * cov_sum
    else:
        var_d = gamma0

    if var_d <= 0 or not np.isfinite(var_d):
        return {
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_loss_diff": d_bar,
            "n_obs": T
        }

    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "DM_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "n_obs": T
    }


def get_scale_factor(series_name):
    row = sg_info.loc[sg_info["series"] == series_name]
    if row.empty:
        raise ValueError(f"No scale factor found in sg_info for {series_name}")
    return float(row["scale_factor"].iloc[0])


def get_lstm_delta_forecasts_bp(series_name, horizon):
    """
    Converts LSTM predicted scaled SG-smoothed levels into h-week changes in bp.
    """

    data = ml_datasets[series_name]
    results = all_ml_results[series_name]

    scale_factor = get_scale_factor(series_name)

    test_df = data["test_df"].copy()
    pred_original_scaled, pred_diff_scaled = results["best_predict_test"]

    h_idx = horizon - 1

    if h_idx >= pred_original_scaled.shape[1]:
        raise ValueError(
            f"Horizon {horizon} is not available. "
            f"N_STEPS_OUT = {pred_original_scaled.shape[1]}"
        )

    last_input_col = f"Input{N_STEPS_IN_ORIGINAL}"

    last_level_scaled = test_df[last_input_col].values
    actual_future_scaled = data["Y_test_original"][:, h_idx]
    pred_future_scaled = pred_original_scaled[:, h_idx]

    # Invert scaling
    last_level_bp = last_level_scaled / scale_factor
    actual_future_bp = actual_future_scaled / scale_factor
    pred_future_bp = pred_future_scaled / scale_factor

    actual_delta_bp = actual_future_bp - last_level_bp
    lstm_delta_bp = pred_future_bp - last_level_bp

    forecast_dates = pd.to_datetime(test_df["forecast_start_date"])

    return pd.DataFrame({
        "forecast_start_date": forecast_dates,
        "actual_delta_bp": actual_delta_bp,
        "LSTM_delta_bp": lstm_delta_bp
    })


def prepare_smoothed_diff_series(series_name):
    """
    Creates the SG-smoothed spread level and first-difference series in bps.
    """

    df = structured_series[series_name].copy()
    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    df = df.sort_values("week_end_date").set_index("week_end_date")

    smoothed_level = df["sg_smoothed_oas_bp"].astype(float)
    smoothed_diff = smoothed_level.diff().dropna()

    # Try to set weekly frequency to avoid statsmodels warnings
    smoothed_diff = smoothed_diff.asfreq("W-FRI")

    return smoothed_level, smoothed_diff.dropna()


def fast_recursive_arma_delta_forecast(series_name, forecast_dates, horizon, order):
    """
    Fast recursive ARMA forecast.

    The model is fitted once before the test period.
    Then it is updated with observed values using append(refit=False).
    This avoids re-estimating ARMA hundreds of times.
    """

    smoothed_level, smoothed_diff = prepare_smoothed_diff_series(series_name)

    forecast_dates = pd.to_datetime(forecast_dates)
    forecast_dates = pd.Series(forecast_dates).sort_values().reset_index(drop=True)

    first_date = forecast_dates.iloc[0]

    train_series = smoothed_diff.loc[smoothed_diff.index < first_date].dropna()

    if len(train_series) < 80:
        raise ValueError(f"Not enough training observations for {series_name}")

    p, q = order

    base_model = ARIMA(
        train_series,
        order=(p, 0, q),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    base_res = base_model.fit(method_kwargs={"maxiter": 100})

    preds = []
    res_current = base_res
    last_update_date = train_series.index.max()

    for i, d in enumerate(forecast_dates):
        # Add realized observations between the last update and current forecast origin
        new_obs = smoothed_diff.loc[
            (smoothed_diff.index > last_update_date) &
            (smoothed_diff.index < d)
        ].dropna()

        if len(new_obs) > 0:
            try:
                res_current = res_current.append(new_obs, refit=False)
                last_update_date = new_obs.index.max()
            except Exception:
                pass

        try:
            fcst = res_current.forecast(steps=horizon)
            pred_delta = float(np.sum(fcst.values[:horizon]))
        except Exception:
            pred_delta = np.nan

        preds.append(pred_delta)

    return np.asarray(preds, dtype=float)


# ----------------------------------------------------------
# Main comparison loop
# ----------------------------------------------------------

all_comparison_rows = []
all_dm_rows = []
all_forecast_tables = {}

for series_name in SERIES_FOR_COMPARISON:
    print("=" * 100)
    print(f"Running fast comparison for {series_name}")
    print("=" * 100)

    order = ARMA_ORDER_MAP.get(series_name, (1, 2))
    print(f"Using ARMA order: {order}")

    all_forecast_tables[series_name] = {}

    for h in HORIZONS_TO_COMPARE:
        print(f"\n{series_name} - horizon {h} week(s)")

        # LSTM forecast target in bp
        comp = get_lstm_delta_forecasts_bp(series_name, horizon=h)

        # Random Walk forecast for h-week change
        comp["RW_delta_bp"] = 0.0

        # Fast ARMA forecast
        comp["ARMA_delta_bp"] = fast_recursive_arma_delta_forecast(
            series_name=series_name,
            forecast_dates=comp["forecast_start_date"],
            horizon=h,
            order=order
        )

        # In the two-step setup, GJR-GARCH does not change the point forecast
        comp["ARMA_GJR_delta_bp"] = comp["ARMA_delta_bp"].copy()

        comp_clean = comp.dropna(subset=[
            "actual_delta_bp",
            "RW_delta_bp",
            "ARMA_delta_bp",
            "ARMA_GJR_delta_bp",
            "LSTM_delta_bp"
        ]).copy()

        all_forecast_tables[series_name][h] = comp_clean

        rw_rmse = rmse_bp(comp_clean["actual_delta_bp"], comp_clean["RW_delta_bp"])

        model_cols = {
            "Random Walk": "RW_delta_bp",
            f"ARMA{order}": "ARMA_delta_bp",
            f"ARMA{order}-GJR-GARCH(1,1)": "ARMA_GJR_delta_bp",
            "Structural LSTM": "LSTM_delta_bp"
        }

        for model_name, col in model_cols.items():
            model_rmse = rmse_bp(comp_clean["actual_delta_bp"], comp_clean[col])
            model_mae = mae_bp(comp_clean["actual_delta_bp"], comp_clean[col])

            all_comparison_rows.append({
                "series": series_name,
                "horizon_weeks": h,
                "model": model_name,
                "RMSE_bp": model_rmse,
                "MAE_bp": model_mae,
                "RMSE_improvement_vs_RW_pct": 100 * (rw_rmse - model_rmse) / rw_rmse,
                "n_obs": len(comp_clean)
            })

        # DM tests
        dm_comparisons = [
            ("RW vs ARMA", "RW_delta_bp", "ARMA_delta_bp"),
            ("RW vs ARMA-GJR", "RW_delta_bp", "ARMA_GJR_delta_bp"),
            ("RW vs LSTM", "RW_delta_bp", "LSTM_delta_bp"),
            ("ARMA vs LSTM", "ARMA_delta_bp", "LSTM_delta_bp")
        ]

        for comparison_name, col_1, col_2 in dm_comparisons:
            dm = diebold_mariano_test(
                y_true=comp_clean["actual_delta_bp"],
                pred_1=comp_clean[col_1],
                pred_2=comp_clean[col_2],
                power=2,
                h=h
            )

            all_dm_rows.append({
                "series": series_name,
                "horizon_weeks": h,
                "comparison": comparison_name,
                **dm
            })


# ----------------------------------------------------------
# Final output tables
# ----------------------------------------------------------

model_comparison_bp = pd.DataFrame(all_comparison_rows)
dm_comparison_bp = pd.DataFrame(all_dm_rows)

model_comparison_bp = model_comparison_bp.sort_values(
    ["series", "horizon_weeks", "RMSE_bp"]
).reset_index(drop=True)

dm_comparison_bp = dm_comparison_bp.sort_values(
    ["series", "horizon_weeks", "comparison"]
).reset_index(drop=True)

print("\nMODEL COMPARISON IN BASIS POINTS")
display(model_comparison_bp.round(4))

print("\nDIEBOLD-MARIANO TESTS")
display(dm_comparison_bp.round(4))

winner_table = (
    model_comparison_bp
    .sort_values(["series", "horizon_weeks", "RMSE_bp"])
    .groupby(["series", "horizon_weeks"])
    .first()
    .reset_index()
    [["series", "horizon_weeks", "model", "RMSE_bp", "MAE_bp", "RMSE_improvement_vs_RW_pct"]]
)

print("\nWINNER BY SERIES AND HORIZON")
display(winner_table.round(4))



# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Notes for interpretation in the TFG
# MAGIC
# MAGIC Use this model carefully. It is useful as an ML extension, but it should not replace the baseline ARMA/GARCH forecasting exercise.
# MAGIC
# MAGIC A reasonable interpretation framework:
# MAGIC
# MAGIC - If the LSTM improves clearly over the structural naive benchmark, this suggests that nonlinear patterns in recent OAS levels and derivative signals may contain additional information.
# MAGIC - If it does not improve, that is still useful: it supports the idea that corporate spread changes are hard to forecast even with more flexible ML structures.
# MAGIC - The derivative branch should be interpreted as a smoothed momentum / slope signal, not as a structural economic variable.
# MAGIC - For the written TFG, this belongs in the extension section, not in the core methodology.

# COMMAND ----------

# MAGIC %md
# MAGIC BEST ML MODEL ALTERNATIVES

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ==========================================================
# DIRECT ML EXTENSION FOR CORPORATE OAS FORECASTING
# ==========================================================
# Objective:
# Forecast future changes in OAS directly:
#   target_h = OAS_{t+h} - OAS_t
#
# This replaces the SG-filter / Diff-LSTM setup with a model
# that is more specific to the corporate spread problem:
# - lagged spread changes
# - spread levels
# - rating gaps
# - rolling volatility
# - macro-financial predictors
#
# Models compared:
# - Random Walk / no-change
# - ARMA
# - ARMA-GJR-GARCH point forecast
# - Elastic Net
# - Random Forest
# - Gradient Boosting
# - HistGradientBoosting
# - MLP
# - XGBoost, if installed
#
# Notes:
# In the two-step ARMA-GJR-GARCH setup, GJR-GARCH models
# conditional volatility, but the point forecast of the mean
# is the ARMA forecast. Therefore, ARMA and ARMA-GJR have
# the same point forecast in this comparison.
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.arima.model import ARIMA


# ==========================================================
# 1) CONFIGURATION
# ==========================================================

CATALOG = "tfg_data"
SCHEMA = "original_data"

TABLES = {
    "IG": "ig_aggregate_oas",
    "HY": "hy_aggregate_oas",
    "BUCKETS": "bbb_bb_b_extensiondata",
    "SP500": "sp500",
    "VIX": "vix",
    "TREASURY_10Y": "treasury_10y",
    "TREASURY_2Y": "treasury_2y"
}

WEEK_RULE = "W-FRI"
TRAIN_RATIO = 0.80
HORIZONS = [1, 4]

CREDIT_SERIES = ["IG", "HY", "BBB", "BB", "B"]

# Use the orders you selected earlier
ARMA_ORDER_MAP = {
    "IG": (1, 2),
    "HY": (3, 2),
    "BBB": (1, 2),
    "BB": (3, 2),
    "B": (3, 2)
}

MAX_LAG_DIFF = 8
MAX_LAG_LEVEL = 4
ROLLING_WINDOWS = [4, 13, 26]

RUN_SERIES = ["IG", "HY", "BBB", "BB", "B"]
# For a quick first run, use:
# RUN_SERIES = ["IG", "HY"]


# ==========================================================
# 2) HELPER FUNCTIONS: LOADING AND COLUMN DETECTION
# ==========================================================

def spark_table_to_pandas(table_name):
    full_name = f"{CATALOG}.{SCHEMA}.{table_name}"
    return spark.table(full_name).toPandas()


def normalize_name(x):
    return str(x).lower().replace(" ", "").replace("_", "").replace("-", "")


def find_date_col(df):
    candidates = ["observation_date", "date", "datetime", "period"]
    norm_map = {normalize_name(c): c for c in df.columns}

    for c in candidates:
        if normalize_name(c) in norm_map:
            return norm_map[normalize_name(c)]

    # fallback: first column that looks like date
    for c in df.columns:
        if "date" in normalize_name(c):
            return c

    raise ValueError("No date column found.")


def find_value_col(df, candidates=None, exclude_cols=None):
    if candidates is None:
        candidates = []

    if exclude_cols is None:
        exclude_cols = []

    exclude_cols = set(exclude_cols)
    norm_map = {normalize_name(c): c for c in df.columns}

    # exact / normalized match first
    for cand in candidates:
        n = normalize_name(cand)
        if n in norm_map and norm_map[n] not in exclude_cols:
            return norm_map[n]

    # contains match second
    for cand in candidates:
        n = normalize_name(cand)
        for col in df.columns:
            if col in exclude_cols:
                continue
            if n in normalize_name(col):
                return col

    # fallback: first numeric column not excluded
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric value column found.")

    return numeric_cols[0]


def make_raw_series(df, series_name, value_candidates):
    date_col = find_date_col(df)
    value_col = find_value_col(df, candidates=value_candidates, exclude_cols=[date_col])

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", series_name]

    out["date"] = pd.to_datetime(out["date"])
    out[series_name] = pd.to_numeric(out[series_name], errors="coerce")

    out = (
        out.sort_values("date")
           .drop_duplicates(subset=["date"])
           .set_index("date")
    )

    return out


def make_weekly_series(raw_series, method="mean"):
    """
    Creates a weekly series from daily data.
    For credit spreads we use weekly averages.
    For market variables like S&P 500, VIX and yields we usually use last weekly observation.
    """

    daily = raw_series.asfreq("D")
    daily = daily.interpolate(method="time", limit_direction="both").ffill().bfill()

    if method == "mean":
        weekly = daily.resample(WEEK_RULE).mean()
    elif method == "last":
        weekly = daily.resample(WEEK_RULE).last()
    else:
        raise ValueError("method must be 'mean' or 'last'")

    return weekly


def convert_to_bp_if_needed(s):
    """
    FRED OAS series are usually in percentage points.
    Example: 3.25 means 3.25%, which equals 325 bps.
    If the median is small, we convert to bps.
    """
    med = s.dropna().median()
    if med < 50:
        return s * 100
    return s


# ==========================================================
# 3) LOAD DATA FROM DATABRICKS TABLES
# ==========================================================

print("Loading raw tables from Databricks...")

ig_raw_table = spark_table_to_pandas(TABLES["IG"])
hy_raw_table = spark_table_to_pandas(TABLES["HY"])
bucket_raw_table = spark_table_to_pandas(TABLES["BUCKETS"])

sp500_raw_table = spark_table_to_pandas(TABLES["SP500"])
vix_raw_table = spark_table_to_pandas(TABLES["VIX"])
t10_raw_table = spark_table_to_pandas(TABLES["TREASURY_10Y"])
t2_raw_table = spark_table_to_pandas(TABLES["TREASURY_2Y"])

# Credit series candidates
ig_raw = make_raw_series(
    ig_raw_table,
    "IG",
    ["IG", "BAMLC0A0CM", "investmentgrade", "investment_grade", "value"]
)

hy_raw = make_raw_series(
    hy_raw_table,
    "HY",
    ["HY", "BAMLH0A0HYM2", "highyield", "high_yield", "value"]
)

bbb_raw = make_raw_series(
    bucket_raw_table,
    "BBB",
    ["BBB", "BAMLC0A4CBBB", "bbb_oas"]
)

bb_raw = make_raw_series(
    bucket_raw_table,
    "BB",
    ["BB", "BAMLH0A1HYBB", "bb_oas"]
)

b_raw = make_raw_series(
    bucket_raw_table,
    "B",
    ["B", "BAMLH0A2HYB", "singleb", "single_b", "b_oas"]
)

# Macro-financial series candidates
sp500_raw = make_raw_series(
    sp500_raw_table,
    "SP500",
    ["SP500", "sp500", "s&p500", "close", "value"]
)

vix_raw = make_raw_series(
    vix_raw_table,
    "VIX",
    ["VIX", "VIXCLS", "vixcls", "value"]
)

t10_raw = make_raw_series(
    t10_raw_table,
    "T10Y",
    ["DGS10", "10Y", "treasury_10y", "t10y", "value"]
)

t2_raw = make_raw_series(
    t2_raw_table,
    "T2Y",
    ["DGS2", "2Y", "treasury_2y", "t2y", "value"]
)

print("Raw data loaded successfully.")


# ==========================================================
# 4) WEEKLY TRANSFORMATION AND COMBINED DATASET
# ==========================================================

credit_weekly = pd.concat(
    [
        make_weekly_series(ig_raw, method="mean"),
        make_weekly_series(hy_raw, method="mean"),
        make_weekly_series(bbb_raw, method="mean"),
        make_weekly_series(bb_raw, method="mean"),
        make_weekly_series(b_raw, method="mean")
    ],
    axis=1
)

macro_weekly = pd.concat(
    [
        make_weekly_series(sp500_raw, method="last"),
        make_weekly_series(vix_raw, method="last"),
        make_weekly_series(t10_raw, method="last"),
        make_weekly_series(t2_raw, method="last")
    ],
    axis=1
)

# Convert credit spreads to bps if needed
for s in CREDIT_SERIES:
    credit_weekly[f"{s}_bp"] = convert_to_bp_if_needed(credit_weekly[s])

credit_bp = credit_weekly[[f"{s}_bp" for s in CREDIT_SERIES]].copy()

data = pd.concat([credit_bp, macro_weekly], axis=1).sort_index()
data = data.dropna(how="all")

print("Combined weekly dataset:")
display(data.head())
print(data.shape)


# ==========================================================
# 5) FEATURE ENGINEERING
# ==========================================================

features = data.copy()

# --------------------------
# Credit spread changes
# --------------------------
for s in CREDIT_SERIES:
    features[f"{s}_diff"] = features[f"{s}_bp"].diff()

# --------------------------
# Rating gaps and relative credit stress
# --------------------------
features["HY_IG_gap"] = features["HY_bp"] - features["IG_bp"]
features["BBB_IG_gap"] = features["BBB_bp"] - features["IG_bp"]
features["BB_BBB_gap"] = features["BB_bp"] - features["BBB_bp"]
features["B_BB_gap"] = features["B_bp"] - features["BB_bp"]
features["B_BBB_gap"] = features["B_bp"] - features["BBB_bp"]
features["HY_BBB_gap"] = features["HY_bp"] - features["BBB_bp"]

for gap in ["HY_IG_gap", "BBB_IG_gap", "BB_BBB_gap", "B_BB_gap", "B_BBB_gap", "HY_BBB_gap"]:
    features[f"{gap}_diff"] = features[gap].diff()

# --------------------------
# Macro-financial predictors
# --------------------------
features["SP500_ret"] = np.log(features["SP500"]).diff()
features["VIX_change"] = features["VIX"].diff()
features["T10Y_change"] = features["T10Y"].diff()
features["T2Y_change"] = features["T2Y"].diff()
features["Treasury_slope_10y_2y"] = features["T10Y"] - features["T2Y"]
features["Treasury_slope_change"] = features["Treasury_slope_10y_2y"].diff()

# --------------------------
# Lagged levels and changes
# --------------------------
for s in CREDIT_SERIES:
    for lag in range(1, MAX_LAG_LEVEL + 1):
        features[f"{s}_bp_lag{lag}"] = features[f"{s}_bp"].shift(lag)

    for lag in range(1, MAX_LAG_DIFF + 1):
        features[f"{s}_diff_lag{lag}"] = features[f"{s}_diff"].shift(lag)

# --------------------------
# Rolling volatility and momentum
# --------------------------
for s in CREDIT_SERIES:
    for w in ROLLING_WINDOWS:
        features[f"{s}_diff_roll_mean_{w}"] = features[f"{s}_diff"].rolling(w).mean()
        features[f"{s}_diff_roll_vol_{w}"] = features[f"{s}_diff"].rolling(w).std()
        features[f"{s}_level_roll_mean_{w}"] = features[f"{s}_bp"].rolling(w).mean()

    features[f"{s}_level_z_26"] = (
        (features[f"{s}_bp"] - features[f"{s}_bp"].rolling(26).mean()) /
        features[f"{s}_bp"].rolling(26).std()
    )

# --------------------------
# Macro lags
# --------------------------
macro_base_cols = [
    "SP500_ret",
    "VIX",
    "VIX_change",
    "T10Y",
    "T2Y",
    "T10Y_change",
    "T2Y_change",
    "Treasury_slope_10y_2y",
    "Treasury_slope_change"
]

for col in macro_base_cols:
    for lag in range(1, 5):
        features[f"{col}_lag{lag}"] = features[col].shift(lag)

# Remove initial rows with too many missing feature values later inside model function
print("Feature dataset created.")
print("Number of columns:", len(features.columns))


# ==========================================================
# 6) EVALUATION FUNCTIONS
# ==========================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return np.mean(np.sign(y_true) == np.sign(y_pred))


def diebold_mariano_test(y_true, pred_1, pred_2, power=2, h=1):
    """
    H0: equal predictive accuracy.

    mean_loss_diff = loss(model 1) - loss(model 2)

    If mean_loss_diff > 0, model 2 has lower average loss.
    """

    y_true = np.asarray(y_true, dtype=float)
    pred_1 = np.asarray(pred_1, dtype=float)
    pred_2 = np.asarray(pred_2, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(pred_1) & np.isfinite(pred_2)
    y_true = y_true[mask]
    pred_1 = pred_1[mask]
    pred_2 = pred_2[mask]

    e1 = y_true - pred_1
    e2 = y_true - pred_2

    d = np.abs(e1) ** power - np.abs(e2) ** power
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
        cov_sum = 0
        for lag in range(1, h):
            if lag < T:
                cov = np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
                cov_sum += cov
        var_d = gamma0 + 2 * cov_sum
    else:
        var_d = gamma0

    if var_d <= 0 or not np.isfinite(var_d):
        return {
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_loss_diff": d_bar,
            "n_obs": T
        }

    dm_stat = d_bar / np.sqrt(var_d / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "DM_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "n_obs": T
    }


# ==========================================================
# 7) ARMA BENCHMARK FORECAST
# ==========================================================

def fast_recursive_arma_forecast(diff_series, forecast_dates, horizon, order):
    """
    Fit ARMA once before the test period.
    Then recursively append realized observations without refitting.
    Forecast is h-week cumulative change.
    """

    diff_series = diff_series.dropna().copy()
    diff_series = diff_series.asfreq(WEEK_RULE).dropna()

    forecast_dates = pd.to_datetime(pd.Series(forecast_dates)).sort_values().reset_index(drop=True)

    first_date = forecast_dates.iloc[0]

    train_series = diff_series.loc[diff_series.index < first_date].dropna()

    if len(train_series) < 80:
        return np.repeat(np.nan, len(forecast_dates))

    p, q = order

    try:
        base_model = ARIMA(
            train_series,
            order=(p, 0, q),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        res_current = base_model.fit(method_kwargs={"maxiter": 100})

    except Exception as e:
        print(f"Initial ARMA fit failed for order {order}: {e}")
        return np.repeat(np.nan, len(forecast_dates))

    preds = []

    last_update_date = train_series.index.max()

    for d in forecast_dates:
        # At forecast origin d, the current spread and current weekly change are known.
        new_obs = diff_series.loc[
            (diff_series.index > last_update_date) &
            (diff_series.index <= d)
        ].dropna()

        if len(new_obs) > 0:
            try:
                res_current = res_current.append(new_obs, refit=False)
                last_update_date = new_obs.index.max()
            except Exception:
                pass

        try:
            fcst = res_current.forecast(steps=horizon)
            pred_delta = float(np.sum(fcst.values[:horizon]))
        except Exception:
            pred_delta = np.nan

        preds.append(pred_delta)

    return np.asarray(preds, dtype=float)


# ==========================================================
# 8) ML MODELS
# ==========================================================

def build_ml_models():
    tscv = TimeSeriesSplit(n_splits=5)

    models = {}

    models["Elastic Net"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-4, 1, 30),
            cv=tscv,
            max_iter=10000,
            random_state=42
        ))
    ])

    models["Random Forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    models["Gradient Boosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=10,
            random_state=42
        ))
    ])

    models["HistGradientBoosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42
        ))
    ])

    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        ))
    ])

    # Optional XGBoost if installed
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            ))
        ])

        print("XGBoost detected and added.")

    except Exception:
        print("XGBoost not installed. Skipping XGBoost.")

    return models


# ==========================================================
# 9) DATASET BUILDER FOR EACH SERIES AND HORIZON
# ==========================================================

def build_supervised_dataset(series_name, horizon):
    """
    Creates X, y for target:
        OAS_{t+h} - OAS_t

    The split avoids leakage:
    train origins must have target dates before the test start date.
    """

    df = features.copy()

    level_col = f"{series_name}_bp"

    df["target"] = df[level_col].shift(-horizon) - df[level_col]
    df["target_date"] = df.index.to_series().shift(-horizon)

    # Candidate feature columns:
    # remove future target and raw current target objects
    exclude = {"target", "target_date"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Keep only rows with target available
    df_model = df.dropna(subset=["target", "target_date"]).copy()

    # Need enough non-missing features, but imputer will handle partial missingness
    df_model = df_model.dropna(subset=[level_col]).copy()

    all_dates = df_model.index.sort_values()
    split_idx = int(len(all_dates) * TRAIN_RATIO)
    split_start_date = all_dates[split_idx]

    # Avoid leakage:
    # training target must be fully before the test period starts
    train_mask = (df_model.index < split_start_date) & (df_model["target_date"] < split_start_date)
    test_mask = df_model.index >= split_start_date

    train_df = df_model.loc[train_mask].copy()
    test_df = df_model.loc[test_mask].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    return {
        "series": series_name,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "split_start_date": split_start_date
    }


# ==========================================================
# 10) MAIN LOOP: TRAIN AND COMPARE MODELS
# ==========================================================

all_metric_rows = []
all_dm_rows = []
all_prediction_tables = {}

for series_name in RUN_SERIES:
    print("=" * 100)
    print(f"Running direct ML extension for {series_name}")
    print("=" * 100)

    all_prediction_tables[series_name] = {}

    for h in HORIZONS:
        print(f"\nSeries: {series_name} | Horizon: {h} week(s)")

        ds = build_supervised_dataset(series_name, h)

        X_train = ds["X_train"]
        y_train = ds["y_train"]
        X_test = ds["X_test"]
        y_test = ds["y_test"]
        test_dates = X_test.index

        print("Train rows:", len(X_train))
        print("Test rows:", len(X_test))
        print("Test starts:", ds["split_start_date"].date())

        # ------------------------------------------
        # Benchmarks
        # ------------------------------------------
        preds = pd.DataFrame(index=test_dates)
        preds["actual"] = y_test.values

        # Random Walk / no-change
        preds["Random Walk"] = 0.0

        # ARMA and ARMA-GJR point forecast
        diff_series = features[f"{series_name}_diff"].dropna()
        arma_order = ARMA_ORDER_MAP.get(series_name, (1, 2))

        arma_pred = fast_recursive_arma_forecast(
            diff_series=diff_series,
            forecast_dates=test_dates,
            horizon=h,
            order=arma_order
        )

        preds[f"ARMA{arma_order}"] = arma_pred
        preds[f"ARMA{arma_order}-GJR-GARCH(1,1)"] = arma_pred.copy()

        # ------------------------------------------
        # ML models
        # ------------------------------------------
        ml_models = build_ml_models()

        for model_name, model in ml_models.items():
            print(f"Fitting {model_name}...")

            try:
                model.fit(X_train, y_train)
                preds[model_name] = model.predict(X_test)
            except Exception as e:
                print(f"{model_name} failed for {series_name}, h={h}: {e}")
                preds[model_name] = np.nan

        # Clean valid rows
        preds_clean = preds.dropna().copy()
        all_prediction_tables[series_name][h] = preds_clean

        # ------------------------------------------
        # Metrics
        # ------------------------------------------
        model_cols = [c for c in preds_clean.columns if c != "actual"]

        rw_rmse = rmse(preds_clean["actual"], preds_clean["Random Walk"])

        for col in model_cols:
            row = {
                "series": series_name,
                "horizon_weeks": h,
                "model": col,
                "RMSE_bp": rmse(preds_clean["actual"], preds_clean[col]),
                "MAE_bp": mae(preds_clean["actual"], preds_clean[col]),
                "Directional_Accuracy": directional_accuracy(preds_clean["actual"], preds_clean[col]),
                "RMSE_improvement_vs_RW_pct": 100 * (rw_rmse - rmse(preds_clean["actual"], preds_clean[col])) / rw_rmse,
                "n_train": len(X_train),
                "n_test": len(preds_clean)
            }

            all_metric_rows.append(row)

        # ------------------------------------------
        # Diebold-Mariano tests
        # Compare all models against Random Walk and ARMA
        # ------------------------------------------
        arma_col = f"ARMA{arma_order}"

        for col in model_cols:
            if col == "Random Walk":
                continue

            dm_vs_rw = diebold_mariano_test(
                y_true=preds_clean["actual"],
                pred_1=preds_clean["Random Walk"],
                pred_2=preds_clean[col],
                power=2,
                h=h
            )

            all_dm_rows.append({
                "series": series_name,
                "horizon_weeks": h,
                "comparison": f"Random Walk vs {col}",
                **dm_vs_rw
            })

            if col != arma_col:
                dm_vs_arma = diebold_mariano_test(
                    y_true=preds_clean["actual"],
                    pred_1=preds_clean[arma_col],
                    pred_2=preds_clean[col],
                    power=2,
                    h=h
                )

                all_dm_rows.append({
                    "series": series_name,
                    "horizon_weeks": h,
                    "comparison": f"{arma_col} vs {col}",
                    **dm_vs_arma
                })


# ==========================================================
# 11) FINAL TABLES
# ==========================================================

direct_ml_comparison = pd.DataFrame(all_metric_rows)
direct_ml_dm_tests = pd.DataFrame(all_dm_rows)

direct_ml_comparison = (
    direct_ml_comparison
    .sort_values(["series", "horizon_weeks", "RMSE_bp"])
    .reset_index(drop=True)
)

direct_ml_dm_tests = (
    direct_ml_dm_tests
    .sort_values(["series", "horizon_weeks", "comparison"])
    .reset_index(drop=True)
)

winner_direct_ml = (
    direct_ml_comparison
    .sort_values(["series", "horizon_weeks", "RMSE_bp"])
    .groupby(["series", "horizon_weeks"])
    .first()
    .reset_index()
    [["series", "horizon_weeks", "model", "RMSE_bp", "MAE_bp", "Directional_Accuracy", "RMSE_improvement_vs_RW_pct"]]
)

print("\nDIRECT ML MODEL COMPARISON")
display(direct_ml_comparison.round(4))

print("\nDIEBOLD-MARIANO TESTS")
display(direct_ml_dm_tests.round(4))

print("\nWINNER BY SERIES AND HORIZON")
display(winner_direct_ml.round(4))


# ==========================================================
# 12) OPTIONAL EXPORT
# ==========================================================

direct_ml_comparison.to_csv("direct_ml_model_comparison.csv", index=False)
direct_ml_dm_tests.to_csv("direct_ml_dm_tests.csv", index=False)
winner_direct_ml.to_csv("direct_ml_winners.csv", index=False)

print("\nExported:")
print("- direct_ml_model_comparison.csv")
print("- direct_ml_dm_tests.csv")
print("- direct_ml_winners.csv")

# COMMAND ----------

# ==========================================================
# ML WINNER VISUAL DASHBOARDS - 1 WEEK ONLY
# ==========================================================
# This box plots only the winning ML models at the 1-week horizon.
# ARMA-type winners and Random Walk winners are excluded.
#
# Expected existing objects from the previous notebook section:
# - winner_direct_ml
# - direct_ml_comparison
# - all_prediction_tables
# - build_supervised_dataset()
# - build_ml_models()
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ----------------------------------------------------------
# 1) Configuration
# ----------------------------------------------------------




# ----------------------------------------------------------
# 2) Helper functions
# ----------------------------------------------------------

def is_ml_model(model_name):
    """
    Keep only proper ML models.
    Exclude Random Walk, ARMA and GARCH-type models.
    """
    name = str(model_name).lower()

    excluded_terms = [
        "arma",
        "arima",
        "garch",
        "random walk",
        "rw",
        "selected arma"
    ]

    return not any(term in name for term in excluded_terms)


def directional_accuracy_local(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return np.mean(np.sign(y_true) == np.sign(y_pred))


def get_metric_row(series_name, horizon, model_name):
    """
    Gets metrics from direct_ml_comparison if available.
    Otherwise computes them from all_prediction_tables.
    """
    if "direct_ml_comparison" in globals():
        tmp = direct_ml_comparison.copy()
        tmp["model_clean"] = tmp["model"].astype(str).str.strip()

        row = tmp[
            (tmp["series"] == series_name) &
            (tmp["horizon_weeks"] == horizon) &
            (tmp["model_clean"] == str(model_name).strip())
        ]

        if len(row) > 0:
            return row.iloc[0].to_dict()

    # Fallback
    preds = all_prediction_tables[series_name][horizon].copy()
    y_true = preds["actual"]
    y_pred = preds[model_name]

    rw_rmse = np.sqrt(mean_squared_error(y_true, preds["Random Walk"]))
    model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "series": series_name,
        "horizon_weeks": horizon,
        "model": model_name,
        "RMSE_bp": model_rmse,
        "MAE_bp": mean_absolute_error(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy_local(y_true, y_pred),
        "RMSE_improvement_vs_RW_pct": 100 * (rw_rmse - model_rmse) / rw_rmse,
        "n_test": len(preds)
    }


def fit_winner_model_again(series_name, horizon, model_name):
    """
    Refit the winner model so we can compute permutation feature importance.
    This is necessary because the previous comparison section usually stores
    predictions but not fitted model objects.
    """
    ds = build_supervised_dataset(series_name, horizon)

    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_test = ds["X_test"]
    y_test = ds["y_test"]

    model_dict = build_ml_models()

    if model_name not in model_dict:
        raise ValueError(
            f"Model '{model_name}' not found in build_ml_models(). "
            f"Available models: {list(model_dict.keys())}"
        )

    model = model_dict[model_name]
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test


def compute_top_permutation_importance(model, X_test, y_test, top_n=12):
    """
    Computes permutation importance using negative MSE.
    Positive values mean the feature is useful for forecast accuracy.
    """
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=42,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean,
        "importance_std": perm.importances_std
    })

    imp = imp.sort_values("importance", ascending=False).head(top_n)

    return imp


def plot_ml_winner_dashboard(series_name, horizon, model_name):
    """
    Creates one complete ML-style dashboard for a given winning model.
    """

    print("=" * 100)
    print(f"Plotting ML winner: {series_name} | {horizon}-week | {model_name}")
    print("=" * 100)

    # ------------------------------------------------------
    # Predictions already computed in the model comparison section
    # ------------------------------------------------------
    preds = all_prediction_tables[series_name][horizon].copy()
    preds = preds.dropna(subset=["actual", "Random Walk", model_name]).copy()

    y_true = preds["actual"]
    y_pred = preds[model_name]
    y_rw = preds["Random Walk"]

    residuals = y_true - y_pred

    squared_loss_rw = (y_true - y_rw) ** 2
    squared_loss_model = (y_true - y_pred) ** 2
    cumulative_loss_advantage = (squared_loss_rw - squared_loss_model).cumsum()

    metrics = get_metric_row(series_name, horizon, model_name)

    # ------------------------------------------------------
    # Refit model only for feature importance
    # ------------------------------------------------------
    try:
        fitted_model, X_train, y_train, X_test, y_test = fit_winner_model_again(
            series_name=series_name,
            horizon=horizon,
            model_name=model_name
        )

        feature_importance = compute_top_permutation_importance(
            model=fitted_model,
            X_test=X_test,
            y_test=y_test,
            top_n=12
        )

    except Exception as e:
        print(f"Feature importance failed for {series_name} - {model_name}: {e}")
        feature_importance = pd.DataFrame({
            "feature": [],
            "importance": []
        })

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"{series_name} - {model_name} winner dashboard | {horizon}-week ΔOAS forecast",
        fontsize=18,
        fontweight="bold"
    )

    grid = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1])

    ax1 = fig.add_subplot(grid[0, :])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])
    ax4 = fig.add_subplot(grid[2, 0])
    ax5 = fig.add_subplot(grid[2, 1])

    # ------------------------------------------------------
    # Panel 1: actual vs predicted over time
    # ------------------------------------------------------
    ax1.plot(preds.index, y_true, label="Actual ΔOAS", linewidth=1.6)
    ax1.plot(preds.index, y_pred, label=f"{model_name} forecast", linewidth=1.8)
    ax1.axhline(0, linewidth=1, linestyle="--", alpha=0.7)

    ax1.set_title("Actual vs predicted weekly spread changes")
    ax1.set_ylabel("ΔOAS forecast / actual (bps)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)

    metric_text = (
        f"RMSE: {metrics.get('RMSE_bp', np.nan):.3f} bps\n"
        f"MAE: {metrics.get('MAE_bp', np.nan):.3f} bps\n"
        f"Directional accuracy: {metrics.get('Directional_Accuracy', np.nan):.2%}\n"
        f"RMSE improvement vs RW: {metrics.get('RMSE_improvement_vs_RW_pct', np.nan):.2f}%"
    )

    ax1.text(
        0.015, 0.95,
        metric_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    # ------------------------------------------------------
    # Panel 2: predicted vs actual scatter
    # ------------------------------------------------------
    ax2.scatter(y_true, y_pred, alpha=0.65)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax2.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1.2)

    ax2.set_title("Predicted vs actual")
    ax2.set_xlabel("Actual ΔOAS (bps)")
    ax2.set_ylabel("Predicted ΔOAS (bps)")
    ax2.grid(True, alpha=0.25)

    # ------------------------------------------------------
    # Panel 3: residuals over time
    # ------------------------------------------------------
    ax3.plot(preds.index, residuals, linewidth=1.3)
    ax3.axhline(0, linestyle="--", linewidth=1)

    ax3.set_title("Forecast residuals over time")
    ax3.set_ylabel("Actual - predicted (bps)")
    ax3.grid(True, alpha=0.25)

    # ------------------------------------------------------
    # Panel 4: cumulative squared loss advantage vs Random Walk
    # ------------------------------------------------------
    ax4.plot(preds.index, cumulative_loss_advantage, linewidth=1.8)
    ax4.axhline(0, linestyle="--", linewidth=1)

    ax4.set_title("Cumulative squared-loss advantage vs Random Walk")
    ax4.set_ylabel("Cumulative loss difference")
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.25)

    ax4.text(
        0.015, 0.95,
        "Upward = model beats RW\nDownward = RW beats model",
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.12)
    )

    # ------------------------------------------------------
    # Panel 5: permutation feature importance
    # ------------------------------------------------------
    if len(feature_importance) > 0:
        fi = feature_importance.sort_values("importance", ascending=True)

        ax5.barh(fi["feature"], fi["importance"])
        ax5.set_title("Top permutation feature importance")
        ax5.set_xlabel("Increase in forecast loss when permuted")

    else:
        ax5.text(
            0.5, 0.5,
            "Feature importance not available",
            ha="center",
            va="center",
            transform=ax5.transAxes
        )
        ax5.set_title("Feature importance")

    ax5.grid(True, axis="x", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------
    # Save and show
    # ------------------------------------------------------
    if SAVE_FIGURES:
        safe_model_name = (
            str(model_name)
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace("/", "_")
        )
    plt.show()


# ----------------------------------------------------------
# 3) Select only 1-week ML winners
# ----------------------------------------------------------

winners_1w_ml = winner_direct_ml.copy()
winners_1w_ml["model"] = winners_1w_ml["model"].astype(str).str.strip()

winners_1w_ml = winners_1w_ml[
    (winners_1w_ml["horizon_weeks"] == HORIZON_TO_PLOT) &
    (winners_1w_ml["model"].apply(is_ml_model))
].copy()

print("1-week ML winners to plot:")
display(winners_1w_ml)


# ----------------------------------------------------------
# 4) Plot each ML winner
# ----------------------------------------------------------

for _, row in winners_1w_ml.iterrows():
    plot_ml_winner_dashboard(
        series_name=row["series"],
        horizon=int(row["horizon_weeks"]),
        model_name=row["model"]
    )


# COMMAND ----------

# ==========================================================
# IMPACTFUL ML WINNER PLOT - 1 WEEK ONLY
# Directional success + cumulative advantage vs Random Walk
# No figure saving
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

HORIZON_TO_PLOT = 1


def is_ml_model(model_name):
    name = str(model_name).lower()
    excluded_terms = ["arma", "arima", "garch", "random walk", "rw", "selected arma"]
    return not any(term in name for term in excluded_terms)


def directional_accuracy_local(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))


def plot_impact_winner(series_name, horizon, model_name):
    preds = all_prediction_tables[series_name][horizon].copy()
    preds = preds.dropna(subset=["actual", "Random Walk", model_name]).copy()

    y_true = preds["actual"]
    y_pred = preds[model_name]
    y_rw = preds["Random Walk"]

    correct_direction = np.sign(y_true) == np.sign(y_pred)

    squared_loss_rw = (y_true - y_rw) ** 2
    squared_loss_model = (y_true - y_pred) ** 2
    cumulative_advantage = (squared_loss_rw - squared_loss_model).cumsum()

    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_rw = np.sqrt(mean_squared_error(y_true, y_rw))
    mae_model = mean_absolute_error(y_true, y_pred)
    da_model = directional_accuracy_local(y_true, y_pred)
    improvement = 100 * (rmse_rw - rmse_model) / rmse_rw

    fig, axes = plt.subplots(
        3, 1,
        figsize=(17, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]}
    )

    fig.suptitle(
        f"{series_name} - {model_name}: 1-week ΔOAS forecasting performance",
        fontsize=18,
        fontweight="bold"
    )

    # ------------------------------------------------------
    # 1) Actual vs forecast with directional success markers
    # ------------------------------------------------------
    axes[0].plot(preds.index, y_true, label="Actual ΔOAS", linewidth=1.5)
    axes[0].plot(preds.index, y_pred, label=f"{model_name} forecast", linewidth=1.6)
    axes[0].axhline(0, linestyle="--", linewidth=1)

    axes[0].scatter(
        preds.index[correct_direction],
        y_true[correct_direction],
        s=35,
        marker="o",
        label="Correct direction",
        alpha=0.75
    )

    axes[0].scatter(
        preds.index[~correct_direction],
        y_true[~correct_direction],
        s=45,
        marker="x",
        label="Wrong direction",
        alpha=0.85
    )

    axes[0].set_title("Forecast path and directional success")
    axes[0].set_ylabel("Weekly ΔOAS (bps)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    metric_box = (
        f"Model RMSE: {rmse_model:.2f} bps\n"
        f"RW RMSE: {rmse_rw:.2f} bps\n"
        f"RMSE improvement: {improvement:.2f}%\n"
        f"MAE: {mae_model:.2f} bps\n"
        f"Directional accuracy: {da_model:.1%}"
    )

    axes[0].text(
        0.015, 0.95,
        metric_box,
        transform=axes[0].transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    # ------------------------------------------------------
    # 2) Forecast errors over time
    # ------------------------------------------------------
    forecast_error = y_true - y_pred

    axes[1].bar(preds.index, forecast_error, width=5)
    axes[1].axhline(0, linestyle="--", linewidth=1)
    axes[1].set_title("Forecast errors over time")
    axes[1].set_ylabel("Error: actual - forecast")
    axes[1].grid(True, alpha=0.25)

    # ------------------------------------------------------
    # 3) Cumulative advantage versus Random Walk
    # ------------------------------------------------------
    axes[2].plot(preds.index, cumulative_advantage, linewidth=2)
    axes[2].axhline(0, linestyle="--", linewidth=1)

    axes[2].fill_between(
        preds.index,
        cumulative_advantage,
        0,
        where=cumulative_advantage >= 0,
        alpha=0.25,
        interpolate=True
    )

    axes[2].fill_between(
        preds.index,
        cumulative_advantage,
        0,
        where=cumulative_advantage < 0,
        alpha=0.25,
        interpolate=True
    )

    axes[2].set_title("Cumulative squared-loss advantage vs Random Walk")
    axes[2].set_ylabel("Cumulative advantage")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.25)

    axes[2].text(
        0.015, 0.95,
        "Above zero: ML model beats Random Walk\nBelow zero: Random Walk performs better",
        transform=axes[2].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.12)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ----------------------------------------------------------
# Select only 1-week ML winners
# ----------------------------------------------------------

winners_1w_ml = winner_direct_ml.copy()
winners_1w_ml["model"] = winners_1w_ml["model"].astype(str).str.strip()

winners_1w_ml = winners_1w_ml[
    (winners_1w_ml["horizon_weeks"] == HORIZON_TO_PLOT) &
    (winners_1w_ml["model"].apply(is_ml_model))
].copy()

print("Impact plots for these 1-week ML winners:")
display(winners_1w_ml)

for _, row in winners_1w_ml.iterrows():
    plot_impact_winner(
        series_name=row["series"],
        horizon=int(row["horizon_weeks"]),
        model_name=row["model"]
    )

# COMMAND ----------

# ==========================================================
# 3D ML WINNER FORECAST RIBBON PLOT - 1 WEEK ONLY
# Actual vs predicted ΔOAS for ML winner models
# No saving, only display
# ==========================================================

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    %pip install plotly
    import plotly.graph_objects as go


HORIZON_TO_PLOT = 1


def is_ml_model(model_name):
    name = str(model_name).lower()
    excluded_terms = ["arma", "arima", "garch", "random walk", "rw", "selected arma"]
    return not any(term in name for term in excluded_terms)


# ----------------------------------------------------------
# 1) Select only 1-week ML winners
# ----------------------------------------------------------

winners_1w_ml = winner_direct_ml.copy()
winners_1w_ml["model"] = winners_1w_ml["model"].astype(str).str.strip()

winners_1w_ml = winners_1w_ml[
    (winners_1w_ml["horizon_weeks"] == HORIZON_TO_PLOT) &
    (winners_1w_ml["model"].apply(is_ml_model))
].copy()

print("1-week ML winners included in 3D plot:")
display(winners_1w_ml)


# ----------------------------------------------------------
# 2) Build 3D plot
# ----------------------------------------------------------

fig = go.Figure()

series_order = list(winners_1w_ml["series"])
series_y_map = {series: i for i, series in enumerate(series_order)}

for _, row in winners_1w_ml.iterrows():

    series_name = row["series"]
    model_name = row["model"]
    y_position = series_y_map[series_name]

    preds = all_prediction_tables[series_name][HORIZON_TO_PLOT].copy()
    preds = preds.dropna(subset=["actual", model_name]).copy()

    x_dates = preds.index
    y_axis = np.repeat(y_position, len(preds))

    actual = preds["actual"].values
    forecast = preds[model_name].values

    # Actual line
    fig.add_trace(go.Scatter3d(
        x=x_dates,
        y=y_axis,
        z=actual,
        mode="lines",
        name=f"{series_name} actual",
        line=dict(width=5),
        hovertemplate=(
            f"<b>{series_name} actual</b><br>"
            "Date: %{x}<br>"
            "ΔOAS: %{z:.2f} bps<extra></extra>"
        )
    ))

    # Forecast line, slightly shifted on y-axis so both lines are visible
    fig.add_trace(go.Scatter3d(
        x=x_dates,
        y=y_axis + 0.08,
        z=forecast,
        mode="lines",
        name=f"{series_name} forecast ({model_name})",
        line=dict(width=5, dash="dash"),
        hovertemplate=(
            f"<b>{series_name} forecast</b><br>"
            f"Model: {model_name}<br>"
            "Date: %{x}<br>"
            "Predicted ΔOAS: %{z:.2f} bps<extra></extra>"
        )
    ))


# ----------------------------------------------------------
# 3) Layout
# ----------------------------------------------------------

fig.update_layout(
    title="3D Forecast Ribbon Plot - ML winners, 1-week ΔOAS forecasts",
    width=1100,
    height=750,
    scene=dict(
        xaxis_title="Date",
        yaxis_title="Credit segment",
        zaxis_title="Weekly ΔOAS (bps)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(series_y_map.values()),
            ticktext=list(series_y_map.keys())
        )
    ),
    legend=dict(
        x=0.02,
        y=0.98
    ),
    margin=dict(l=0, r=0, b=0, t=60)
)

fig.show()

# COMMAND ----------

# ==========================================================
# 3D ERROR CLOUD - ML WINNERS, 1 WEEK ONLY
# Shows where ML winner models make larger errors
# ==========================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go

fig = go.Figure()

for _, row in winners_1w_ml.iterrows():

    series_name = row["series"]
    model_name = row["model"]
    y_position = series_y_map[series_name]

    preds = all_prediction_tables[series_name][HORIZON_TO_PLOT].copy()
    preds = preds.dropna(subset=["actual", model_name]).copy()

    actual = preds["actual"]
    forecast = preds[model_name]
    abs_error = np.abs(actual - forecast)

    correct_direction = np.sign(actual) == np.sign(forecast)

    fig.add_trace(go.Scatter3d(
        x=preds.index,
        y=np.repeat(y_position, len(preds)),
        z=abs_error,
        mode="markers",
        name=f"{series_name} error cloud",
        marker=dict(
            size=5,
            color=abs_error,
            colorscale="Viridis",
            opacity=0.75,
            colorbar=dict(title="Abs. error")
        ),
        text=[
            f"Series: {series_name}<br>"
            f"Model: {model_name}<br>"
            f"Actual: {a:.2f} bps<br>"
            f"Forecast: {f:.2f} bps<br>"
            f"Abs error: {e:.2f} bps<br>"
            f"Correct direction: {cd}"
            for a, f, e, cd in zip(actual, forecast, abs_error, correct_direction)
        ],
        hovertemplate="%{text}<extra></extra>"
    ))


fig.update_layout(
    title="3D Error Cloud - ML winners, 1-week forecast errors",
    width=1100,
    height=750,
    scene=dict(
        xaxis_title="Date",
        yaxis_title="Credit segment",
        zaxis_title="Absolute forecast error (bps)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(series_y_map.values()),
            ticktext=list(series_y_map.keys())
        )
    ),
    margin=dict(l=0, r=0, b=0, t=60)
)

fig.show()