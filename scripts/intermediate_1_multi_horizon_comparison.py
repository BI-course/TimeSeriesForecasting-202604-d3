from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from tslab import config
from tslab.data import load_market_data
from tslab.features import ALL_FEATURES, make_dataset
from tslab.validation import walk_forward_splits


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

HORIZONS = [1, 5, 21, 63]

RIDGE_ALPHA = 100000

GB_SETTINGS = {
    "max_depth": 2,
    "learning_rate": 0.03,
    "max_iter": 150,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": config.SEED,
}

OUTPUT_DIR = Path("outputs/intermediate_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(actual_price, predicted_price,
                      actual_return, predicted_return):
    actual_price = np.asarray(actual_price, dtype=float)
    predicted_price = np.asarray(predicted_price, dtype=float)
    actual_return = np.asarray(actual_return, dtype=float)
    predicted_return = np.asarray(predicted_return, dtype=float)

    price_error = predicted_price - actual_price
    return_error = predicted_return - actual_return

    rmse = float(np.sqrt(np.mean(price_error ** 2)))
    mae = float(np.mean(np.abs(price_error)))
    mape = float(
        np.mean(np.abs(price_error) / np.abs(actual_price)) * 100
    )

    # Out-of-sample R2 relative to the zero-return (naive) forecast.
    denominator = np.sum(actual_return ** 2)
    if denominator > 0:
        r2 = float(
            1 - np.sum(return_error ** 2) / denominator
        )
    else:
        r2 = np.nan

    # Exclude zero predicted returns because they do not express
    # an upward or downward directional forecast.
    directional_mask = (
        (predicted_return != 0)
        & (actual_return != 0)
    )

    if directional_mask.any():
        directional_accuracy = float(
            np.mean(
                np.sign(predicted_return[directional_mask])
                == np.sign(actual_return[directional_mask])
            ) * 100
        )
    else:
        directional_accuracy = np.nan

    return {
        "RMSE_price": rmse,
        "MAE_price": mae,
        "MAPE_price_pct": mape,
        "R2_returns": r2,
        "Directional_accuracy_pct": directional_accuracy,
    }


def calculate_dm_test(result, model_price_column, horizon):
    """Compare a model with Naive using full-series HAC loss variance."""
    ordered = result.sort_values("origin_date")

    model_error = (
        ordered[model_price_column].to_numpy(dtype=float)
        - ordered["actual_close"].to_numpy(dtype=float)
    )
    naive_error = (
        ordered["naive_price"].to_numpy(dtype=float)
        - ordered["actual_close"].to_numpy(dtype=float)
    )

    differences = model_error ** 2 - naive_error ** 2
    n = len(differences)

    if n < 2:
        raise ValueError("Insufficient observations for DM test.")

    mean_difference = float(np.mean(differences))
    centred = differences - mean_difference

    max_lag = min(horizon - 1, n - 1)
    gamma0 = float(np.dot(centred, centred) / n)
    long_run_variance = gamma0

    for lag in range(1, max_lag + 1):
        gamma = float(
            np.dot(centred[lag:], centred[:-lag]) / n
        )
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * gamma

    variance_of_mean = max(long_run_variance, 0.0) / n

    if variance_of_mean <= 0:
        statistic, p_value = np.nan, np.nan
    else:
        statistic = mean_difference / np.sqrt(variance_of_mean)
        p_value = float(2.0 * norm.sf(abs(statistic)))

    return {
        "test_rows": n,
        "mean_loss_difference": mean_difference,
        "DM_statistic": statistic,
        "p_value": p_value,
        "HAC_max_lag": max_lag,
    }


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 70)
print("INTERMEDIATE 1: MULTI-HORIZON MODEL COMPARISON")
print("=" * 70)

frame = load_market_data(
    target=config.TARGET_TICKER,
    verbose=True,
)

if frame.empty:
    raise ValueError("The market-data frame is empty.")

if "Close" not in frame.columns:
    raise KeyError("The market-data frame has no Close column.")

if not frame.index.is_monotonic_increasing:
    raise ValueError("The market-data dates are not in ascending order.")

if frame.index.has_duplicates:
    raise ValueError("Duplicate dates found in market data.")

print(f"\nTicker: {config.TARGET_TICKER}")
print(f"Market rows: {len(frame):,}")
print(f"First market date: {frame.index.min().date()}")
print(f"Last market date: {frame.index.max().date()}")
print(f"Features: {len(ALL_FEATURES)}")
print(f"Horizons: {HORIZONS}")


# ---------------------------------------------------------
# Run each horizon
# ---------------------------------------------------------

all_predictions = []
all_metrics = []
all_dm_results = []

for horizon in HORIZONS:
    print("\n" + "=" * 70)
    print(f"HORIZON: {horizon} TRADING DAY(S)")
    print("=" * 70)

    data_h = make_dataset(frame, horizon=horizon)
    data_h = data_h.dropna(subset=["target_return", "target_close"])

    if data_h.empty:
        raise ValueError(f"No usable rows for horizon {horizon}.")

    X = data_h[ALL_FEATURES].astype(float)
    y = data_h["target_return"].astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Non-finite feature values at horizon {horizon}.")

    if not np.isfinite(y.to_numpy()).all():
        raise ValueError(f"Non-finite target values at horizon {horizon}.")

    # Map every forecast origin to its actual future trading date.
    origin_positions = frame.index.get_indexer(data_h.index)

    if (origin_positions < 0).any():
        raise ValueError("Could not align dataset dates with market dates.")

    target_positions = origin_positions + horizon

    if (target_positions >= len(frame)).any():
        raise ValueError(
            f"Target-date position exceeds market data at horizon {horizon}."
        )

    target_dates = frame.index.take(target_positions)

    splits = list(
        walk_forward_splits(
            len(data_h),
            config.INITIAL_TRAIN,
            config.TEST_SIZE,
            gap=horizon - 1,
        )
    )

    if not splits:
        raise ValueError(f"No walk-forward folds for horizon {horizon}.")

    print(f"Usable modelling rows: {len(data_h):,}")
    print(f"Walk-forward folds: {len(splits)}")
    print(f"Initial training rows: {config.INITIAL_TRAIN}")
    print(f"Test block size: {config.TEST_SIZE}")
    print(f"Gap between training and test: {horizon - 1}")

    # Each model predicts the same test rows.
    predictions = {
        "naive_return": np.full(len(data_h), np.nan),
        "ses_return": np.full(len(data_h), np.nan),
        "drift_return": np.full(len(data_h), np.nan),
        "ridge_return": np.full(len(data_h), np.nan),
        "gradient_boosting_return": np.full(len(data_h), np.nan),
    }

    fold_numbers = np.full(len(data_h), np.nan)

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        # 1. Naive: forecast no change in price.
        predictions["naive_return"][test_idx] = 0.0

        # 2. Drift: use the mean historical horizon return.
        predictions["drift_return"][test_idx] = float(y_train.mean())

        # 3. Simple Exponential Smoothing.
        # Estimate alpha on training closes only. Update the level
        # with observed closes up to each forecast origin.
        training_close = pd.Series(
            data_h["close"].iloc[train_idx].to_numpy(dtype=float)
        )
        ses_fit = SimpleExpSmoothing(
            training_close,
            initialization_method="estimated",
        ).fit(optimized=True)
        alpha = float(ses_fit.params["smoothing_level"])
        ses_level = float(ses_fit.level.iloc[-1])
        last_updated_idx = int(train_idx[-1])

        for position in test_idx:
            for update_idx in range(last_updated_idx + 1, position + 1):
                observed = float(data_h["close"].iloc[update_idx])
                ses_level = alpha * observed + (1.0 - alpha) * ses_level
            current_close = float(data_h["close"].iloc[position])
            predictions["ses_return"][position] = np.log(
                ses_level / current_close
            )
            last_updated_idx = int(position)

        # 4. Ridge: fit the scaler on training data only.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_train_scaled, y_train)

        predictions["ridge_return"][test_idx] = ridge.predict(
            X_test_scaled
        )

        # 5. Gradient boosting.
        boosting = HistGradientBoostingRegressor(**GB_SETTINGS)
        boosting.fit(X_train, y_train)

        predictions["gradient_boosting_return"][test_idx] = (
            boosting.predict(X_test)
        )

        fold_numbers[test_idx] = fold_number

        print(
            f"Fold {fold_number:02d}/{len(splits)} | "
            f"train={len(train_idx):4d} | "
            f"test={len(test_idx):3d} | "
            f"test dates={data_h.index[test_idx[0]].date()} "
            f"to {data_h.index[test_idx[-1]].date()}"
        )

    # Verify that every model has predictions for exactly the same rows.
    common_mask = np.isfinite(fold_numbers)

    for model_name, values in predictions.items():
        if not np.isfinite(values[common_mask]).all():
            raise AssertionError(
                f"{model_name} has missing predictions "
                f"inside the test sample for horizon {horizon}."
            )

    if common_mask.sum() == 0:
        raise ValueError(f"No test predictions for horizon {horizon}.")

    # Build the prediction table.
    close_today = data_h["close"].to_numpy(dtype=float)
    actual_close = data_h["target_close"].to_numpy(dtype=float)
    actual_return = y.to_numpy(dtype=float)

    result = pd.DataFrame({
        "origin_date": data_h.index,
        "target_date": target_dates,
        "horizon": horizon,
        "fold": fold_numbers,
        "close_today": close_today,
        "actual_close": actual_close,
        "actual_return": actual_return,
    })

    for model_name, predicted_return in predictions.items():
        result[model_name] = predicted_return

        price_column = model_name.replace("_return", "_price")
        result[price_column] = (
            close_today * np.exp(predicted_return)
        )

    result = result.loc[common_mask].copy()
    result["fold"] = result["fold"].astype(int)

    # Calculate metrics on the common out-of-sample test rows.
    model_columns = {
        "Naive": "naive_return",
        "SES": "ses_return",
        "Drift": "drift_return",
        "Ridge": "ridge_return",
        "Gradient boosting": "gradient_boosting_return",
    }

    for model_label, return_column in model_columns.items():
        price_column = return_column.replace("_return", "_price")

        metrics = calculate_metrics(
            actual_price=result["actual_close"],
            predicted_price=result[price_column],
            actual_return=result["actual_return"],
            predicted_return=result[return_column],
        )

        all_metrics.append({
            "horizon": horizon,
            "model": model_label,
            "test_rows": len(result),
            **metrics,
        })

    # Compare each non-naive model against the naive benchmark.
    for model_label, return_column in model_columns.items():
        if model_label == "Naive":
            continue
        price_column = return_column.replace("_return", "_price")
        dm_result = calculate_dm_test(
            result, price_column, horizon
        )
        all_dm_results.append({
            "horizon": horizon,
            "model_vs_naive": model_label,
            **dm_result,
        })

    all_predictions.append(result)

    print(f"\nScored test rows: {len(result):,}")
    print("Price RMSE by model:")

    for model_label, return_column in model_columns.items():
        price_column = return_column.replace("_return", "_price")
        rmse = np.sqrt(
            np.mean(
                (
                    result[price_column] - result["actual_close"]
                ) ** 2
            )
        )
        print(f"  {model_label:20s}: ${rmse:.4f}")


# ---------------------------------------------------------
# Save and validate outputs
# ---------------------------------------------------------

predictions_df = pd.concat(all_predictions, ignore_index=True)
metrics_df = pd.DataFrame(all_metrics)
dm_df = pd.DataFrame(all_dm_results)

predictions_path = OUTPUT_DIR / "multi_horizon_predictions.csv"
metrics_path = OUTPUT_DIR / "multi_horizon_metrics.csv"
dm_path = OUTPUT_DIR / "diebold_mariano_tests.csv"

predictions_df.to_csv(predictions_path, index=False)
metrics_df.to_csv(metrics_path, index=False)
dm_df.to_csv(dm_path, index=False)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nMetrics:")
print(
    metrics_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nDiebold-Mariano tests against Naive:")
print(dm_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

print("\nPrediction rows by horizon:")
print(
    predictions_df.groupby("horizon").size().to_string()
)

print("\nOutput files:")
print(f"  {predictions_path} ({predictions_path.stat().st_size:,} bytes)")
print(f"  {metrics_path} ({metrics_path.stat().st_size:,} bytes)")
print(f"  {dm_path} ({dm_path.stat().st_size:,} bytes)")

# Final integrity checks.
assert predictions_df["origin_date"].notna().all()
assert predictions_df["target_date"].notna().all()
assert predictions_df["actual_close"].notna().all()
assert predictions_df["actual_return"].notna().all()
assert not predictions_df.duplicated(
    subset=["origin_date", "horizon"]
).any()

expected_models = 5
assert len(metrics_df) == len(HORIZONS) * expected_models
assert len(dm_df) == len(HORIZONS) * (expected_models - 1)
assert set(dm_df["model_vs_naive"]) == {
    "SES", "Drift", "Ridge", "Gradient boosting"
}

print("\nAll integrity checks passed.")
