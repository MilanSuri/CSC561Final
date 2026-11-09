import polars as pl
from pathlib import Path

# Establishing directories, features thresholds, and global vars
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

REAL_FILE = DATA_DIR / "transaction_features.csv"
SYNTHETIC_FILE = DATA_DIR / "synthetic_transactions.csv"
OUTPUT_FILE = DATA_DIR / "weighted_transactions.csv"

WINDOW_SIZE = 100
OUTLIER_THRESHOLD = 3.5


FEATURES = [
    "compute_units",
    "total_fee_lamports",
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "token_balance_change_count",
]

# Data loader function
def load_data():

    if not REAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing real data: {REAL_FILE}"
        )

    if not SYNTHETIC_FILE.exists():
        raise FileNotFoundError(
            f"Missing synthetic data: {SYNTHETIC_FILE}. "
            "Run SyntheticData.py first."
        )

    real_df = pl.read_csv(REAL_FILE)

    synthetic_df = pl.read_csv(SYNTHETIC_FILE)

    real_df = real_df.with_columns([
        pl.col(feature).cast(pl.Float64, strict=False)
        for feature in FEATURES
    ])

    synthetic_df = synthetic_df.with_columns([
        pl.col(feature).cast(pl.Float64, strict=False)
        for feature in FEATURES
    ])

    # Keep chronological order for real observations
    if "notification_time" in real_df.columns:
        real_df = real_df.sort("notification_time")

    real_df = real_df.drop_nulls(subset=FEATURES)
    synthetic_df = synthetic_df.drop_nulls(subset=FEATURES)

    # Combine real and synthetic observations
    real_features = real_df.select(FEATURES)
    synthetic_features = synthetic_df.select(FEATURES)

    combined_df = pl.concat(
        [real_features, synthetic_features],
        how="vertical"
    )

    print("Datasets loaded.")
    print(f"Real rows: {real_df.height}")
    print(f"Synthetic rows: {synthetic_df.height}")
    print(f"Combined rows: {combined_df.height}")

    return combined_df

# calculate the medians
def calculate_running_medians(df):

    results = df

    for feature in FEATURES:

        results = results.with_columns(

            pl.col(feature)
            .rolling_median(
                window_size=WINDOW_SIZE,
                min_samples=1
            )
            .alias(f"{feature}_running_median")

        )

    return results

# Determine if outliers or not
def calculate_mad_weights(df):

    results = df

    for feature in FEATURES:

        median_column = f"{feature}_running_median"

        deviation_column = f"{feature}_deviation"

        mad_column = f"{feature}_mad"

        score_column = f"{feature}_robust_score"

        outlier_column = f"{feature}_outlier"

        weight_column = f"{feature}_weight"

        # Absolute deviation from running median
        results = results.with_columns(

            (
                pl.col(feature) - pl.col(median_column)
            )
            .abs()
            .alias(deviation_column)

        )

        # Rolling median absolute deviation
        results = results.with_columns(

            pl.col(deviation_column)
            .rolling_median(
                window_size=WINDOW_SIZE,
                min_samples=1
            )
            .alias(mad_column)

        )

        # Robust deviation score
        results = results.with_columns(

            pl.when(pl.col(mad_column) > 0)
            .then(
                0.6745
                * pl.col(deviation_column)
                / pl.col(mad_column)
            )
            .otherwise(0.0)
            .alias(score_column)

        )

        # Outlier detection
        results = results.with_columns(

            (
                pl.col(score_column) > OUTLIER_THRESHOLD
            )
            .alias(outlier_column)

        )

        # Observation weight
        # 1.0 = normal
        # Lower = more unusual

        results = results.with_columns(

            pl.when(pl.col(outlier_column))
            .then(
                1.0 / (1.0 + pl.col(score_column))
            )
            .otherwise(1.0)
            .alias(weight_column)

        )

    return results


# --------------------------------------------------
# FEATURE WEIGHTS
# --------------------------------------------------

def calculate_feature_weights(df):

    weights = {}

    for feature in FEATURES:

        weight_column = f"{feature}_weight"

        average_weight = df[weight_column].mean()

        weights[feature] = average_weight

    total = sum(weights.values())

    if total > 0:

        weights = {
            feature: weight / total
            for feature, weight in weights.items()
        }

    print("\nNormalized feature weights:")

    for feature, weight in weights.items():

        print(f"{feature}: {weight:.4f}")

    return weights


def main():

    df = load_data()

    print("\nCalculating running medians...")

    df = calculate_running_medians(df)

    print("Calculating MAD and outlier weights...")

    df = calculate_mad_weights(df)

    feature_weights = calculate_feature_weights(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.write_csv(OUTPUT_FILE)

    print("\nProcessing complete.")
    print(f"Total rows: {df.height}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()