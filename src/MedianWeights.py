import polars as pl
import numpy as np
from pathlib import Path

# I/O files for the median weights
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

REAL_FILE = DATA_DIR / "transaction_features.csv"
SYNTHETIC_FILE = DATA_DIR / "synthetic_transactions.csv"
OUTPUT_FILE = DATA_DIR / "weighted_transactions.csv"

# Set a parameter for determining when to label a threshold
OUTLIER_THRESHOLD = 3.5

# Identify the features the isoforest uses
FEATURES = [
    "compute_units",
    "total_fee_lamports",
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "token_balance_change_count",
]

def clean_data(df):
    # Removes missing data from features cols
    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Converts each feature into a 64 bit float so no unexpected formatting errors/probs
    df = df.with_columns([
        pl.col(feature).cast(pl.Float64, strict=False)
        for feature in FEATURES
    ])

    # Ensures no nulls, no infinite numbers, no negatives
    df = df.drop_nulls(subset=FEATURES)

    df = df.filter(
        pl.all_horizontal([
            pl.col(feature).is_finite()
            for feature in FEATURES
        ])
    )

    df = df.filter(
        pl.all_horizontal([
            pl.col(feature) >= 0
            for feature in FEATURES
        ])
    )

    return df

def load_data():
    print("Loading transaction datasets")

    # Validation for if file isn't present
    if not REAL_FILE.exists():
        raise FileNotFoundError(f"Missing real data: {REAL_FILE}")

    if not SYNTHETIC_FILE.exists():
        raise FileNotFoundError(f"Missing synthetic data: {SYNTHETIC_FILE}")

    # prep both dfs
    real_df = clean_data(pl.read_csv(REAL_FILE))
    synthetic_df = clean_data(pl.read_csv(SYNTHETIC_FILE))

    # create a column data source and flag real/synthetic
    real_df = real_df.with_columns(pl.lit("real").alias("data_source"))
    synthetic_df = synthetic_df.with_columns(pl.lit("synthetic").alias("data_source"))

    print(f"Real transactions: {real_df.height}")
    print(f"Synthetic transactions: {synthetic_df.height}")

    return real_df, synthetic_df

def calculate_reference_statistics(real_df):
    # Get the stats of the real data
    print("\nCalculating statistics.")

    reference = {}

    # loop through features
    for feature in FEATURES:
        values = real_df[feature].to_numpy()

        # Calculate the median, the deviation, and the mean average deviation to get the variability of the dataset
        median = float(np.median(values))
        absolute_deviations = np.abs(values - median)
        mad = float(np.median(absolute_deviations))

        # Fallback for features with zero MAD, calculate by doing stdev of IQR
        if mad == 0:
            q25, q75 = np.percentile(values,[25, 75])
            mad = float((q75 - q25) / 1.349)

        # Final fallback for constant-valued features.
        if mad == 0:
            mad = float(np.std(values))

        # Hardcode if it fails lol
        if mad == 0:
            mad = 1.0

        reference[feature] = {"median": median, "mad": mad,}

        print(f"{feature}: Median={median:.4f}, MAD={mad:.4f}")

    return reference

def calculate_mad_weights(df, reference):
    # Get the weights for each feature
    print("\nCalculating global MAD scores and weights...")

    results = df

    for feature in FEATURES:
        # Loop through and create unique columns
        median_column = f"{feature}_global_median"
        deviation_column = f"{feature}_deviation"
        mad_column = f"{feature}_mad"
        score_column = f"{feature}_robust_score"
        outlier_column = f"{feature}_outlier"
        weight_column = f"{feature}_weight"

        # gets the pre-calculated values
        median = reference[feature]["median"]
        mad = reference[feature]["mad"]

        # Global median.
        results = results.with_columns(pl.lit(median).alias(median_column))

        # Absolute deviation from global median.
        results = results.with_columns( ( pl.col(feature) - pl.col(median_column) ).abs().alias(deviation_column) )

        # Global MAD.
        results = results.with_columns( pl.lit(mad).alias(mad_column) )

        # Modified Z-score. Zscore is datapoint relationship to the mean
        results = results.with_columns( ( 0.6745 * ( pl.col(feature) - median ) / mad ).alias(score_column) )

        # Outlier detection uses absolute score.
        results = results.with_columns( (pl.col(score_column).abs()  > OUTLIER_THRESHOLD ).alias(outlier_column))

        # Assign lower weights to extreme observations.
        results = results.with_columns(( 1.0 / ( 1.0 + pl.col(score_column).abs() )).alias(weight_column) )
    return results

def calculate_feature_weights(real_df):
    # Method that takes in the weights from above and then normalizes so they are 0-1.0
    print("\nCalculating normalized feature weights")

    weights = {}

    # Loops through each feature and calculates the average weight and stores it
    for feature in FEATURES:
        weight_column = f"{feature}_weight"
        average_weight = real_df[weight_column].mean()
        weights[feature] = float(average_weight)

    # Get a total to normalize them
    total = sum(weights.values())

    if total <= 0:
        raise ValueError("Unable to calculate normalized feature weights.")

    # Normalize the weights to be between 0-1.0
    weights = {
        feature: weight / total
        for feature, weight in weights.items()
    }

    print("\nNormalized Feature Weights")

    for feature, weight in weights.items():
        print(f"{feature}: {weight:.4f}")

    return weights

def print_outlier_summary(real_df, synthetic_df):
    # Tells me the outliers in the real/synthetic data
    print("\nOutlier Summary")

    # Loops through the feature column and calculates the sum of outliers in real/synthetic, then percentage
    for feature in FEATURES:
        outlier_column = f"{feature}_outlier"

        real_outliers = real_df[outlier_column].sum()
        synthetic_outliers = synthetic_df[outlier_column].sum()

        real_percentage = (100 * real_outliers / real_df.height)
        synthetic_percentage = (100 * synthetic_outliers / synthetic_df.height)

        print(f"\n{feature}")

        print(f"Real outliers: {real_outliers} ({real_percentage:.2f}%)")

        print(f"Synthetic outliers: {synthetic_outliers}  ({synthetic_percentage:.2f}%)")

def main():

    # Load data
    real_df, synthetic_df = load_data()

    # calculate reference statistics using all real data.
    reference = calculate_reference_statistics(real_df)

    # Score real and synthetic observations against the same global real-data reference.
    real_df = calculate_mad_weights(real_df, reference)
    synthetic_df = calculate_mad_weights(synthetic_df, reference)

    # Calculate feature weights using real observations.
    feature_weights = calculate_feature_weights(real_df)

    # outlier comparison.
    print_outlier_summary(real_df, synthetic_df)

    # combine both datasets.
    combined_df = pl.concat([real_df, synthetic_df], how="diagonal_relaxed")

    # Save output.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    combined_df.write_csv(OUTPUT_FILE)

    print(f"Real rows: {real_df.height}")
    print(f"Synthetic rows: {synthetic_df.height}")
    print(f"Combined rows: {combined_df.height}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
