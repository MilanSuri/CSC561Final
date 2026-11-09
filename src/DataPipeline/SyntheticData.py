
import polars as pl
import numpy as np

from pathlib import Path


#I/O and params
INPUT_FILE = Path("../../data/transaction_features.csv")
OUTPUT_FILE = Path("../../data/synthetic_transactions.csv")

SYNTHETIC_ROWS = 1000
RANDOM_SEED = 42

# Features generated for
FEATURES = [
    "compute_units",
    "total_fee_lamports",
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "token_balance_change_count",
]



# Load steamed data
def load_real_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}"
        )

    df = pl.read_csv(INPUT_FILE)

    df = df.with_columns([
        pl.col(feature).cast(
            pl.Float64,
            strict=False
        )
        for feature in FEATURES
    ])

    df = df.drop_nulls(subset=FEATURES)

    if df.height == 0:
        raise ValueError("No valid transaction rows found.")

    print("Real transaction data loaded.")
    print(f"Rows: {df.height}")

    return df

# Generate random synthetic data per feature
def generate_synthetic_data(df):

    rng = np.random.default_rng(RANDOM_SEED)

    synthetic_columns = {}

    for feature in FEATURES:

        values = df[feature].drop_nulls().to_numpy()

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:

            generated = np.full(
                SYNTHETIC_ROWS,
                mean
            )

        else:

            generated = rng.normal(
                mean,
                std,
                SYNTHETIC_ROWS
            )

        # Transaction counts and other listed features
        # cannot be negative.
        generated = np.maximum(
            generated,
            0
        )

        synthetic_columns[feature] = generated

    synthetic_df = pl.DataFrame(
        synthetic_columns
    )

    # Round count-based features to integers
    count_features = [
        "account_count",
        "instruction_count",
        "inner_instruction_count",
        "token_balance_change_count",
    ]

    synthetic_df = synthetic_df.with_columns([
        pl.col(feature).round(0)
        for feature in count_features
    ])

    return synthetic_df

def main():

    df = load_real_data()

    synthetic_df = generate_synthetic_data(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    synthetic_df.write_csv(OUTPUT_FILE)

    print("\nSynthetic data generated.")
    print(f"Rows generated: {synthetic_df.height}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()