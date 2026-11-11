from pathlib import Path
import numpy as np
import polars as pl

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation import evaluate_quality, run_diagnostic

# Load file paths for i/o
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "transaction_features.csv"
OUTPUT_FILE = DATA_DIR / "synthetic_transactions.csv"

QUALITY_REPORT_FILE = DATA_DIR / "synthetic_quality_report.html"
DIAGNOSTIC_REPORT_FILE = DATA_DIR / "synthetic_diagnostic_report.html"

TABLE_NAME = "transactions"

# Parameters for the data generated
SYNTHETIC_ROWS = 10_000
EPOCHS = 500

# List of the features
FEATURES = [
    "compute_units",
    "total_fee_lamports",
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "token_balance_change_count",
]

# Features that are counting the number of x associated with the txn
COUNT_FEATURES = [
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "token_balance_change_count",
]

# Unbounded features
CONTINUOUS_FEATURES = [
    "compute_units",
    "total_fee_lamports",
]


def load_data():
    # Load the real txn data and prep it for synthetic replication

    print("Loading original transaction data")

    if not INPUT_FILE.exists():
        raise FileNotFoundError( f"Input file not found: {INPUT_FILE}" )

    # Use polars bc polars is speeeedy :) (for python and lazyframes would be better usually but this is so small it doesn't matter)
    df = pl.read_csv(INPUT_FILE)

    missing_columns = [
        column for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only the features used by CTGAN.
    df = df.select(FEATURES)

    # Convert features to numeric values.
    df = df.with_columns([
        pl.col(column).cast(pl.Float64, strict=False)
        for column in FEATURES
    ])

    # Remove rows containing null or invalid values.
    df = df.drop_nulls()

    # Convert to pandas :((((
    pdf = df.to_pandas()

    # Remove rows containing infinite values.
    pdf = pdf.replace([np.inf, -np.inf], np.nan)
    pdf = pdf.dropna()

    # Transaction features should be non-negative.
    pdf = pdf[(pdf[FEATURES] >= 0).all(axis=1)]

    # Count features should contain whole numbers.
    for column in COUNT_FEATURES:
        pdf = pdf[
            np.isclose(
                pdf[column],
                np.round(pdf[column])
            )
        ]

    if len(pdf) == 0:
        raise ValueError("No valid transaction records remain after cleaning.")

    # Ensure count columns are integers.
    for column in COUNT_FEATURES:
        pdf[column] = pdf[column].round().astype("int64")

    print(f"Loaded {len(pdf)} valid transactions.")
    print("\nOriginal data summary:")
    print(pdf.describe())

    return pdf.reset_index(drop=True)

def create_metadata():
    # Create the metadata for Synthetic Data Vault (SDV), basically how to structure it

    print("\nCreating SDV metadata")

    metadata = Metadata()
    metadata.add_table(table_name=TABLE_NAME)

    for column in CONTINUOUS_FEATURES:
        metadata.add_column(
            table_name=TABLE_NAME,
            column_name=column,
            sdtype="numerical",
        )

    for column in COUNT_FEATURES:
        metadata.add_column(
            table_name=TABLE_NAME,
            column_name=column,
            sdtype="numerical",
        )

    metadata.validate()

    print("Metadata created and validated.")
    print(metadata.to_dict())

    return metadata

def train_model(pdf, metadata):
    # CTGANSynthesizer is an ML model used for tabular data synthetic generation
    print("\nInitializing CTGAN synthesizer...")

    # Setting up the model and it's parameters to fit within the parameters of the data
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=EPOCHS,
        verbose=True,
        enforce_min_max_values=True,
        enforce_rounding=True,
    )

    print(f"Training CTGAN for {EPOCHS} epochs")

    # Train the CTGAN on the data
    synthesizer.fit({TABLE_NAME: pdf})

    print("CTGAN training completed.")

    return synthesizer

def generate_synthetic_data(synthesizer):
    # Generate the synthetic txns using the CTGAN

    print(f"\nGenerating {SYNTHETIC_ROWS} synthetic transactions")

    # Generate 10_000 rows of synthetic data based on the real data it trained on
    generated_data = synthesizer.sample(num_rows=SYNTHETIC_ROWS)

    # Validation in case of SDV to extract the pandas df if it comes out as a dict or raw
    if isinstance(generated_data, dict):
        synthetic_pdf = generated_data[TABLE_NAME]
    else:
        synthetic_pdf = generated_data

    # Filters to keep only the features I care about
    synthetic_pdf = synthetic_pdf[FEATURES].copy()

    # Ensure count features are integers.
    for column in COUNT_FEATURES:
        synthetic_pdf[column] = (
            synthetic_pdf[column]
            .round()
            .clip(lower=0)
            .astype("int64")
        )

    # Ensure continuous features are non-negative.
    for column in CONTINUOUS_FEATURES:
        synthetic_pdf[column] = (
            synthetic_pdf[column]
            .clip(lower=0)
        )

    print(f"Generated {len(synthetic_pdf)} synthetic transactions.")

    return synthetic_pdf.reset_index(drop=True)


def validate_synthetic_data(synthetic_pdf):
    # Validation of the synthetic data

    print("\nValidating synthetic data")

    # Ensure that the number of rows matches
    if len(synthetic_pdf) != SYNTHETIC_ROWS:
        raise ValueError(f"Expected {SYNTHETIC_ROWS} rows, but got {len(synthetic_pdf)}.")

    # No nulls
    if synthetic_pdf[FEATURES].isnull().any().any():
        raise ValueError("Synthetic data contains missing values.")

    # Fits numerical constraints
    if not np.isfinite(synthetic_pdf[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Synthetic data contains infinite values.")

    if (synthetic_pdf[FEATURES] < 0).any().any():
        raise ValueError("Synthetic data contains negative values.")

    for column in COUNT_FEATURES:
        if not np.allclose( synthetic_pdf[column], np.round(synthetic_pdf[column]) ):
            raise ValueError(f"{column} contains non-integer values.")

    # This is a consistency check, not a hard CTGAN constraint.
    if (synthetic_pdf["inner_instruction_count"] > synthetic_pdf["instruction_count"]).any():
        print("WARNING: Some records have more inner instructions than instructions. Review this relationship.")

    print("Synthetic data validation passed.")

def evaluate_synthetic_data(pdf, synthetic_pdf, metadata):
    # Evaluate the quality of the data. Quality report is how well does it match real data, diagnostic ensures nothing broke

    print("\nRunning SDV quality evaluation")

    quality_report = evaluate_quality(real_data=pdf, synthetic_data=synthetic_pdf, metadata=metadata,)

    quality_score = quality_report.get_score()
    print(f"Overall Quality Score: {quality_score:.4f}")

    # Export the quality report for future ref
    quality_report.save(filepath=str(QUALITY_REPORT_FILE))
    print(f"Quality report saved to: {QUALITY_REPORT_FILE}")

    print("\nRunning SDV diagnostic evaluation")

    diagnostic_report = run_diagnostic(real_data=pdf, synthetic_data=synthetic_pdf, metadata=metadata,)
    diagnostic_score = diagnostic_report.get_score()

    print(f"Diagnostic Score: {diagnostic_score:.4f}")

    diagnostic_report.save(filepath=str(DIAGNOSTIC_REPORT_FILE))

    print(f"Diagnostic report saved to: {DIAGNOSTIC_REPORT_FILE}")

def compare_data(original_pdf, synthetic_pdf):
    # Prints the stats between both og and synthetic data
    print("\nOriginal vs Synthetic Statistics")

    for column in FEATURES:
        print(f"\nFeature: {column}")

        print(f"Original mean: {original_pdf[column].mean():.4f}")
        print(f"Synthetic mean: {synthetic_pdf[column].mean():.4f}")

        print(f"Original std: {original_pdf[column].std():.4f}")
        print(f"Synthetic std: {synthetic_pdf[column].std():.4f}")

        print( f"Original min/max: {original_pdf[column].min():.4f} / {original_pdf[column].max():.4f}")
        print(f"Synthetic min/max: {synthetic_pdf[column].min():.4f} / {synthetic_pdf[column].max():.4f}")

def save_synthetic_data(synthetic_pdf):
    print("\nSaving synthetic transactions")
    synthetic_df = pl.from_pandas(synthetic_pdf)
    synthetic_df.write_csv(OUTPUT_FILE)
    print(f"Synthetic data saved to: {OUTPUT_FILE}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # load the original data
    original_pdf = load_data()

    # create the metadata
    metadata = create_metadata()

    # train the CTGAN
    synthesizer = train_model(original_pdf, metadata)

    # generate synthetic data
    synthetic_pdf = generate_synthetic_data(synthesizer)

    # validate synthetic data.
    validate_synthetic_data(synthetic_pdf)

    # Save synthetic transactions.
    save_synthetic_data(synthetic_pdf)

    # Compare original and synthetic statistics.
    compare_data(original_pdf, synthetic_pdf)

    # Generate SDV evaluation reports.
    evaluate_synthetic_data( original_pdf, synthetic_pdf,  metadata)

if __name__ == "__main__":
    main()
