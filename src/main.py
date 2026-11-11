from pathlib import Path
import polars as pl
import onnx
from onnx import helper, TensorProto
from isoforest import *

# Uploaded the paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

REAL_FILE = DATA_DIR / "transaction_features.csv"
SYNTHETIC_FILE = DATA_DIR / "synthetic_transactions.csv"

OUTPUT_FILE = DATA_DIR / "weighted_iforest_results.csv"
SUMMARY_FILE = DATA_DIR / "weighted_iforest_summary.csv"

WEIGHTED_MODEL_FILE = MODEL_DIR / "weighted_isolation_forest.onnx"

# Features used in the model
FEATURES = [
    "compute_units",
    "total_fee_lamports",
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "token_balance_change_count",
]

# Model hyperparameters
N_ESTIMATORS = 200
MAX_SAMPLES = 512
CONTAMINATION = 0.05
RANDOM_STATE = 42

# Real data split proportions
TRAIN_RATIO = 0.60
CALIBRATION_RATIO = 0.20
# Remaining 20% is reserved for testing.


def load_feature_data(file_path):
    # Load the data and prep it so that no missing/nulls

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pl.read_csv(file_path)

    missing = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing features in {file_path.name}: {missing}")

    # Convert features to numeric and replace NaNs with nulls.
    df = df.with_columns([
        pl.col(feature)
        .cast(pl.Float64, strict=False)
        .fill_nan(None)
        for feature in FEATURES
    ])

    return df

def prepare_features(df, training_medians):
    # prep the features for training

    # Use real training medians for both datasets.
    df = df.with_columns([
        pl.col(feature).fill_null(training_medians[feature])
        for feature in FEATURES
    ])

    X = df.select(FEATURES).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains infinite or invalid values.")

    return df, X

def split_real_data(X, random_state=42):
    # Create the train/test/calibration splits

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))

    # Split into train calibration and then split into train/calibrate/test
    train_end = int(len(X) * TRAIN_RATIO)
    calibration_end = int(len(X) * (TRAIN_RATIO + CALIBRATION_RATIO))

    train_indices = indices[:train_end]
    calibration_indices = indices[train_end:calibration_end]
    test_indices = indices[calibration_end:]

    if (len(train_indices) < 2 or len(calibration_indices) < 2 or len(test_indices) < 1):
        raise ValueError("Not enough observations for the requested split.")

    return (
        train_indices,
        calibration_indices,
        test_indices,
    )

def calculate_heuristic_weights(X_train):
    """
    Calculate feature-selection weights using real training data.
    Uses modified Z-scores based on median and MAD.

    Observations closer to a feature's median receive higher
    observation weights. The mean observation weight becomes
    that feature's selection weight.
    """

    feature_weights = []

    for i, feature in enumerate(FEATURES):

        values = X_train[:, i]
        median = np.median(values)
        mad = np.median( np.abs(values - median) )

        if mad == 0:
            weight = 1.0
        else:
            modified_z = ( 0.6745 * (values - median) / mad)
            observation_weights = ( 1.0 / (1.0 + np.abs(modified_z)) )
            weight = np.mean(observation_weights)

        feature_weights.append(weight)

    feature_weights = np.asarray(feature_weights, dtype=float,)
    feature_weights /= feature_weights.sum()

    print("\nHeuristic feature-selection weights:")

    for feature, weight in zip(FEATURES, feature_weights):

        print(f"{feature}: {weight:.4f} ({weight * 100:.2f}%)")

    return feature_weights

def train_model(X_train, X_calibration, feature_weights,):

    print("\nTraining weighted Isolation Forest")

    # Create the isoforest
    model = WeightedIsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        feature_weights=feature_weights,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )

    # Train the forest using real training observations.
    model.fit(X_train)

    # Calibrate the anomaly threshold using separate real observations that were not used for training.
    model.calibrate(X_calibration)

    return model

def evaluate_dataset(df, X, source, model,):
    # have the model predict and then get the anomalies

    # Score the different rows + eval if anomaly or not
    scores = model.score_samples(X)
    predictions = model.predict(X)

    results = df.select(FEATURES).with_columns([
        pl.lit(source).alias("source"),
        pl.Series("anomaly_score", scores,),
        pl.Series("is_anomaly", predictions,),
    ])

    return results

def summarize_results(results):
    # Generate and print a summary from the training of the model

    summary_rows = []

    for source in ["Real test", "Synthetic"]:
        subset = results.filter(pl.col("source") == source)
        scores = subset["anomaly_score"].to_numpy()
        predictions = subset["is_anomaly"].to_numpy()

        total = len(predictions)

        if total == 0:
            continue

        anomaly_count = int(np.sum(predictions == 1))

        row = {
            "dataset": source,
            "transactions": total,
            "anomalies": anomaly_count,
            "anomaly_percentage": (
                100.0 * anomaly_count / total
            ),
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "p95_score": float(np.quantile(scores, 0.95)),
            "max_score": float(np.max(scores)),
        }

        summary_rows.append(row)

        print(f"\n{source}")

        print(f"Transactions: {total}")
        print(f"Anomalies: {anomaly_count}")
        print(f"Anomaly percentage: {row['anomaly_percentage']:.2f}%")
        print( f"Mean score: {row['mean_score']:.4f}")
        print(f"Median score: {row['median_score']:.4f}")
        print(f"95th percentile: {row['p95_score']:.4f}")
        print(f"Maximum score: {row['max_score']:.4f}")

    return pl.DataFrame(summary_rows)

def calculate_leaf_depth(nodes, target_node):
    # Calculates how isolated a node is

    def search(node_id, depth):
        # Searches for a node by recursively iterating through the tree
        node = nodes[node_id]

        # return depth if found
        if node_id == target_node:
            return depth

        if node["feature"] is None:
            return None

        # Look Left
        left_result = search(node["left"], depth + 1,)

        if left_result is not None:
            return left_result

        # Check right if left isn't found
        return search(node["right"], depth + 1,)

    result = search(0, 0)

    if result is None:
        raise ValueError("Could not determine leaf depth.")

    return result

def export_onnx(model, model_file):

    MODEL_DIR.mkdir(parents=True, exist_ok=True,)

    if model.threshold is None:
        raise ValueError("Cannot export an uncalibrated model.")

    # Store the diff vals for ONNX
    nodes_treeids = []
    nodes_nodeids = []
    nodes_featureids = []
    nodes_modes = []
    nodes_values = []
    nodes_truenodeids = []
    nodes_falsenodeids = []
    nodes_missing_value_tracks_true = []

    target_treeids = []
    target_nodeids = []
    target_ids = []
    target_weights = []

    # Loop through the trees and nodes
    for tree_id, tree in enumerate(model.trees):
        for node_id, node in enumerate(tree.nodes):

            nodes_treeids.append(tree_id)
            nodes_nodeids.append(node_id)

            if node["feature"] is None:
                # Leaf node.
                nodes_featureids.append(0)
                nodes_modes.append("LEAF")
                nodes_values.append(0.0)

                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)

                nodes_missing_value_tracks_true.append(0)

                # Calculate total path length, including the expected remaining path at the leaf.
                leaf_path = ( calculate_leaf_depth(tree.nodes, node_id,) + average_path_length(node["size"]) )

                target_treeids.append(tree_id)
                target_nodeids.append(node_id)
                target_ids.append(0)
                target_weights.append(float(leaf_path))
            else:
                # Internal split node.
                nodes_featureids.append(node["feature"])
                nodes_modes.append("BRANCH_LT")
                nodes_values.append(node["split"])

                nodes_truenodeids.append(node["left"])
                nodes_falsenodeids.append(node["right"])

                nodes_missing_value_tracks_true.append(0)

    # Tree ensemble calculates the average path length across all trees.
    tree_ensemble = helper.make_node(
        "TreeEnsembleRegressor",
        inputs=["features"],
        outputs=["average_path_length"],
        domain="ai.onnx.ml",
        n_targets=1,
        aggregate_function="AVERAGE",
        post_transform="NONE",
        nodes_treeids=nodes_treeids,
        nodes_nodeids=nodes_nodeids,
        nodes_featureids=nodes_featureids,
        nodes_modes=nodes_modes,
        nodes_values=nodes_values,
        nodes_truenodeids=nodes_truenodeids,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_missing_value_tracks_true=(
            nodes_missing_value_tracks_true
        ),
        target_treeids=target_treeids,
        target_nodeids=target_nodeids,
        target_ids=target_ids,
        target_weights=target_weights,
    )

    normalization = average_path_length(model.sample_size)

    if normalization <= 0:
        raise ValueError("Invalid path-length normalization.")

    # Uses ONNX.helper to create tensors for the weights, threshold for anomaly, and constant
    initializers = [
        helper.make_tensor(
            "two",
            TensorProto.FLOAT,
            [1],
            [2.0],
        ),

        helper.make_tensor(
            "normalization",
            TensorProto.FLOAT,
            [1],
            [float(normalization)],
        ),

        helper.make_tensor(
            "threshold",
            TensorProto.FLOAT,
            [1],
            [float(model.threshold)],
        ),
    ]

    # Convert average path length into anomaly score: score = 2 ^ (-average_path_length / normalization)
    # Higher scores indicate more anomalous observations. These create nodes for doing diff operations.
    scoring_nodes = [

        helper.make_node(
            "Neg",
            ["average_path_length"],
            ["negative_path"],
        ),

        helper.make_node(
            "Div",
            ["negative_path", "normalization"],
            ["exponent"],
        ),

        helper.make_node(
            "Pow",
            ["two", "exponent"],
            ["anomaly_score"],
        ),

        helper.make_node(
            "GreaterOrEqual",
            ["anomaly_score", "threshold"],
            ["anomaly_boolean"],
        ),

        helper.make_node(
            "Cast",
            ["anomaly_boolean"],
            ["is_anomaly"],
            to=TensorProto.INT64,
        ),
    ]

    # Makes the graph combine tree and scoring nodes, define the inpts/outputs of the model
    graph = helper.make_graph(
        nodes=[
            tree_ensemble,
            *scoring_nodes,
        ],
        name="WeightedIsolationForest",
        inputs=[
            helper.make_tensor_value_info(
                "features",
                TensorProto.FLOAT,
                [None, len(FEATURES)],
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "anomaly_score",
                TensorProto.FLOAT,
                [None, 1],
            ),

            helper.make_tensor_value_info(
                "is_anomaly",
                TensorProto.INT64,
                [None, 1],
            ),
        ],
        initializer=initializers,
    )

    # Creates the entire model so we can export it using all the logic above
    onnx_model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 3),
        ],
        producer_name="CSC561-WeightedIsolationForest",
    )

    # Validate the ONNX graph.
    onnx.checker.check_model(onnx_model)

    # Save the model.
    onnx.save(onnx_model, str(model_file),)

    print(f"\nWeighted ONNX model exported successfully:\n {model_file}")

def main():
    print("\nLoading transaction datasets")

    real_df = load_feature_data(REAL_FILE)
    synthetic_df = load_feature_data(SYNTHETIC_FILE)

    print(f"Real transactions: {real_df.height}")
    print(f"Synthetic transactions: {synthetic_df.height}")

    raw_real_X = real_df.select(FEATURES).to_numpy()

    (train_idx, calibration_idx, test_idx,) = split_real_data( raw_real_X,RANDOM_STATE,)

    training_df = real_df[train_idx.tolist()]
    training_medians = {
        feature: training_df[feature].median()
        for feature in FEATURES
    }

    for feature, median in training_medians.items():
        if median is None or not np.isfinite(median):
            raise ValueError(f"No valid training values for {feature}.")

    real_df, X_real = prepare_features(real_df, training_medians,)

    synthetic_df, X_synthetic = prepare_features(synthetic_df, training_medians,)

    X_train = X_real[train_idx]
    X_calibration = X_real[calibration_idx]
    X_test = X_real[test_idx]

    print("\nReal data split:")

    print(f"Training: {len(X_train)}")
    print(f"Calibration: {len(X_calibration)}")
    print(f"Testing: {len(X_test)}")
    print(f"Synthetic evaluation: {len(X_synthetic)}")

    print("\nCalculating heuristic feature weights.")

    feature_weights = calculate_heuristic_weights(X_train)

    weighted_model = train_model(X_train, X_calibration, feature_weights,)

    print("\nEvaluating held-out real transactions")

    real_results = evaluate_dataset(
        real_df[test_idx.tolist()],
        X_test,
        "Real test",
        weighted_model,
    )

    print("\nEvaluating synthetic transactions")

    synthetic_results = evaluate_dataset(
        synthetic_df,
        X_synthetic,
        "Synthetic",
        weighted_model,
    )

    results = pl.concat([real_results, synthetic_results,])

    print("RESULTS")
    summary = summarize_results(results)

    results.write_csv(OUTPUT_FILE)
    print(f"\nTransaction results saved:\n{OUTPUT_FILE}")

    summary.write_csv(SUMMARY_FILE)
    print(f"\nSummary saved:\n{SUMMARY_FILE}")

    print("\n")
    print("TOP 10 WEIGHTED ANOMALIES")

    top_anomalies = results.sort("anomaly_score", descending=True,).head(10)
    print(top_anomalies)

    print("\nExporting weighted ONNX model...")

    export_onnx(weighted_model, WEIGHTED_MODEL_FILE,)


if __name__ == "__main__":
    main()