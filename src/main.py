from pathlib import Path
import numpy as np
import polars as pl
import onnx
from onnx import helper, TensorProto

# Load files
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

INPUT_FILE = DATA_DIR / "weighted_transactions.csv"

# Get output files
OUTPUT_FILE = DATA_DIR / "weighted_isolation_forest_results.csv"
MODEL_FILE = MODEL_DIR / "weighted_isolation_forest.onnx"

# Feature Array
FEATURES = [
    "compute_units",
    "total_fee_lamports",
    "account_count",
    "instruction_count",
    "inner_instruction_count",
    "token_balance_change_count",
]

# Hyperparams
N_ESTIMATORS = 200
MAX_SAMPLES = 256
CONTAMINATION = 0.6
RANDOM_STATE = 42


def average_path_length(n):
    """Expected path length of an unsuccessful search in a binary search tree.
    Used to normalize Isolation Forest scores. """
    if n <= 1:
        return 0.0

    if n == 2:
        return 1.0

    harmonic_number = (
        np.log(n - 1)
        + 0.5772156649
        + (1.0 / (2.0 * (n - 1)))
        - (1.0 / (12.0 * (n - 1) ** 2))
    )

    return (2.0 * harmonic_number) - (2.0 * (n - 1) / n)


# Use the weights to infulence branching

# Get a single binary-tree
class WeightedIsolationTree:

    def __init__(self, feature_weights, max_depth, random_state=None):
        self.feature_weights = np.asarray(feature_weights, dtype=float)
        self.feature_weights = np.maximum(self.feature_weights, 0.0)
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)
        self.nodes = []

    # build the treee

    def fit(self, X):
        indices = np.arange(len(X))
        self.nodes = []
        self._build_tree(X, indices, depth=0)
        return self

    def _build_tree(self, X, indices, depth):
        node_id = len(self.nodes)
        self.nodes.append(None)

        n_samples = len(indices)

        if n_samples <= 1 or depth >= self.max_depth:
            self.nodes[node_id] = {"feature": None, "size": n_samples}
            return node_id

        subset = X[indices]

        # Find features with variation
        feature_ranges = np.ptp(subset, axis=0)
        available_features = np.flatnonzero(feature_ranges > 0)

        # All features are constant
        if len(available_features) == 0:
            self.nodes[node_id] = {"feature": None, "size": n_samples}
            return node_id

        # Get heuristic weights for available features
        selection_weights = self.feature_weights[available_features]

        # Normalize probabilities
        if selection_weights.sum() <= 0:
            probabilities = np.full(
                len(available_features), 1.0 / len(available_features)
            )
        else:
            probabilities = selection_weights / selection_weights.sum()

        # Feature weighted split selection
        selected_feature = self.rng.choice(available_features, p=probabilities)

        # Random split between minimum and maximum
        feature_values = subset[:, selected_feature]

        minimum = feature_values.min()
        maximum = feature_values.max()

        split_value = self.rng.uniform(minimum, maximum)

        left_mask = feature_values < split_value

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        # Safety fallback if numerical precision produces an empty partition
        if len(left_indices) == 0 or len(right_indices) == 0:
            sorted_values = np.unique(feature_values)

            if len(sorted_values) < 2:
                self.nodes[node_id] = {"feature": None, "size": n_samples}
                return node_id

            split_value = (sorted_values[0] + sorted_values[1]) / 2.0

            left_mask = feature_values < split_value

            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]

        # Recursively build children
        left_child = self._build_tree(X, left_indices, depth + 1)
        right_child = self._build_tree(X, right_indices, depth + 1)

        # Store split node
        self.nodes[node_id] = {
            "feature": int(selected_feature),
            "split": float(split_value),
            "left": left_child,
            "right": right_child,
            "size": n_samples,
        }

        return node_id

    # Calculate the path length to determine if it's going to be anomaly or not
    def path_length(self, observation):
        node_id = 0
        depth = 0

        while True:
            node = self.nodes[node_id]

            # Leaf node
            if node["feature"] is None:
                return depth + average_path_length(node["size"])

            feature = node["feature"]
            split_value = node["split"]

            if observation[feature] < split_value:
                node_id = node["left"]
            else:
                node_id = node["right"]

            depth += 1


# Create the weighted isolation forest
class WeightedIsolationForest:

    def __init__(
        self,
        n_estimators=200,
        max_samples=256,
        feature_weights=None,
        contamination=0.05,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feature_weights = feature_weights
        self.contamination = contamination
        self.random_state = random_state
        self.trees = []
        self.threshold = None
        self.sample_size = None

    # train
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("At least two observations are required.")

        if self.feature_weights is None:
            self.feature_weights = np.ones(n_features) / n_features

        self.feature_weights = np.asarray(self.feature_weights, dtype=float)

        if len(self.feature_weights) != n_features:
            raise ValueError("Feature weight count does not match feature count.")

        if (not np.isfinite(self.feature_weights).all() or np.any(self.feature_weights < 0) or self.feature_weights.sum() <= 0):
            raise ValueError("Feature weights must be finite, nonnegative, and have a positive sum.")

        # Normalize weights
        self.feature_weights = self.feature_weights / self.feature_weights.sum()

        self.sample_size = min(self.max_samples, n_samples)
        max_depth = int(np.ceil(np.log2(self.sample_size)))

        rng = np.random.default_rng(self.random_state)
        self.trees = []

        print("\nBuilding weighted isolation trees...")

        for i in range(self.n_estimators):
            # Random subsample without replacement
            sample_indices = rng.choice(
                n_samples, size=self.sample_size, replace=False
            )
            sample = X[sample_indices]

            tree = WeightedIsolationTree(
                feature_weights=self.feature_weights,
                max_depth=max_depth,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )

            tree.fit(sample)
            self.trees.append(tree)

            if (i + 1) % 25 == 0:
                print(f"Built {i + 1}/{self.n_estimators} trees.")

        print("Forest training complete.")

        # Calculate training scores to establish the anomaly threshold
        training_scores = self.score_samples(X)
        self.threshold = np.quantile(training_scores, self.contamination)

        return self

    # Anomaly scores
    def score_samples(self, X):
        X = np.asarray(X, dtype=float)

        if not self.trees:
            raise ValueError("Model has not been trained.")

        if self.sample_size <= 1:
            return np.full(len(X), 0.5)

        path_lengths = np.zeros(len(X))

        for tree in self.trees:
            for i, observation in enumerate(X):
                path_lengths[i] += tree.path_length(observation)

        average_paths = path_lengths / len(self.trees)
        normalization = average_path_length(self.sample_size)

        if normalization == 0:
            return np.full(len(X), 0.5)

        # Higher score = more anomalous
        anomaly_scores = np.power(2.0, -average_paths / normalization)
        return anomaly_scores

    # predictions
    def predict(self, X):
        scores = self.score_samples(X)

        # 1 = anomaly, 0 = normal
        return (scores >= self.threshold).astype(int)


# Load data
def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pl.read_csv(INPUT_FILE)

    # Convert features to numeric
    df = df.with_columns([
        pl.col(feature).cast(pl.Float64, strict=False).fill_nan(None)
        for feature in FEATURES
    ])

    # Fill missing values with feature medians
    medians = {}

    for feature in FEATURES:
        median_value = df[feature].median()

        if median_value is None:
            raise ValueError(f"Feature {feature} contains no valid values.")

        medians[feature] = median_value

    df = df.with_columns([
        pl.col(feature).fill_null(medians[feature]) for feature in FEATURES
    ])

    X = df.select(FEATURES).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Dataset contains infinite or invalid values.")

    print("\nDataset loaded.")
    print(f"Observations: {df.height}")

    return df, X

# Load the heuristic weights
def load_feature_weights(df):
    weights = []
    print("\nLoading heuristic feature weights:")

    for feature in FEATURES:
        weight_column = f"{feature}_weight"

        if weight_column not in df.columns:
            raise ValueError(
                f"Missing weight column: {weight_column}. Run MedianWeights.py first."
            )

        weight = df[weight_column].cast(pl.Float64, strict=False).mean()

        if weight is None or not np.isfinite(weight):
            raise ValueError(f"Invalid weight for {feature}")

        weights.append(weight)

    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(
            "Feature weights must be nonnegative and have a positive sum."
        )

    # Normalize to probabilities
    weights = weights / weights.sum()

    print("\nFeature selection probabilities:")
    for feature, weight in zip(FEATURES, weights):
        print(f"{feature}: {weight:.4f} ({weight * 100:.2f}%)")

    return weights


# Save model to ONNX
def export_onnx(model):
    """Exports our trained  to ONNX format so it can be run efficiently in other environments"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

    # Convert to format ONNX's TreeEnsembleRegressor expects.
    for tree_id, tree in enumerate(model.trees):
        for node_id, node in enumerate(tree.nodes):
            nodes_treeids.append(tree_id)
            nodes_nodeids.append(node_id)

            if node["feature"] is None:
                nodes_featureids.append(0)
                nodes_modes.append("LEAF")
                nodes_values.append(0.0)

                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)
                nodes_missing_value_tracks_true.append(0)

                leaf_path = calculate_leaf_depth(tree.nodes, node_id) + average_path_length(node["size"])

                target_treeids.append(tree_id)
                target_nodeids.append(node_id)
                target_ids.append(0)
                target_weights.append(float(leaf_path))
            else:
                nodes_featureids.append(node["feature"])
                nodes_modes.append("BRANCH_LT")
                nodes_values.append(node["split"])
                nodes_truenodeids.append(node["left"])
                nodes_falsenodeids.append(node["right"])
                nodes_missing_value_tracks_true.append(0)

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
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        target_treeids=target_treeids,
        target_nodeids=target_nodeids,
        target_ids=target_ids,
        target_weights=target_weights,
    )

    # Grab our normalization constant
    normalization = average_path_length(model.sample_size)

    if normalization <= 0:
        raise ValueError("Invalid path-length normalization.")

    initializers = [
        helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0]),
        helper.make_tensor("normalization", TensorProto.FLOAT, [1], [float(normalization)]),
        helper.make_tensor("threshold", TensorProto.FLOAT, [1], [float(model.threshold)]),
    ]

    # Scoring pipeline: Score = 2 ^ (-average_path / normalization)
    scoring_nodes = [
        helper.make_node("Neg", ["average_path_length"], ["negative_path"]),
        helper.make_node("Div", ["negative_path", "normalization"], ["exponent"]),
        helper.make_node("Pow", ["two", "exponent"], ["anomaly_score"]),
        # Compare score against threshold (1 = anomaly, 0 = normal)
        helper.make_node("GreaterOrEqual", ["anomaly_score", "threshold"], ["anomaly_boolean"]),
        helper.make_node("Cast", ["anomaly_boolean"], ["is_anomaly"], to=TensorProto.INT64),
    ]

    graph = helper.make_graph(
        nodes=[tree_ensemble, *scoring_nodes],
        name="WeightedIsolationForest",
        inputs=[
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(FEATURES)])
        ],
        outputs=[
            helper.make_tensor_value_info("anomaly_score", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("is_anomaly", TensorProto.INT64, [None, 1]),
        ],
        initializer=initializers,
    )

    onnx_model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.onnx.ml", 3),
        ],
        producer_name="CSC561-WeightedIsolationForest",
    )

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(MODEL_FILE))

    print(f"\nONNX model exported successfully: {MODEL_FILE}")


def calculate_leaf_depth(nodes, target_node):
    """Helper to find how deep a specific leaf sits within the tree structure."""

    def search(node_id, depth):
        node = nodes[node_id]

        if node_id == target_node:
            return depth

        if node["feature"] is None:
            return None

        left_result = search(node["left"], depth + 1)
        if left_result is not None:
            return left_result

        return search(node["right"], depth + 1)

    result = search(0, 0)

    if result is None:
        raise ValueError("Could not determine leaf depth.")

    return result

def calculate_leaf_depth(nodes, target_node):
    def search(node_id, depth):
        node = nodes[node_id]
        if node_id == target_node:
            return depth
        if node["feature"] is None:
            return None
        left_result = search(node["left"], depth + 1)
        if left_result is not None:
            return left_result
        return search(node["right"], depth + 1)

    result = search(0, 0)
    if result is None:
        raise ValueError("Could not determine leaf depth.")
    return result


def calculate_leaf_depth(nodes, target_node):

    def search(node_id, depth):
        node = nodes[node_id]

        if node_id == target_node:
            return depth

        if node["feature"] is None:
            return None

        left_result = search(node["left"], depth + 1)

        if left_result is not None:
            return left_result

        return search(node["right"], depth + 1)

    result = search(0, 0)

    if result is None:
        raise ValueError("Could not determine leaf depth.")

    return result

def save_results(df, predictions, scores):
    results = df.with_columns([
        pl.Series("is_anomaly", predictions),
        pl.Series("anomaly_score", scores),
    ])

    results.write_csv(OUTPUT_FILE)
    print(f"\nResults saved: {OUTPUT_FILE}")

def display_results(predictions, scores):
    total = len(predictions)
    anomaly_count = int(np.sum(predictions == 1))
    normal_count = total - anomaly_count

    print(f"Total transactions: {total}")
    print(f"Normal: {normal_count}")
    print(f"Anomalies: {anomaly_count}")
    print(f"Anomaly percentage: {(anomaly_count / total) * 100:.2f}%")
    print(f"Mean anomaly score: {np.mean(scores):.4f}")
    print(f"Minimum anomaly score: {np.min(scores):.4f}")
    print(f"Maximum anomaly score: {np.max(scores):.4f}")



def main():
    # Load transaction data
    df, X = load_data()

    # Load heuristic weights
    feature_weights = load_feature_weights(df)

    # Initialize weighted forest
    model = WeightedIsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        feature_weights=feature_weights,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )

    # Train
    model.fit(X)

    # Predict
    predictions = model.predict(X)
    scores = model.score_samples(X)

    # Display results
    display_results(predictions, scores)

    # Save model
    export_onnx(model)

    # Save predictions
    save_results(df, predictions, scores)


if __name__ == "__main__":
    main()