import numpy as np

def average_path_length(n):
    """Expected path length used to normalize Isolation Forest scores."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0

    harmonic_number = (np.log(n - 1) + 0.5772156649 + 1.0 / (2.0 * (n - 1)) - 1.0 / (12.0 * (n - 1) ** 2))
    return 2.0 * harmonic_number - 2.0 * (n - 1) / n


class WeightedIsolationTree:
    """A single Isolation Tree with weighted feature selection."""

    def __init__(self, feature_weights, max_depth, random_state=None):
        # Constructor for the weighted isolation class takes in depth, weights, rng
        self.feature_weights = np.asarray(feature_weights, dtype=float)
        self.max_depth = max_depth
        self.rng = np.random.default_rng(random_state)

        self.nodes = []


    def _build_tree(self, X, indices, depth):
        # Recursively builds the trees for one feature per node

        # Creates a new node in the node array
        node_id = len(self.nodes)
        self.nodes.append(None)

        # How many txns have made it to this node
        n_samples = len(indices)

        # Validation to prevent splitting if only one sample left or if greater than max depth
        if n_samples <= 1 or depth >= self.max_depth:
            self.nodes[node_id] = {"feature": None, "size": n_samples,}
            return node_id

        # Looks at the rows for the sample subset
        subset = X[indices]

        # Gets the max to min for each feature and checks for variance
        feature_ranges = np.ptp(subset, axis=0)
        available_features = np.flatnonzero(feature_ranges > 0)

        # Validation in case no variance
        if len(available_features) == 0:
            self.nodes[node_id] = {"feature": None, "size": n_samples,}
            return node_id

        # Get the weights for each feature
        selection_weights = self.feature_weights[available_features]

        # Normalizes so the weights sum to 1.0
        if selection_weights.sum() <= 0:
            probabilities = np.full( len(available_features), 1.0 / len(available_features), )
        else:
            probabilities = ( selection_weights / selection_weights.sum() )

        # Uses the probabilities to determine a feature to split on
        selected_feature = self.rng.choice( available_features, p=probabilities, )

        # Isolates all the vals of that feature and grabs max/min and where to split on
        feature_values = subset[:, selected_feature]
        minimum = feature_values.min()
        maximum = feature_values.max()
        split_value = self.rng.uniform(minimum, maximum)

        left_mask = feature_values < split_value
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        # Handle an empty partition.
        if len(left_indices) == 0 or len(right_indices) == 0:
            sorted_values = np.unique(feature_values)

            if len(sorted_values) < 2:
                self.nodes[node_id] = {"feature": None, "size": n_samples,}
                return node_id

            split_value = ( sorted_values[0] + sorted_values[1] ) / 2.0

            left_mask = feature_values < split_value

            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]

        # Recursion
        left_child = self._build_tree(X, left_indices, depth + 1,)
        right_child = self._build_tree(X, right_indices, depth + 1,)

        # Build the node
        self.nodes[node_id] = {
            "feature": int(selected_feature),
            "split": float(split_value),
            "left": left_child,
            "right": right_child,
            "size": n_samples,
        }

        return node_id

    def fit(self, X):
        # Method for fitting the isolation tree to the data
        indices = np.arange(len(X))
        self.nodes = []
        self._build_tree(X, indices, depth=0)
        return self

    def path_length(self, observation):
        # Get the path length for a specific node by traversal of tree
        node_id = 0
        depth = 0

        while True:
            node = self.nodes[node_id]

            if node["feature"] is None:
                return depth + average_path_length(node["size"])

            feature = node["feature"]
            split_value = node["split"]

            if observation[feature] < split_value:
                node_id = node["left"]
            else:
                node_id = node["right"]

            depth += 1