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


class WeightedIsolationForest:
    """
    Reusable Weighted Isolation Forest.

    Feature weights control the probability of selecting each
    feature when creating a random split.

    Higher anomaly scores indicate more unusual observations.
    """

    def __init__(self, n_estimators=200, max_samples=512, feature_weights=None, contamination=0.05, random_state=42, ):
        # Constructor for the tree has pre-set vals but allows user to define their own.
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feature_weights = feature_weights
        self.contamination = contamination
        self.random_state = random_state

        self.trees = []
        self.threshold = None
        self.sample_size = None
        self.n_features = None

    def fit(self, X):
        # This is the method for fitting the forest that allows us to take the data and then train the model

        X = np.asarray(X, dtype=float)

        # Validation checks for ensuring that it's a 2D array and that it is finite
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        if not np.isfinite(X).all():
            raise ValueError("Training data contains invalid values.")

        n_samples, n_features = X.shape

        # Get the number of samples from the features, so needs more than 1txn
        if n_samples < 2:
            raise ValueError("At least two observations are required.")
        # Needs features
        if n_features < 1:
            raise ValueError("At least one feature is required.")
        # At least one tree
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive.")
        # Must have multiple samples
        if self.max_samples < 2:
            raise ValueError("max_samples must be at least 2.")
        # Can't assume half of behavior is abnormal
        if not 0 < self.contamination < 0.5:
            raise ValueError("Contamination must be between 0 and 0.5.")

        # Default to equal feature weights.
        if self.feature_weights is None:
            self.feature_weights = (np.ones(n_features) / n_features)

        # get the feature weights
        self.feature_weights = np.asarray(self.feature_weights, dtype=float,).copy()

        if len(self.feature_weights) != n_features:
            raise ValueError("Feature weight count does not match feature count.")

        if (not np.isfinite(self.feature_weights).all() or np.any(self.feature_weights < 0) or self.feature_weights.sum() <= 0):
            raise ValueError("Invalid feature weights.")

        # Normalize weights.
        self.feature_weights /= self.feature_weights.sum()

        self.n_features = n_features
        self.sample_size = min(self.max_samples, n_samples)

        max_depth = int(np.ceil(np.log2(self.sample_size)))

        rng = np.random.default_rng(self.random_state)

        self.trees = []

        print("\nBuilding weighted isolation trees")

        # Loop through the estimators and then create isolation trees to create the forest
        for i in range(self.n_estimators):
            sample_indices = rng.choice(n_samples, size=self.sample_size, replace=False,)

            sample = X[sample_indices]

            # Create the isolation trees
            tree = WeightedIsolationTree(
                feature_weights=self.feature_weights,
                max_depth=max_depth,
                random_state=int(
                    rng.integers(0, 2**32 - 1)
                ),
            )

            tree.fit(sample)

            self.trees.append(tree)

            if (i + 1) % 25 == 0:
                print(f"Built {i + 1}/{self.n_estimators} trees.")

        self.threshold = None
        print("Weighted forest training complete.")

        return self

    def score_samples(self, X):
        # Get the scores for the trees based on the depth after training

        X = np.asarray(X, dtype=float)

        # Validation to ensure the data and training fulfill expectations
        if not self.trees:
            raise ValueError("Model has not been trained.")

        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        if X.shape[1] != self.n_features:
            raise ValueError("Input feature count does not match training data.")

        if not np.isfinite(X).all():
            raise ValueError("Input data contains invalid values.")

        # Validation for too small sample size
        if self.sample_size <= 1:
            return np.full(len(X), 0.5)

        path_lengths = np.zeros(len(X))

        # Calculate the path lengths for each tree in the feature
        for tree in self.trees:
            for i, observation in enumerate(X):
                path_lengths[i] += tree.path_length(observation)

        # Get the average length and normalize to a 0 to 1 scale
        average_paths = path_lengths / len(self.trees)
        normalization = average_path_length(self.sample_size)

        if normalization == 0:
            return np.full(len(X), 0.5)

        # Higher score = more anomalous.
        return np.power(2.0, -average_paths / normalization,)

    def calibrate(self, X_calibration):
        """
        Set the anomaly threshold using a separate calibration dataset.

        The calibration data should be real observations that were
        not used to train the forest.
        """

        scores = self.score_samples(X_calibration)

        # Finds the 95th percentile score in the calibration dataset of anomaly scores
        self.threshold = float( np.quantile(scores, 1.0 - self.contamination,) )
        print(f"Anomaly threshold calibrated: {self.threshold:.6f}")

        return self

    def predict(self, X):

        # Validation for if there's no calibration
        if self.threshold is None:
            raise ValueError("Model threshold has not been calibrated.")

        # gets the scores for the inputs
        scores = self.score_samples(X)

        # 1 = anomaly, 0 = normal. If above the threshold (95th percentile they're anomaly)
        return (scores >= self.threshold).astype(int)

    def fit_predict(self, X):

        """
        Convenience method for training and predicting.

        Uses the training observations for calibration, so for a
        proper held-out evaluation prefer fit() followed by
        calibrate() on a separate dataset.
        """

        self.fit(X)
        self.calibrate(X)

        return self.predict(X)