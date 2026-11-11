import numpy as np

def average_path_length(n):
    """Expected path length used to normalize Isolation Forest scores."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0

    harmonic_number = (np.log(n - 1) + 0.5772156649 + 1.0 / (2.0 * (n - 1)) - 1.0 / (12.0 * (n - 1) ** 2))
    return 2.0 * harmonic_number - 2.0 * (n - 1) / n

