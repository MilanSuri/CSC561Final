import numpy as np
from sklearn.ensemble import IsolationForest

X_inliers = np.random.normal(loc=0.0, scale=1.0, size=(100, 2))
X_outliers = np.random.uniform(low=-10, high=10, size=(5, 2))
X = np.vstack([X_inliers, X_outliers])

clf = IsolationForest(contamination=0.05, random_state=42)

clf.fit(X)
predictions = clf.predict(X)

print("Predictions (-1 indicates an outlier):")
print(predictions)
