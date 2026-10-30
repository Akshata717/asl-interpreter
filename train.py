import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
 
 
def normalize(pts):
    """Make landmarks independent of hand position and distance from camera.
    MUST be identical to the normalize() in predict.py."""
    pts = np.asarray(pts, dtype=float).reshape(21, 3)
    pts = pts - pts[0]                           
    scale = np.linalg.norm(pts[9]) + 1e-6       
    return (pts / scale).flatten()
 
 
df = pd.read_csv("dataset.csv", header=None)
X_raw = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X = np.array([normalize(r) for r in X_raw])
 
 
def augment(X, y, copies=4, mirror=True, seed=42):
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(copies):
        P = X.reshape(-1, 21, 3).copy()
        ang = np.radians(rng.uniform(-15, 15, size=(len(P), 1)))
        c, s = np.cos(ang), np.sin(ang)
        x, yy = P[:, :, 0].copy(), P[:, :, 1].copy()
        P[:, :, 0] = c * x - s * yy                        # small in-plane rotation
        P[:, :, 1] = s * x + c * yy
        P *= rng.uniform(0.9, 1.1, size=(len(P), 1, 1))    # size jitter
        P += rng.normal(0, 0.01, P.shape)                  # noise
        Xs.append(P.reshape(len(P), -1))
        ys.append(y)
    if mirror:
        M = X.reshape(-1, 21, 3).copy()
        M[:, :, 0] *= -1                                   # other hand
        Xs.append(M.reshape(len(M), -1))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)
 
 
# Per-letter chronological split (last 20% of each letter's rows = test)
train_idx, test_idx = [], []
for label in np.unique(y):
    idx = np.where(y == label)[0]
    cut = int(len(idx) * 0.8)
    train_idx += idx[:cut].tolist()
    test_idx += idx[cut:].tolist()
 
X_train, y_train = augment(X[train_idx], y[train_idx])
X_test, y_test = X[test_idx], y[test_idx]
 
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
 
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
print(classification_report(y_test, preds))
 
# Final model: retrain on everything
X_all, y_all = augment(X, y)
model.fit(X_all, y_all)
 
with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to asl_model.pkl")
 
