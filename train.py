"""
Train the Iris classifier and save it as model.pkl.

Run:
    python train.py

Uses the same scikit-learn version pinned in requirements.txt, so the saved
model loads cleanly inside the Docker container / on Render.
"""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test accuracy: {acc:.3f}")

joblib.dump(model, "model.pkl")
print("Saved model.pkl")
