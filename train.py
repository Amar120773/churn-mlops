import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("churn.csv")

# Convert bool to int
df = df.astype(int)

# -----------------------------
# 2. Split Data
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# 3. Model Training + MLflow
# -----------------------------
best_model = None
best_f1 = 0
best_run_id = None

# -------- Logistic Regression --------
with mlflow.start_run(run_name="Logistic Regression") as run:
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("Logistic Regression -> Accuracy:", acc, "F1:", f1)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_run_id = run.info.run_id


# -------- Random Forest --------
with mlflow.start_run(run_name="Random Forest") as run:
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("Random Forest -> Accuracy:", acc, "F1:", f1)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_run_id = run.info.run_id


# -----------------------------
# 4. Register Best Model
# -----------------------------
if best_run_id is not None:
    model_uri = f"runs:/{best_run_id}/model"
    
    mlflow.register_model(
        model_uri=model_uri,
        name="ChurnPredictionModel"
    )

    print("\nBest model registered in MLflow Model Registry!")
    print("Best F1 Score:", best_f1)


# -----------------------------
# 5. Save Model for Deployment
# -----------------------------
import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/model.pkl")

print("\nModel saved successfully for deployment!")