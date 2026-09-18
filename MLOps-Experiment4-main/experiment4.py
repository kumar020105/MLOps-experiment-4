import os
import tempfile

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 4
# Experiment Tracking and Model Management using MLflow
# Dataset: Iris
# ============================================================

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Iris MLflow Experiment"
MODEL_NAME = "IrisClassifier"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# 1. Load Iris Dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n============================================================")
print("EXPERIMENT 4 - MLOps WITH MLFLOW")
print("============================================================")

print("\nDataset: Iris")
print("Samples:", len(X))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 2. Evaluation Function
# ============================================================

def evaluate_model(model):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    return predictions, accuracy, precision, recall, f1


# ============================================================
# 3. Create and Log Artifacts
# ============================================================

def log_artifacts(model, model_name, predictions):

    with tempfile.TemporaryDirectory() as temp_dir:

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.colorbar()

        plt.xticks(
            range(len(iris.target_names)),
            iris.target_names
        )

        plt.yticks(
            range(len(iris.target_names)),
            iris.target_names
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        confusion_path = os.path.join(
            temp_dir,
            "confusion_matrix.png"
        )

        plt.savefig(confusion_path)
        plt.close()

        mlflow.log_artifact(confusion_path)

        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        report = classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
            zero_division=0
        )

        report_path = os.path.join(
            temp_dir,
            "classification_report.txt"
        )

        with open(report_path, "w") as file:
            file.write(
                f"Classification Report - {model_name}\n\n"
            )
            file.write(report)

        mlflow.log_artifact(report_path)

        # ----------------------------------------------------
        # Predictions CSV
        # ----------------------------------------------------

        prediction_df = pd.DataFrame({
            "actual_class": [
                iris.target_names[value]
                for value in y_test
            ],
            "predicted_class": [
                iris.target_names[value]
                for value in predictions
            ]
        })

        predictions_path = os.path.join(
            temp_dir,
            "predictions.csv"
        )

        prediction_df.to_csv(
            predictions_path,
            index=False
        )

        mlflow.log_artifact(predictions_path)

        # ----------------------------------------------------
        # Model Summary
        # ----------------------------------------------------

        summary_path = os.path.join(
            temp_dir,
            "model_summary.txt"
        )

        with open(summary_path, "w") as file:
            file.write(f"Model: {model_name}\n")
            file.write(f"Dataset: Iris\n")
            file.write(f"Training Samples: {len(X_train)}\n")
            file.write(f"Testing Samples: {len(X_test)}\n")

        mlflow.log_artifact(summary_path)

        # ----------------------------------------------------
        # Feature Importance for Tree Models
        # ----------------------------------------------------

        if hasattr(model, "feature_importances_"):

            importance_df = pd.DataFrame({
                "feature": iris.feature_names,
                "importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                "importance",
                ascending=False
            )

            importance_path = os.path.join(
                temp_dir,
                "feature_importance.csv"
            )

            importance_df.to_csv(
                importance_path,
                index=False
            )

            mlflow.log_artifact(importance_path)

            # Feature importance plot

            plt.figure(figsize=(8, 5))

            plt.bar(
                importance_df["feature"],
                importance_df["importance"]
            )

            plt.title(
                f"Feature Importance - {model_name}"
            )

            plt.xlabel("Feature")
            plt.ylabel("Importance")

            plt.xticks(rotation=30)
            plt.tight_layout()

            feature_plot_path = os.path.join(
                temp_dir,
                "feature_importance.png"
            )

            plt.savefig(feature_plot_path)
            plt.close()

            mlflow.log_artifact(feature_plot_path)


# ============================================================
# 4. Train Model and Track with MLflow
# ============================================================

def train_and_log_model(
    model,
    model_name,
    parameters
):

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        predictions, accuracy, precision, recall, f1 = (
            evaluate_model(model)
        )

        # Log parameters
        mlflow.log_params(parameters)

        # Log metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # Log model
        model_info = mlflow.sklearn.log_model(
            model,
            name="model"
        )

        # Log artifacts
        log_artifacts(
            model,
            model_name,
            predictions
        )

        print("\n------------------------------------------------------------")
        print(model_name)
        print("------------------------------------------------------------")

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))

        return {
            "run_id": run.info.run_id,
            "model": model,
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "model_uri": model_info.model_uri
        }


# ============================================================
# 5. Logistic Regression
# ============================================================

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

logistic_result = train_and_log_model(
    logistic_model,
    "Logistic Regression",
    {
        "model": "Logistic Regression",
        "max_iter": 1000
    }
)


# ============================================================
# 6. Decision Tree
# ============================================================

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree_result = train_and_log_model(
    decision_tree,
    "Decision Tree",
    {
        "model": "Decision Tree",
        "max_depth": 5,
        "random_state": 42
    }
)


# ============================================================
# 7. Random Forest
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

random_forest_result = train_and_log_model(
    random_forest,
    "Random Forest",
    {
        "model": "Random Forest",
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    }
)


# ============================================================
# 8. Support Vector Machine
# ============================================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale"
        )
    )
])

svm_result = train_and_log_model(
    svm_model,
    "Support Vector Machine",
    {
        "model": "Support Vector Machine",
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
    }
)


# ============================================================
# 9. Model Comparison
# ============================================================

results = [
    logistic_result,
    decision_tree_result,
    random_forest_result,
    svm_result
]

print("\n============================================================")
print("MODEL COMPARISON")
print("============================================================")

for result in results:

    print(
        f"{result['model_name']:<25}"
        f" Accuracy = {result['accuracy']:.4f}"
        f" | F1 = {result['f1_score']:.4f}"
    )

best_initial_model = max(
    results,
    key=lambda result: result["accuracy"]
)

print(
    "\nBest initial model:",
    best_initial_model["model_name"]
)


# ============================================================
# 10. Hyperparameter Tuning - Random Forest
# ============================================================

print("\n============================================================")
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("============================================================")

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run(
    run_name="Random Forest Hyperparameter Tuning"
) as tuning_run:

    grid_search.fit(
        X_train,
        y_train
    )

    tuned_model = grid_search.best_estimator_

    predictions, accuracy, precision, recall, f1 = (
        evaluate_model(tuned_model)
    )

    # Log best parameters
    mlflow.log_params(
        grid_search.best_params_
    )

    # Log CV score
    mlflow.log_metric(
        "best_cv_accuracy",
        grid_search.best_score_
    )

    # Log test metrics
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # Log tuned model
    tuned_model_info = mlflow.sklearn.log_model(
        tuned_model,
        name="tuned_model"
    )

    # Log artifacts
    log_artifacts(
        tuned_model,
        "Tuned Random Forest",
        predictions
    )

    tuning_run_id = tuning_run.info.run_id
    tuned_model_uri = tuned_model_info.model_uri

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation Accuracy:")
    print(round(grid_search.best_score_, 4))

    print("\nTuned Model Test Performance:")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# ============================================================
# 11. Model Registry
# ============================================================

print("\n============================================================")
print("MODEL REGISTRY")
print("============================================================")

client = MlflowClient()

# Register all four base models.
# This creates multiple model versions.

registered_results = []

for result in results:

    registered = mlflow.register_model(
        model_uri=result["model_uri"],
        name=MODEL_NAME
    )

    registered_results.append(
        registered
    )

    print(
        f"{result['model_name']}"
        f" -> Version {registered.version}"
    )


# Register tuned Random Forest

tuned_registered = mlflow.register_model(
    model_uri=tuned_model_uri,
    name=MODEL_NAME
)

print(
    "Tuned Random Forest"
    f" -> Version {tuned_registered.version}"
)


# ============================================================
# 12. Display Model Versions
# ============================================================

print("\nAvailable Model Versions:")

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

versions = sorted(
    versions,
    key=lambda version: int(version.version)
)

for version in versions:

    print(
        f"Version {version.version}"
        f" | Run ID: {version.run_id}"
    )


# ============================================================
# 13. Select Tuned Model for Production
# ============================================================

production_version = tuned_registered.version

client.set_registered_model_alias(
    MODEL_NAME,
    "production",
    production_version
)

print(
    f"\nProduction alias assigned to "
    f"Version {production_version}"
)


# ============================================================
# 14. Load Production Model
# ============================================================

production_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@production"
)

print(
    "\nProduction model loaded successfully."
)


# ============================================================
# 15. Production Inference
# ============================================================

unseen_samples = X_test[:5]
actual_values = y_test[:5]

production_predictions = (
    production_model.predict(
        unseen_samples
    )
)

print("\n============================================================")
print("PRODUCTION MODEL INFERENCE")
print("============================================================")

print(
    "Predicted:",
    [
        iris.target_names[value]
        for value in production_predictions
    ]
)

print(
    "Actual   :",
    [
        iris.target_names[value]
        for value in actual_values
    ]
)


correct_predictions = sum(
    production_predictions == actual_values
)

print(
    "\nCorrect predictions:",
    f"{correct_predictions}/{len(actual_values)}"
)


# ============================================================
# 16. Final Result
# ============================================================

print("\n============================================================")
print("EXPERIMENT 4 COMPLETED SUCCESSFULLY")
print("============================================================")

print("Experiment :", EXPERIMENT_NAME)
print("Models     : 4")
print("Tuning     : Random Forest GridSearchCV")
print("Registry   :", MODEL_NAME)
print(
    "Production : "
    f"Version {production_version}"
)
print("Artifacts  : Multiple artifacts logged")
print("Inference  : Completed")