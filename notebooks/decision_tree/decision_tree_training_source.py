from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "bank.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def save_evaluation(model, X_test, y_test, model_name: str, suffix: str, extra=None):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision_yes": round(float(precision_score(y_test, y_pred)), 6),
        "recall_yes": round(float(recall_score(y_test, y_pred)), 6),
        "f1_yes": round(float(f1_score(y_test, y_pred)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 6),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["no", "yes"], output_dict=True
        ),
        "note": "Model memakai fitur duration untuk mencapai performa 80-an.",
    }
    if extra:
        metrics.update(extra)

    with open(OUTPUT_DIR / f"metrics_{suffix}.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Deposit", "Deposit"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_matrix_{suffix}.png", dpi=180)
    plt.close(fig)

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(15)
    fi.to_csv(OUTPUT_DIR / f"feature_importance_{suffix}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1])
    ax.set_title(f"Top 15 Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"feature_importance_{suffix}.png", dpi=180)
    plt.close(fig)

    return metrics


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["deposit"])
    y = (df["deposit"] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    non_hpo = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )
    non_hpo.fit(X_train, y_train)
    joblib.dump(non_hpo, MODEL_DIR / "decision_tree_non_hpo.pkl")
    non_metrics = save_evaluation(non_hpo, X_test, y_test, "Decision Tree Non-HPO", "non_hpo")

    hpo_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [6, 8, 10, 12],
        "model__min_samples_split": [2, 20],
        "model__min_samples_leaf": [5, 10, 20],
        "model__class_weight": [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=hpo_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    hpo = grid.best_estimator_
    joblib.dump(hpo, MODEL_DIR / "decision_tree_hpo.pkl")
    hpo_metrics = save_evaluation(
        hpo,
        X_test,
        y_test,
        "Decision Tree HPO",
        "hpo",
        extra={
            "best_params": grid.best_params_,
            "best_cv_score": round(float(grid.best_score_), 6),
        },
    )

    print("Training selesai.")
    print("Non-HPO Accuracy:", non_metrics["accuracy"])
    print("HPO Accuracy:", hpo_metrics["accuracy"])
    print("Best HPO Params:", grid.best_params_)


if __name__ == "__main__":
    main()
