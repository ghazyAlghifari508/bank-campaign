from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from common.evaluation import save_evaluation, save_feature_importance
from common.preprocessing import split_features_target


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "bank.csv"
MODEL_DIR = BASE_DIR / "models" / "decision_tree"
OUTPUT_DIR = BASE_DIR / "outputs" / "decision_tree"


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


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    non_hpo = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )
    non_hpo.fit(X_train, y_train)
    joblib.dump(non_hpo, MODEL_DIR / "non_hpo.pkl")
    non_metrics = save_evaluation(
        non_hpo,
        X_test,
        y_test,
        "Decision Tree Non-HPO",
        OUTPUT_DIR / "non_hpo",
        extra={"method": "non_hpo"},
    )
    save_feature_importance(non_hpo, OUTPUT_DIR / "non_hpo", "Decision Tree Non-HPO")

    hpo_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
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
    joblib.dump(hpo, MODEL_DIR / "hpo.pkl")
    hpo_metrics = save_evaluation(
        hpo,
        X_test,
        y_test,
        "Decision Tree HPO",
        OUTPUT_DIR / "hpo",
        extra={
            "method": "hpo",
            "best_params": grid.best_params_,
            "best_cv_score": round(float(grid.best_score_), 6),
        },
    )
    save_feature_importance(hpo, OUTPUT_DIR / "hpo", "Decision Tree HPO")

    print("Decision Tree training selesai.")
    print("Non-HPO Accuracy:", non_metrics["accuracy"])
    print("HPO Accuracy:", hpo_metrics["accuracy"])
    print("Best HPO Params:", grid.best_params_)


if __name__ == "__main__":
    main()
