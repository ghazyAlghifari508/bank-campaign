from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from common.evaluation import save_evaluation
from common.preprocessing import build_svm_preprocessor, split_features_target


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "bank.csv"
MODEL_DIR = BASE_DIR / "models" / "svm"
OUTPUT_DIR = BASE_DIR / "outputs" / "svm"


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
            ("preprocess", build_svm_preprocessor()),
            ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
        ]
    )
    non_hpo.fit(X_train, y_train)
    joblib.dump(non_hpo, MODEL_DIR / "non_hpo.pkl")
    non_metrics = save_evaluation(
        non_hpo,
        X_test,
        y_test,
        "SVM Non-HPO",
        OUTPUT_DIR / "non_hpo",
        extra={"method": "non_hpo", "notebook_style": "SVM notebook"},
    )

    hpo_pipeline = Pipeline(
        steps=[
            ("preprocess", build_svm_preprocessor()),
            ("model", SVC(random_state=42)),
        ]
    )
    param_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto", 0.01, 0.1],
        "model__kernel": ["rbf", "linear", "poly"],
        "model__degree": [2, 3, 4],
    }
    grid = GridSearchCV(
        estimator=hpo_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    clean_best_params = {key.replace("model__", ""): value for key, value in grid.best_params_.items()}
    hpo = Pipeline(
        steps=[
            ("preprocess", build_svm_preprocessor()),
            ("model", SVC(**clean_best_params, probability=True, random_state=42)),
        ]
    )
    hpo.fit(X_train, y_train)
    joblib.dump(hpo, MODEL_DIR / "hpo.pkl")
    hpo_metrics = save_evaluation(
        hpo,
        X_test,
        y_test,
        "SVM HPO",
        OUTPUT_DIR / "hpo",
        extra={
            "method": "hpo",
            "notebook_style": "SVM notebook",
            "best_params": grid.best_params_,
            "best_cv_score": round(float(grid.best_score_), 6),
        },
    )

    print("SVM training selesai.")
    print("Non-HPO Accuracy:", non_metrics["accuracy"])
    print("HPO Accuracy:", hpo_metrics["accuracy"])
    print("Best HPO Params:", grid.best_params_)


if __name__ == "__main__":
    main()
