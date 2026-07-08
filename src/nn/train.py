from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from common.evaluation import save_evaluation
from common.preprocessing import build_knn_nn_preprocessor, split_features_target


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "bank.csv"
MODEL_DIR = BASE_DIR / "models" / "nn"
OUTPUT_DIR = BASE_DIR / "outputs" / "nn"


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
            ("preprocess", build_knn_nn_preprocessor()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(100,),
                    activation="relu",
                    solver="adam",
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )
    non_hpo.fit(X_train, y_train)
    joblib.dump(non_hpo, MODEL_DIR / "non_hpo.pkl")
    non_metrics = save_evaluation(
        non_hpo,
        X_test,
        y_test,
        "Neural Network Non-HPO",
        OUTPUT_DIR / "non_hpo",
        extra={"method": "non_hpo", "notebook_style": "NN notebook"},
    )

    hpo_pipeline = Pipeline(
        steps=[
            ("preprocess", build_knn_nn_preprocessor()),
            ("model", MLPClassifier(random_state=42)),
        ]
    )
    param_grid = [
        {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50, 25)],
            "model__max_iter": [200, 300, 350],
            "model__activation": ["relu", "tanh"],
            "model__solver": ["adam", "sgd"],
        }
    ]
    grid = GridSearchCV(
        hpo_pipeline,
        param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    hpo = grid.best_estimator_
    joblib.dump(hpo, MODEL_DIR / "hpo.pkl")
    hpo_metrics = save_evaluation(
        hpo,
        X_test,
        y_test,
        "Neural Network HPO",
        OUTPUT_DIR / "hpo",
        extra={
            "method": "hpo",
            "notebook_style": "NN notebook",
            "best_params": grid.best_params_,
            "best_cv_score": round(float(grid.best_score_), 6),
        },
    )

    print("Neural Network training selesai.")
    print("Non-HPO Accuracy:", non_metrics["accuracy"])
    print("HPO Accuracy:", hpo_metrics["accuracy"])
    print("Best HPO Params:", grid.best_params_)


if __name__ == "__main__":
    main()
