import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict(X)


def save_evaluation(model, X_test, y_test, model_name, output_dir, extra=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_score = positive_scores(model, X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision_yes": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall_yes": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "f1_yes": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_score)), 6),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["no", "yes"],
            output_dict=True,
            zero_division=0,
        ),
    }
    if extra:
        metrics.update(extra)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Deposit", "Deposit"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    return metrics


def save_feature_importance(model, output_dir, model_name, top_n=15):
    output_dir = Path(output_dir)
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["model"]

    if not hasattr(classifier, "feature_importances_"):
        return None

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    feature_importance = pd.DataFrame({"feature": feature_names, "importance": importances})
    feature_importance = feature_importance.sort_values("importance", ascending=False).head(top_n)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feature_importance["feature"][::-1], feature_importance["importance"][::-1])
    ax.set_title(f"Top {top_n} Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=180)
    plt.close(fig)

    return feature_importance
