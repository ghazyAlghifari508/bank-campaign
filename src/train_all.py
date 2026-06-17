from decision_tree.train import main as train_decision_tree
from knn.train import main as train_knn
from nn.train import main as train_nn
from svm.train import main as train_svm


def main():
    trainers = [
        ("Decision Tree", train_decision_tree),
        ("KNN", train_knn),
        ("Neural Network", train_nn),
        ("SVM", train_svm),
    ]

    for model_name, trainer in trainers:
        print(f"\n=== Training {model_name} ===")
        trainer()

    print("\nSemua training selesai.")


if __name__ == "__main__":
    main()
