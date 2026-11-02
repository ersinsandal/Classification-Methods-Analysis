import pandas as pd
import matplotlib.pyplot as plt

from config import METRICS_DIR, FIGURES_DIR, CONFUSION_MATRIX_DIR


def plot_metric_charts():
    file_path = METRICS_DIR / "classifier_summary_results.csv"
    df = pd.read_csv(file_path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "macro_f1"]

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            plt.bar(subset["classifier"], subset[metric])

            plt.title(f"{dataset} - {metric.upper()} Comparison")
            plt.xlabel("Classifier")
            plt.ylabel(metric.upper())
            plt.ylim(0, 1)

            plt.xticks(rotation=20)
            plt.tight_layout()

            save_path = FIGURES_DIR / f"{dataset}_{metric}.png"
            plt.savefig(save_path, dpi=200)
            plt.close()

            print(f"Saved: {save_path}")


def plot_time_charts():
    file_path = METRICS_DIR / "classifier_summary_results.csv"
    df = pd.read_csv(file_path)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    time_metrics = ["train_time", "test_time"]

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]

        for metric in time_metrics:
            plt.figure(figsize=(8, 5))
            plt.bar(subset["classifier"], subset[metric])

            plt.title(f"{dataset} - {metric.replace('_', ' ').title()} Comparison")
            plt.xlabel("Classifier")
            plt.ylabel("Seconds")

            plt.xticks(rotation=20)
            plt.tight_layout()

            save_path = FIGURES_DIR / f"{dataset}_{metric}.png"
            plt.savefig(save_path, dpi=200)
            plt.close()

            print(f"Saved: {save_path}")


def plot_confusion_matrix(cm, dataset_name, classifier_name, class_names):
    CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{dataset_name} - {classifier_name} Confusion Matrix")
    plt.colorbar()

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    save_path = CONFUSION_MATRIX_DIR / f"{dataset_name}_{classifier_name}_cm.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")