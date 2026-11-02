import pandas as pd

from config import DATASETS, METRICS_DIR
from src.data_loader import load_dataset
from src.features import extract_hog_features
from src.classifiers import get_classifiers
from src.train_eval import train_and_evaluate_classifier
from src.visualization import plot_confusion_matrix


def run_single_dataset_experiments(dataset_name):
    print(f"\n==============================")
    print(f"DATASET: {dataset_name}")
    print(f"==============================")

    x_train, y_train, x_test, y_test, class_names = load_dataset(dataset_name)

    print("Extracting HOG features...")
    X_train = extract_hog_features(x_train)
    X_test = extract_hog_features(x_test)

    print("Train HOG shape:", X_train.shape)
    print("Test HOG shape:", X_test.shape)

    classifiers = get_classifiers()
    results = []

    for classifier_name, classifier in classifiers.items():
        print(f"\n--- {dataset_name} | {classifier_name} başladı ---")

        result = train_and_evaluate_classifier(
            classifier,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        plot_confusion_matrix(
            cm=result["confusion_matrix"],
            dataset_name=dataset_name,
            classifier_name=classifier_name,
            class_names=class_names,
        )

        print("Accuracy:", result["accuracy"])
        print("Precision:", result["precision"])
        print("Recall:", result["recall"])
        print("Macro F1:", result["macro_f1"])
        print("Train time:", result["train_time"])
        print("Test time:", result["test_time"])

        results.append(
            {
                "dataset": dataset_name,
                "feature_method": "hog",
                "classifier": classifier_name,
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "macro_f1": result["macro_f1"],
                "train_time": result["train_time"],
                "test_time": result["test_time"],
                "train_feature_shape": str(X_train.shape),
                "test_feature_shape": str(X_test.shape),
            }
        )

    return results


def run_all_experiments():
    all_results = []

    for dataset_name in DATASETS:
        dataset_results = run_single_dataset_experiments(dataset_name)
        all_results.extend(dataset_results)

    results_df = pd.DataFrame(all_results)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / "classifier_summary_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nTüm deneyler tamamlandı.")
    print(f"Sonuç dosyası kaydedildi: {output_path}")
    print("\n", results_df)

    return results_df