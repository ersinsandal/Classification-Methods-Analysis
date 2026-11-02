import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    """
    Verilen classifier'ı eğitir, test eder ve metrikleri döndürür.
    """

    start_train = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_test = time.time()
    y_pred = classifier.predict(X_test)
    test_time = time.time() - start_test

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "train_time": train_time,
        "test_time": test_time,
        "confusion_matrix": cm,
    }