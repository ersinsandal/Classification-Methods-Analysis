from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_classifiers():
    """
    Tüm classifierları dictionary olarak döner
    """

    classifiers = {
        "svm": SVC(kernel="linear"),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300),
    }

    return classifiers