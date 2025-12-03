"""Classifier training and evaluation."""
import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


def train_classifier(X_train, y_train, method='knn', **kwargs):
    """Train classifier on training data."""
    if method == 'knn':
        n_neighbors = kwargs.get('n_neighbors', 5)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    elif method == 'svm':
        C = kwargs.get('C', 1.0)
        kernel = kwargs.get('kernel', 'rbf')
        clf = SVC(C=C, kernel=kernel, probability=True, gamma='scale')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test):
    """Evaluate classifier and return accuracy, confusion matrix, predictions."""
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm, y_pred


def save_classifier(clf, filepath):
    """Save trained classifier to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(clf, f)


def load_classifier(filepath):
    """Load trained classifier from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
