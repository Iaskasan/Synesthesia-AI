#!/usr/bin/env python3
"""
Simple classifier for song mood/genre.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def train_classifier(X, y):
    """
    Train a logistic regression classifier.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.

    Returns:
        model: Trained sklearn model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model


# TODO: support more models (RandomForest, Neural Networks, etc.)
