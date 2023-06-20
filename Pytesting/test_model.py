
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import hypothesis.strategies as st
from hypothesis import given, settings
import pytest
import stroke_model 


def test_data_split():
    # Generate sample data for testing
    np.random.seed(0)
    x = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = stroke_model.train_test_sets(x, y, test_size=0.2)

    # Assertions to check the sizes and properties of the train and test sets
    assert len(x_train) > 0, "Empty training set"
    assert len(x_test) > 0, "Empty test set"
    assert len(x_train) == len(y_train), "Size mismatch between x_train and y_train"
    assert len(x_test) == len(y_test), "Size mismatch between x_test and y_test"
    assert set(np.unique(y_train)) == set(np.unique(y_test)), "Different class labels in train and test sets"


@given(y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=1))
@settings(max_examples=10)
def test_accuracy_score(y_true):
    # Generate random predictions of the same length as y_true
    y_pred = np.random.randint(0, 2, len(y_true))

    # Use accuracy_score from sklearn's accuracy_score
    sklearn_acc = stroke_model.accuracy(y_true, y_pred)

    # Calculate accuracy manually
    manual_acc = np.sum(y_true == y_pred) / len(y_true)

    # Assert that the accuracy scores match
    assert np.isclose(sklearn_acc, manual_acc)
