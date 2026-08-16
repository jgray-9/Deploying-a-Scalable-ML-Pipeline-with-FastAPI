import pytest
from ml.model import compute_model_metrics, train_model, inference

def test_compute_model_metrics():
    """
    A test that computes model metrics on simple true labels and predicted labels.  This function tests
    the compute_model_metrics() function against known values for precision, recall, and fbeta.
    """
    y = [1, 1, 0, 0]
    preds = [1, 0, 1, 0]

    expected_precision = 0.5
    expected_recall = 0.5
    expected_fbeta = 0.5
    expected_metrics = expected_precision, expected_recall, expected_fbeta

    returned_precision, returned_recall, returned_fbeta = compute_model_metrics(y, preds)
    returned_metrics = returned_precision, returned_recall, returned_fbeta

    assert returned_metrics == expected_metrics

def test_train_model():
    """
    A test that trains the model with simple training sets.  The function tests that train_model()
    returns a model object.
    """
    X_train = [
        [32, 40],
        [46, 40],
        [19, 20],
        [24, 36]
    ]

    y_train = [1, 1, 0, 0]

    model_object = train_model(X_train, y_train)

    assert model_object is not None

def test_inference():
    """
    A test that uses the train_model() function to create a model to be used as input for 
    inference().  This function tests that inference() returns the correct number of predictions
    by using the provided model and a simple testing set.
    """
    X_train = [
        [32, 40],
        [46, 40],
        [19, 20],
        [24, 36]
    ]

    y_train = [1, 1, 0, 0]

    model = train_model(X_train, y_train)

    X = [
        [68, 40],
        [29, 40],
        [18, 16],
        [48, 34]
    ]

    preds = inference(model, X)

    assert len(preds) == len(X)
