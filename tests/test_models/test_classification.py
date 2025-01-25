import pytest
from typing import Tuple
import numpy as np
import optuna
from ai_toolkit.models.classification import (
    LogisticRegressionModel,
    SVCModel,
    KNNModel,
    NaiveBayesModel,
    DecisionTreeModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    get_all_classification_models,
)


class TestLogisticRegression:
    """Test suite for LogisticRegression model."""

    def test_initialization(self):
        """Test model initialization."""

        model = LogisticRegressionModel()
        assert model.model_name == "Logistic Regression Classifier"
        assert model.model is None
        assert model.best_params is None

    def test_param_space(self, optuna_trial: optuna.trial.Trial):
        """Test parameter space generation.

        Args:
            optuna_trial (optuna.trial.Trial): An Optuna trial object.
        """

        model = LogisticRegressionModel()
        params = model.get_param_space(optuna_trial)

        assert "penalty" in params
        assert "solver" in params
        assert "C" in params
        assert "max_iter" in params
        assert "tol" in params
        assert "class_weight" in params
        assert "random_state" in params

        # Test elasticnet specific parameter
        if params["penalty"] == "elasticnet":
            assert "l1_ratio" in params

    def test_create_model(self, classification_data: Tuple[np.ndarray, np.ndarray]):
        """Test model creation and fitting.

        Args:
            classification_data (Tuple[np.ndarray, np.ndarray]):
                A tuple of features and target arrays.
        """

        X, y = classification_data
        model = LogisticRegressionModel()
        test_params = {
            "penalty": "l2",
            "solver": "saga",
            "C": 1.0,
            "max_iter": 1000,
            "tol": 1e-4,
            "class_weight": None,
            "random_state": 28,
        }

        clf = model.create_model(test_params)
        clf.fit(X, y)

        # Test predictions
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)
        assert np.all(np.unique(y_pred) == np.unique(y))


def test_get_all_classification_models():
    """Test get_all_classification_models function."""

    models = get_all_classification_models()

    expected_models = {
        "Logistic Regression Classifier": LogisticRegressionModel,
        "Support Vector Classifier": SVCModel,
        "K-Nearest Neighbors Classifier": KNNModel,
        "Naive Bayes Classifier": NaiveBayesModel,
        "Decision Tree Classifier": DecisionTreeModel,
        "Random Forest Classifier": RandomForestModel,
        "XGBoost Classifier": XGBoostModel,
        "LightGBM Classifier": LightGBMModel,
    }

    assert len(models) == len(expected_models)
    for name, model in expected_models.items():
        assert name in models
        assert isinstance(models[name], model)


def test_model_integration(classification_data: Tuple[np.ndarray, np.ndarray]):
    """Test complete workflow for all models.

    Args:
        classification_data (Tuple[np.ndarray, np.ndarray]):
            A tuple of features and target arrays.
    """

    X, y = classification_data
    models = get_all_classification_models()

    for name, model in models.items():
        # Create trial and get parameters
        study = optuna.create_study()
        trial = optuna.trial.Trial(
            study, study._storage.create_new_trial(study._study_id)
        )
        params = model.get_param_space(trial)

        # Create and fit model
        clf = model.create_model(params)
        try:
            clf.fit(X, y)

            # Test predictions
            y_pred = clf.predict(X)
            assert y_pred.shape == y.shape
            assert isinstance(y_pred, np.ndarray)
            # If the model does not converge, a class may not be predicted.
            # Therefore, no check is performed.
            # assert np.all(np.unique(y_pred) == np.unique(y))
            assert np.issubdtype(y_pred.dtype, np.number)  # Check numeric type

            # Test probability predictions if available
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X)
                assert y_prob.shape[0] == y.shape[0]
                assert y_prob.shape[1] == len(np.unique(y))

        except Exception as e:
            pytest.fail(f"Model {name} failed: {str(e)}")
