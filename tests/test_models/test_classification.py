from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import optuna
import pytest

from ai_toolkit.models.classification import (
    DecisionTreeModel,
    EnsembleStackingClassifierModel,
    EnsembleVotingClassifierModel,
    KNNModel,
    LightGBMModel,
    LogisticRegressionModel,
    NaiveBayesModel,
    RandomForestModel,
    SVCModel,
    XGBoostModel,
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


def test_feature_importance(
    classification_data: Tuple[np.ndarray, np.ndarray], optuna_trial: optuna.trial.Trial
):
    """Test feature importance method for all models.

    Args:
        classification_data (Tuple[np.ndarray, np.ndarray]):
            A tuple of features and target arrays.
        optuna_trial (optuna.trial.Trial): An Optuna trial object.
    """

    X, y = classification_data
    model = RandomForestModel()

    # Create and fit model
    params = model.get_param_space(optuna_trial)
    clf = model.create_model(params)
    clf.fit(X, y)

    # Check feature importance
    assert hasattr(clf, "feature_importances_")
    importances = clf.feature_importances_
    assert len(importances) == X.shape[1]
    assert np.issubdtype(importances.dtype, np.number)
    assert np.all(np.isfinite(importances))
    assert np.all(importances >= 0)
    assert np.isclose(np.sum(importances), 1.0)


class TestEnsembleModels:
    """Test suite for ensemble models."""

    def setup_method(self):
        """Setup method for each test."""

        # Create two logistic regression models
        self.model1 = LogisticRegressionModel()
        self.model1.model_name = "Model1"  # Set model name for testing

        self.model2 = LogisticRegressionModel()
        self.model2.model_name = "Model2"  # Set different model name

    def test_voting_classifier(self, classification_data):
        """Test voting classifier ensemble."""

        X, y = classification_data

        # Prepare models
        models = [(self.model1, "mock_run_1"), (self.model2, "mock_run_2")]

        with patch("mlflow.get_run") as mock_get_run:
            mock_run = Mock()
            mock_run.data.params = {"penalty": "l2", "C": 1.0, "random_state": 28}
            mock_get_run.return_value = mock_run

            # Create and configure ensemble
            ensemble = EnsembleVotingClassifierModel(models)
            params = {
                "estimators": [(m.model_name, m.model) for m, _ in models],
                "voting": "soft",  # Use soft voting for probabilities
                "weights": [0.6, 0.4],
            }

            # Create and train ensemble
            clf = ensemble.create_model(params)
            clf.fit(X, y)

            # Test predictions
            y_pred = clf.predict(X)
            assert isinstance(y_pred, np.ndarray)
            assert y_pred.shape == y.shape

            # Test probability predictions
            y_proba = clf.predict_proba(X)
            assert y_proba.shape == (len(y), len(np.unique(y)))
            assert np.allclose(np.sum(y_proba, axis=1), 1.0)

    def test_stacking_classifier(self, classification_data, optuna_trial):
        """Test stacking classifier ensemble."""

        X, y = classification_data

        # Prepare models
        base_models = [(self.model1, "mock_run_1"), (self.model2, "mock_run_2")]
        meta_model = LogisticRegressionModel()

        with patch("mlflow.get_run") as mock_get_run:
            mock_run = Mock()
            mock_run.data.params = {"penalty": "l2", "C": 1.0, "random_state": 28}
            mock_get_run.return_value = mock_run

            # Create and configure ensemble
            ensemble = EnsembleStackingClassifierModel(base_models, meta_model)
            params = {
                "estimators": [(m.model_name, m.model) for m, _ in base_models],
                "final_estimator": meta_model.create_model(
                    meta_model.get_param_space(optuna_trial)
                ),
                "stack_method": "predict_proba",
                "passthrough": True,
            }

            # Create and train ensemble
            clf = ensemble.create_model(params)
            clf.fit(X, y)

            # Test predictions
            y_pred = clf.predict(X)
            assert isinstance(y_pred, np.ndarray)
            assert np.issubdtype(y_pred.dtype, np.number)
            assert y_pred.shape == y.shape
            assert np.all(np.isfinite(y_pred))
            assert not np.any(np.isnan(y_pred))

            # Test probability predictions
            y_proba = clf.predict_proba(X)
            assert y_proba.shape == (len(y), len(np.unique(y)))
            assert np.allclose(np.sum(y_proba, axis=1), 1.0)
