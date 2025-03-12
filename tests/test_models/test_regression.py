from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import optuna
import pytest

from ai_toolkit.models.regression import (
    BayesianRidgeRegressionModel,
    CatBoostRegressorModel,
    EnsembleStackingRegressorModel,
    EnsembleVotingRegressorModel,
    KNNRegressorModel,
    LightGBMRegressorModel,
    RidgeRegressionModel,
    SVRModel,
    XGBoostRegressorModel,
    get_all_regression_models,
)


class TestRidgeRegression:
    """Test suite for Ridge Regression model."""

    def test_initialization(self):
        """Test model initialization."""

        model = RidgeRegressionModel()
        assert model.model_name == "Ridge Regressor"
        assert model.model is None
        assert model.best_params is None

    def test_param_space(self, optuna_trial: optuna.trial.Trial):
        """Test parameter space generation.

        Args:
            optuna_trial (optuna.trial.Trial): An Optuna trial object.
        """

        model = RidgeRegressionModel()
        params = model.get_param_space(optuna_trial)

        assert "alpha" in params
        assert "fit_intercept" in params
        assert "solver" in params
        assert "tol" in params
        assert "random_state" in params

    def test_create_model(self, regression_data: Tuple[np.ndarray, np.ndarray]):
        """Test model creation and fitting.

        Args:
            regression_data (Tuple[np.ndarray, np.ndarray]): A tuple of features and target arrays.
        """

        X, y = regression_data
        model = RidgeRegressionModel()
        test_params = {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "auto",
            "tol": 1e-4,
            "random_state": 28,
        }

        reg = model.create_model(test_params)
        reg.fit(X, y)

        # Test predictions
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape
        assert isinstance(y_pred, np.ndarray)
        assert not np.any(np.isnan(y_pred))


def test_get_all_regression_models():
    """Test get_all_regression_models function."""

    models = get_all_regression_models()

    expected_models = {
        "Ridge Regressor": RidgeRegressionModel,
        "Bayesian Ridge Regressor": BayesianRidgeRegressionModel,
        "Support Vector Regressor": SVRModel,
        "K-Nearest Neighbors Regressor": KNNRegressorModel,
        "XGBoost Regressor": XGBoostRegressorModel,
        "LightGBM Regressor": LightGBMRegressorModel,
        "CatBoost Regressor": CatBoostRegressorModel,
    }

    assert len(models) == len(expected_models)
    for name, model in expected_models.items():
        assert name in models
        assert isinstance(models[name], model)


def test_model_integration(regression_data: Tuple[np.ndarray, np.ndarray]):
    """Test complete workflow for all regression models.

    Args:
        regression_data (Tuple[np.ndarray, np.ndarray]): A tuple of features and target arrays.
    """

    X, y = regression_data
    models = get_all_regression_models()

    for name, model in models.items():
        # Create trial and get parameters
        study = optuna.create_study()
        trial = optuna.trial.Trial(
            study, study._storage.create_new_trial(study._study_id)
        )
        params = model.get_param_space(trial)

        # Create and fit model
        try:
            reg = model.create_model(params)
            reg.fit(X, y)

            # Test predictions
            y_pred = reg.predict(X)
            assert y_pred.shape == y.shape
            assert isinstance(y_pred, np.ndarray)
            assert not np.any(np.isnan(y_pred))

            # Additional regression-specific checks
            assert np.all(np.isfinite(y_pred))  # Check for inf values
            assert np.issubdtype(y_pred.dtype, np.number)  # Check numeric type

        except Exception as e:
            pytest.fail(f"Model {name} failed: {str(e)}")


def test_feature_importance(
    regression_data: Tuple[np.ndarray, np.ndarray], optuna_trial: optuna.trial.Trial
):
    """Test feature importance method for all regression models.

    Args:
        regression_data (Tuple[np.ndarray, np.ndarray]): A tuple of features and target arrays.
        optuna_trial (optuna.trial.Trial): An Optuna trial object.
    """

    X, y = regression_data
    model = RidgeRegressionModel()

    # Create and fit model
    params = model.get_param_space(optuna_trial)
    reg = model.create_model(params)
    reg.fit(X, y)

    # Check feature importance
    assert hasattr(reg, "coef_")
    importances = reg.coef_
    assert len(importances) == X.shape[1]
    assert np.issubdtype(importances.dtype, np.number)
    assert np.all(np.isfinite(importances))


class TestEnsembleModels:
    """Test suite for ensemble models."""

    def setup_method(self):
        """Setup method for each test."""

        # Create two Ridge regression models
        self.model1 = RidgeRegressionModel()
        self.model1.model_name = "Model1"

        self.model2 = RidgeRegressionModel()
        self.model2.model_name = "Model2"

    def test_voting_regressor(self, regression_data):
        """Test voting regressor ensemble."""

        X, y = regression_data

        # Prepare models
        models = [(self.model1, "mock_run_1"), (self.model2, "mock_run_2")]

        with patch("mlflow.get_run") as mock_get_run:
            mock_run = Mock()
            mock_run.data.params = {
                "alpha": 1.0,
                "fit_intercept": True,
                "random_state": 28,
            }
            mock_get_run.return_value = mock_run

            # Create and configure ensemble
            ensemble = EnsembleVotingRegressorModel(models)
            params = {
                "estimators": [(m.model_name, m.model) for m, _ in models],
                "weights": [0.6, 0.4],
            }

            # Create and train ensemble
            reg = ensemble.create_model(params)
            reg.fit(X, y)

            # Test predictions
            y_pred = reg.predict(X)
            assert isinstance(y_pred, np.ndarray)
            assert y_pred.shape == y.shape

    def test_stacking_regressor(self, regression_data, optuna_trial):
        """Test stacking regressor ensemble."""

        X, y = regression_data

        # Prepare models
        base_models = [(self.model1, "mock_run_1"), (self.model2, "mock_run_2")]
        meta_model = RidgeRegressionModel()

        with patch("mlflow.get_run") as mock_get_run:
            mock_run = Mock()
            mock_run.data.params = {
                "alpha": 1.0,
                "fit_intercept": True,
                "random_state": 28,
            }
            mock_get_run.return_value = mock_run

            # Create and configure ensemble
            ensemble = EnsembleStackingRegressorModel(base_models, meta_model)
            params = {
                "estimators": [(m.model_name, m.model) for m, _ in base_models],
                "final_estimator": meta_model.create_model(
                    meta_model.get_param_space(optuna_trial)
                ),
                "passthrough": True,
            }

            # Create and train ensemble
            reg = ensemble.create_model(params)
            reg.fit(X, y)

            # Test predictions
            y_pred = reg.predict(X)
            assert isinstance(y_pred, np.ndarray)
            assert np.issubdtype(y_pred.dtype, np.number)
            assert y_pred.shape == y.shape
            assert np.all(np.isfinite(y_pred))
            assert not np.any(np.isnan(y_pred))
