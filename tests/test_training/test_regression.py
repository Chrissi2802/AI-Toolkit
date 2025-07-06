from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ai_toolkit.base.training import MlTrainerConfig
from ai_toolkit.models.regression import RidgeRegressionModel
from ai_toolkit.training.regression import (
    RegressionModelTrainer,
    lazypredict_regression,
)


@pytest.fixture
def simple_model() -> RidgeRegressionModel:
    """Create a simple Ridge regression model.

    Returns:
        RidgeRegressionModel: A simple Ridge regression model.
    """

    return RidgeRegressionModel()


class TestRegressionModelTrainer:
    """Test suite for RegressionModelTrainer."""

    def test_initialization(self, simple_model: RidgeRegressionModel):
        """Test trainer initialization.

        Args:
            simple_model (RidgeRegressionModel): A simple Ridge regression model.
        """

        trainer = RegressionModelTrainer(
            base_model=simple_model,
            config=MlTrainerConfig(
                EXPERIMENT_NAME="test_regression",
                OPTIMIZE_METRIC="root_mean_squared_error",
            ),
        )

        assert trainer.base_model is not None
        assert trainer.n_splits == 5
        assert trainer.random_state == 28
        assert trainer.experiment_name == "test_regression"
        assert trainer.optimize_metric == "root_mean_squared_error"
        assert trainer.best_model is None
        assert trainer.best_score == float("inf")

    @pytest.mark.integration
    def test_training_workflow(self, simple_model, regression_data_pd, mock_mlflow):
        """Test complete training workflow."""

        X, y = regression_data_pd

        trainer = RegressionModelTrainer(
            base_model=simple_model,
            config=MlTrainerConfig(
                EXPERIMENT_NAME="test_regression",
                OPTIMIZE_METRIC="root_mean_squared_error",
            ),
        )

        # Train model
        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)

        # Check results
        assert best_model is not None
        assert isinstance(metrics, dict)
        assert all(
            metric in metrics
            for metric in [
                "explained_variance",
                "max_error",
                "mean_absolute_error",
                "mean_squared_error",
                "root_mean_squared_error",
                "r2",
            ]
        )
        assert all(isinstance(value, float) for value in metrics.values())

        # Verfiy MLflow interactions
        assert mock_mlflow["run"].called
        assert mock_mlflow["log_params"].called
        assert mock_mlflow["log_metric"].called
        assert mock_mlflow["log_table"].called

    def test_prediction(self, regression_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model):
        """Test prediction functionality.

        Args:
            regression_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects
            simple_model (RidgeRegressionModel): A simple Ridge regression model.
        """

        X, y = regression_data_pd
        trainer = RegressionModelTrainer(
            base_model=simple_model,
            config=MlTrainerConfig(
                EXPERIMENT_NAME="test_regression",
                OPTIMIZE_METRIC="root_mean_squared_error",
            ),
        )

        # Train model first
        trainer.train_and_optimize(X, y, n_trials=2)

        # Make predictions
        predictions = trainer.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X),)
        assert np.issubdtype(predictions.dtype, np.number)  # Check numeric type
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_optimization_objective(
        self, regression_data_pd: Tuple[np.ndarray, np.ndarray], simple_model
    ):
        """Test optimization objective function.

        Args:
            regression_data (Tuple[np.ndarray, np.ndarray]): A tuple of features and target arrays
            simple_model (RidgeRegressionModel): A simple Ridge regression model.
        """

        X, y = regression_data_pd
        trainer = RegressionModelTrainer(
            base_model=simple_model,
            config=MlTrainerConfig(
                EXPERIMENT_NAME="test_regression",
                OPTIMIZE_METRIC="root_mean_squared_error",
            ),
        )

        # Create mock trial with correct return values
        mock_trial = Mock()

        # Dictionary for suggest_categorical with correct return values
        suggest_categorical_values = {
            "alpha": 10,
            "fit_intercept": True,
            "solver": "svd",
            "tol": 1e-4,
            "random_state": 28,
        }

        mock_trial.suggest_categorical.side_effect = (
            lambda name, choices: suggest_categorical_values[name]
        )
        mock_trial.suggest_float.return_value = 1.0
        mock_trial.suggest_categorical.return_value = True
        mock_trial.number = 1

        score = trainer._optimize_objective(mock_trial, X.values, y.values)
        assert isinstance(score, float)
        assert score > 0  # RMSE is always positive

    def test_cross_validation_splits(
        self, regression_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model
    ):
        """Test different cross-validation configurations.

        Args:
            regression_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects
            simple_model (RidgeRegressionModel): A simple Ridge regression model.
        """

        X, y = regression_data_pd
        n_splits_list = [3, 5, 10]

        for n_splits in n_splits_list:
            trainer = RegressionModelTrainer(
                base_model=simple_model,
                config=MlTrainerConfig(
                    EXPERIMENT_NAME="test_regression",
                    OPTIMIZE_METRIC="root_mean_squared_error",
                    N_SPLITS=n_splits,
                ),
            )

            best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
            assert best_model is not None
            assert isinstance(metrics, dict)

    @pytest.mark.parametrize(
        "optimize_metric",
        ["root_mean_squared_error", "mean_absolute_error", "explained_variance"],
    )
    def test_different_optimization_metrics(
        self, regression_data_pd, optimize_metric, simple_model
    ):
        """Test optimization with different metrics."""

        X, y = regression_data_pd
        trainer = RegressionModelTrainer(
            base_model=simple_model,
            config=MlTrainerConfig(
                EXPERIMENT_NAME="test_regression",
                OPTIMIZE_METRIC=optimize_metric,
            ),
        )

        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
        assert best_model is not None
        assert optimize_metric in metrics
        assert isinstance(metrics, dict)
        assert isinstance(metrics[optimize_metric], float)
        assert metrics[optimize_metric] > 0

    def test_error_handling(self, regression_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model):
        """Test error handling in trainer.

        Args:
            regression_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects
            simple_model (RidgeRegressionModel): A simple Ridge regression model.
        """

        X, y = regression_data_pd
        trainer = RegressionModelTrainer(
            base_model=simple_model,
            config=MlTrainerConfig(
                EXPERIMENT_NAME="test_regression",
                OPTIMIZE_METRIC="root_mean_squared_error",
            ),
        )

        # Test prediction without training
        with pytest.raises(RuntimeError, match="Prediction failed"):
            trainer.predict(X)

        # Test with invalid optimization metric
        with pytest.raises(ValueError):
            trainer_invalid = RegressionModelTrainer(
                base_model=simple_model,
                config=MlTrainerConfig(
                    EXPERIMENT_NAME="test_regression",
                    OPTIMIZE_METRIC="invalid_metric",
                ),
            )
            trainer_invalid.train_and_optimize(X, y, n_trials=2)


def test_lazypredict_regression(regression_data_pd: Tuple[pd.DataFrame, pd.Series]):
    """Test lazypredict_regression function.

    Args:
        regression_data_pd (Tuple[pd.DataFrame, pd.Series]):
            A tuple of features and target pandas objects
    """

    X, y = regression_data_pd

    # Run lazypredict
    results = lazypredict_regression(X, y, n_splits=5, random_state=28)

    # Validate results
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert all(
        col in results.columns
        for col in [
            "Adjusted R-Squared",
            "R-Squared",
            "RMSE",
            "Time Taken",
            "Adjusted R-Squared Std",
        ]
    )


@pytest.mark.integration
def test_full_training_pipeline(regression_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model):
    """Integration test for full regression pipeline.

    Args:
        regression_data_pd (Tuple[pd.DataFrame, pd.Series]):
            A tuple of features and target pandas objects
        simple_model (RidgeRegressionModel): A simple Ridge regression model.
    """

    X, y = regression_data_pd

    trainer = RegressionModelTrainer(
        base_model=simple_model,
        config=MlTrainerConfig(
            EXPERIMENT_NAME="test_regression",
            N_SPLITS=5,
            OPTIMIZE_METRIC="root_mean_squared_error",
        ),
    )

    # Train model
    best_model, metrics = trainer.train_and_optimize(X, y, n_trials=3)

    # Make predictions
    y_pred = trainer.predict(X)

    # Validate entire pipeline
    assert best_model is not None
    assert isinstance(metrics, dict)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (len(X),)
    assert np.issubdtype(y_pred.dtype, np.number)
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isinf(y_pred))
    assert all(
        metric in metrics
        for metric in [
            "explained_variance",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
            "r2",
        ]
    )
