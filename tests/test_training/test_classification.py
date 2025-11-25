from typing import Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ai_toolkit.base.config import ConfigFactory
from ai_toolkit.models.classification import LogisticRegressionModel
from ai_toolkit.training.classification import (
    ClassificationModelTrainer,
    lazypredict_classification,
)


@pytest.fixture
def simple_model() -> LogisticRegressionModel:
    """Create a simple classification model for testing.

    Returns:
        LogisticRegressionModel: A simple classification model.
    """

    return LogisticRegressionModel()


class TestClassificationModelTrainer:
    """Test suite for ClassificationModelTrainer."""

    def test_initialization(self, simple_model: LogisticRegressionModel):
        """Test trainer initialization.

        Args:
            simple_model (LogisticRegressionModel): A simple model instance.
        """

        config_factory = ConfigFactory()
        config_factory.training.experiment_name = "test_classification"

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            config_factory=config_factory,
        )

        assert trainer.base_model is not None
        assert trainer.n_splits == 5
        assert trainer.random_state == 28
        assert trainer.experiment_name == "test_classification"
        assert trainer.optimize_metric == "f1"
        assert trainer.use_smote is True
        assert trainer.smote_ratio == 1.0
        assert trainer.best_model is None
        assert trainer.best_score == float("-inf")

    @pytest.mark.integration
    def test_training_workflow(self, simple_model, classification_data_pd, mock_mlflow):
        """Test complete training workflow."""

        X, y = classification_data_pd

        config_factory = ConfigFactory()
        config_factory.training.experiment_name = "test_classification"

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            config_factory=config_factory,
        )

        # Train model
        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)

        # Check results
        assert best_model is not None
        assert isinstance(metrics, dict)
        assert all(
            metric in metrics
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
        )
        assert all(isinstance(value, float) for value in metrics.values())

        # Verify MLflow interactions
        assert mock_mlflow["run"].called
        assert mock_mlflow["log_params"].called
        assert mock_mlflow["log_metric"].called
        assert mock_mlflow["log_table"].called

    def test_prediction(self, classification_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model):
        """Test prediction functionality.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
            simple_model (LogisticRegressionModel): A simple model instance.
        """

        X, y = classification_data_pd

        config_factory = ConfigFactory()
        config_factory.training.experiment_name = "test_classification"

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            config_factory=config_factory,
        )

        # Train model first
        trainer.train_and_optimize(X, y, n_trials=2)

        # Make predictions
        y_pred, y_pred_proba = trainer.predict(X)
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(y_pred_proba, np.ndarray)
        assert y_pred.shape == (len(X),)
        assert y_pred_proba.shape == (len(X), 2)  # Binary classification
        assert np.issubdtype(y_pred.dtype, np.number)  # Check numeric type
        assert not np.any(np.isnan(y_pred))
        assert not np.any(np.isinf(y_pred))
        assert set(y_pred).issubset({0, 1})  # Binary classification
        assert np.all(np.sum(y_pred_proba, axis=1) == 1.0)  # Probabilities sum to 1.0

    def test_optimization_objective(
        self, classification_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model
    ):
        """Test optimization objective function.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
            simple_model (LogisticRegressionModel): A simple model instance.
        """

        X, y = classification_data_pd
        config_factory = ConfigFactory()
        config_factory.training.experiment_name = "test_classification"
        config_factory.training.use_smote = False

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            config_factory=config_factory,
        )

        # Create mock trial with correct return values
        mock_trial = Mock()

        # Dictionary for suggest_categorical with correct return values
        suggest_categorical_values = {
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": None,
            "random_state": 28,
        }

        mock_trial.suggest_categorical.side_effect = (
            lambda name, choices: suggest_categorical_values[name]
        )
        mock_trial.suggest_float.return_value = 1.0
        mock_trial.suggest_int.return_value = 100
        mock_trial.number = 1

        # Test objective
        score = trainer._optimize_objective(mock_trial, X.values, y.values)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # F1 score range

    @pytest.mark.parametrize("use_smote,smote_ratio", [(True, 1.0), (False, 1.0), (True, 0.7)])
    def test_smote_integration(self, classification_data_pd, use_smote, smote_ratio, simple_model):
        """Test SMOTE integration with different configurations."""

        X, y = classification_data_pd

        config_factory = ConfigFactory()
        config_factory.training.experiment_name = "test_classification"
        config_factory.training.use_smote = use_smote
        config_factory.training.smote_ratio = smote_ratio

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            config_factory=config_factory,
        )

        # Train model
        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
        assert best_model is not None
        assert isinstance(metrics, dict)

    def test_cross_validation_splits(
        self, classification_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model
    ):
        """Test different cross-validation configurations.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
            simple_model (LogisticRegressionModel): A simple model instance.
        """

        X, y = classification_data_pd
        n_splits_list = [3, 5, 10]

        for n_splits in n_splits_list:
            config_factory = ConfigFactory()
            config_factory.training.experiment_name = "test_classification"
            config_factory.training.n_splits = n_splits

            trainer = ClassificationModelTrainer(
                base_model=simple_model,
                config_factory=config_factory,
            )

            best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
            assert best_model is not None
            assert isinstance(metrics, dict)

    @pytest.mark.parametrize("optimize_metric", ["accuracy", "f1", "precision", "recall"])
    def test_different_optimization_metrics(
        self, classification_data_pd, optimize_metric, simple_model
    ):
        """Test optimization with different metrics."""

        X, y = classification_data_pd

        config_factory = ConfigFactory()
        config_factory.training.experiment_name = "test_classification"
        config_factory.training.optimize_metric = optimize_metric

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            config_factory=config_factory,
        )

        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
        assert best_model is not None
        assert optimize_metric in metrics
        assert isinstance(metrics, dict)
        assert isinstance(metrics[optimize_metric], float)
        assert 0 <= metrics[optimize_metric] <= 1

    def test_error_handling(
        self,
        classification_data_pd: Tuple[pd.DataFrame, pd.Series],
        simple_model,
    ):
        """Test error handling in trainer.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
            simple_model (LogisticRegressionModel): A simple model instance.
        """

        X, y = classification_data_pd

        config_factory = ConfigFactory()
        config_factory.training.experiment_name = "test_classification"

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            config_factory=config_factory,
        )

        # Test prediction without training
        with pytest.raises(RuntimeError, match="Prediction failed"):
            trainer.predict(X)

        # Test with invalid optimization metric
        with pytest.raises(ValueError):
            config_factory = ConfigFactory()
            config_factory.training.experiment_name = "test_classification"
            config_factory.training.optimize_metric = "invalid_metric"

            trainer_invalid = ClassificationModelTrainer(
                base_model=simple_model,
                config_factory=config_factory,
            )

            trainer_invalid.train_and_optimize(X, y, n_trials=2)


def test_lazypredict_classification(
    classification_data_pd: Tuple[pd.DataFrame, pd.Series],
):
    """Test lazypredict_classification function.

    Args:
        classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
            A tuple of features and target pandas objects.
    """

    X, y = classification_data_pd

    # Run lazypredict
    results = lazypredict_classification(X, y, n_splits=5, random_state=28)

    # Validate results
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert all(
        col in results.columns
        for col in [
            "Accuracy",
            "Balanced Accuracy",
            "ROC AUC",
            "F1 Score",
            "Time Taken",
            "Accuracy Std",
        ]
    )


@pytest.mark.integration
def test_full_training_pipeline(
    classification_data_pd: Tuple[pd.DataFrame, pd.Series], simple_model
):
    """Integration test for full training pipeline.

    Args:
        classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
            A tuple of features and target pandas objects.
        simple_model (LogisticRegressionModel): A simple model instance.
    """

    X, y = classification_data_pd

    config_factory = ConfigFactory()
    config_factory.training.experiment_name = "test_classification"
    config_factory.training.n_splits = 5
    config_factory.training.optimize_metric = "f1"
    config_factory.training.use_smote = True
    config_factory.training.smote_ratio = 1.0

    # Create trainer with all features enabled
    trainer = ClassificationModelTrainer(
        base_model=simple_model,
        config_factory=config_factory,
    )

    # Train model
    best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)

    # Make predictions
    y_pred, y_pred_proba = trainer.predict(X)

    # Validate entire pipeline
    assert best_model is not None
    assert isinstance(metrics, dict)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_pred_proba, np.ndarray)
    assert y_pred.shape == (len(X),)
    assert y_pred_proba.shape == (len(X), 2)
    assert np.issubdtype(y_pred.dtype, np.number)
    assert not np.any(np.isnan(y_pred))
    assert not np.any(np.isinf(y_pred))
    assert set(y_pred).issubset({0, 1})
    assert np.all(np.sum(y_pred_proba, axis=1) == 1.0)
    assert all(
        metric in metrics
        for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    )
