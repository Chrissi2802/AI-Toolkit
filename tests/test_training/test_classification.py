import pytest
from typing import Tuple
import pandas as pd
import numpy as np
import mlflow
from unittest.mock import Mock, patch
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


@pytest.fixture(autouse=True)
def mlflow_cleanup():
    """Cleanup MLflow runs before and after each test."""

    mlflow.end_run()
    yield
    mlflow.end_run()


class TestClassificationModelTrainer:
    """Test suite for ClassificationModelTrainer."""

    def test_initialization(self, simple_model: LogisticRegressionModel):
        """Test trainer initialization.

        Args:
            simple_model (LogisticRegressionModel): A simple classification model.
        """

        trainer = ClassificationModelTrainer(
            base_model=simple_model,
            n_splits=5,
            random_state=28,
            experiment_name="test_classification",
            optimize_metric="f1",
            use_smote=True,
            smote_ratio=1.0,
        )

        assert trainer.base_model == simple_model
        assert trainer.n_splits == 5
        assert trainer.random_state == 28
        assert trainer.experiment_name == "test_classification"
        assert trainer.optimize_metric == "f1"
        assert trainer.use_smote is True
        assert trainer.smote_ratio == 1.0
        assert trainer.best_model is None
        assert trainer.best_score == -1

    @patch("mlflow.set_experiment")
    def test_training_workflow(
        self, mock_set_experiment, classification_data_pd, mock_mlflow
    ):
        """Test complete training workflow."""

        X, y = classification_data_pd

        trainer = ClassificationModelTrainer(
            base_model=LogisticRegressionModel(),
            n_splits=5,
            random_state=28,
            experiment_name="test_classification",
            optimize_metric="f1",
            use_smote=False,
        )

        # Train model
        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)

        # Check results
        assert best_model is not None
        assert isinstance(metrics, dict)
        assert all(
            metric in metrics
            for metric in ["accuracy", "balanced_acuracy", "precision", "recall", "f1"]
        )
        assert all(isinstance(value, float) for value in metrics.values())

        # Verify MLflow interactions
        assert mock_set_experiment.called
        assert mock_mlflow["run"].called
        assert mock_mlflow["log_params"].called
        assert mock_mlflow["log_metric"].called

    def test_prediction(self, classification_data_pd: Tuple[pd.DataFrame, pd.Series]):
        """Test prediction functionality.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
        """

        X, y = classification_data_pd
        trainer = ClassificationModelTrainer(
            base_model=LogisticRegressionModel(), n_splits=2, random_state=28
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
        assert set(predictions).issubset({0, 1})  # Binary classification

    def test_optimization_objective(
        self, classification_data_pd: Tuple[pd.DataFrame, pd.Series]
    ):
        """Test optimization objective function.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
        """

        X, y = classification_data_pd
        trainer = ClassificationModelTrainer(
            base_model=LogisticRegressionModel(),
            n_splits=2,
            random_state=28,
            optimize_metric="f1",
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

        # Test objective
        score = trainer._optimize_objective(mock_trial, X.values, y.values)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # F1 score range

    @pytest.mark.parametrize(
        "use_smote,smote_ratio", [(True, 1.0), (False, 1.0), (True, 0.7)]
    )
    def test_smote_integration(self, classification_data_pd, use_smote, smote_ratio):
        """Test SMOTE integration with different configurations."""

        X, y = classification_data_pd
        trainer = ClassificationModelTrainer(
            base_model=LogisticRegressionModel(),
            n_splits=2,
            random_state=28,
            use_smote=use_smote,
            smote_ratio=smote_ratio,
        )

        # Train model
        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
        assert best_model is not None
        assert isinstance(metrics, dict)

    def test_cross_validation_splits(
        self, classification_data_pd: Tuple[pd.DataFrame, pd.Series]
    ):
        """Test different cross-validation configurations.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
        """

        X, y = classification_data_pd
        n_splits_list = [3, 5, 10]

        for n_splits in n_splits_list:
            trainer = ClassificationModelTrainer(
                base_model=LogisticRegressionModel(), n_splits=n_splits, random_state=28
            )

            best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
            assert best_model is not None
            assert isinstance(metrics, dict)

    @pytest.mark.parametrize(
        "optimize_metric", ["accuracy", "f1", "precision", "recall"]
    )
    def test_different_optimization_metrics(
        self, classification_data_pd, optimize_metric
    ):
        """Test optimization with different metrics."""

        X, y = classification_data_pd
        trainer = ClassificationModelTrainer(
            base_model=LogisticRegressionModel(),
            n_splits=2,
            random_state=28,
            optimize_metric=optimize_metric,
        )

        best_model, metrics = trainer.train_and_optimize(X, y, n_trials=2)
        assert best_model is not None
        assert optimize_metric in metrics

    def test_error_handling(
        self, classification_data_pd: Tuple[pd.DataFrame, pd.Series]
    ):
        """Test error handling in trainer.

        Args:
            classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
                A tuple of features and target pandas objects.
        """

        X, y = classification_data_pd
        trainer = ClassificationModelTrainer(
            base_model=LogisticRegressionModel(), n_splits=2, random_state=28
        )

        # Test prediction without training
        with pytest.raises(ValueError):
            trainer.predict(X)

        # Test with invalid optimization metric
        trainer_invalid = ClassificationModelTrainer(
            base_model=LogisticRegressionModel(), optimize_metric="invalid_metric"
        )
        with pytest.raises(Exception):
            trainer_invalid.train_and_optimize(X, y, n_trials=2)


def test_lazypredict_classification(
    classification_data_pd: Tuple[pd.DataFrame, pd.Series]
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
def test_full_training_pipeline(classification_data_pd: Tuple[pd.DataFrame, pd.Series]):
    """Integration test for full training pipeline.

    Args:
        classification_data_pd (Tuple[pd.DataFrame, pd.Series]):
            A tuple of features and target pandas objects.
    """

    X, y = classification_data_pd

    # Create trainer with all features enabled
    trainer = ClassificationModelTrainer(
        base_model=LogisticRegressionModel(),
        n_splits=5,
        random_state=28,
        experiment_name="test_classification",
        optimize_metric="f1",
        use_smote=True,
        smote_ratio=1.0,
    )

    # Train model
    best_model, metrics = trainer.train_and_optimize(X, y, n_trials=3)

    # Make predictions
    predictions = trainer.predict(X)

    # Validate entire pipeline
    assert best_model is not None
    assert isinstance(metrics, dict)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X),)
    assert np.issubdtype(predictions.dtype, np.number)
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))
    assert set(predictions).issubset({0, 1})
    assert all(
        metric in metrics
        for metric in ["accuracy", "balanced_acuracy", "precision", "recall", "f1"]
    )
