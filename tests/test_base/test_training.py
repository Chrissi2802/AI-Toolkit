from operator import gt, lt
from types import SimpleNamespace
from typing import Any

import numpy as np
import optuna
import pytest

from ai_toolkit.base.config import ConfigFactory, get_default_metric_configs
from ai_toolkit.base.models import BaseMlModel
from ai_toolkit.base.training import (
    BaseMlTrainer,
)


class DummyTrainer(BaseMlTrainer):
    """Dummy trainer implementation for testing."""

    def __init__(
        self, base_model: BaseMlModel, config_factory: ConfigFactory = ConfigFactory()
    ) -> None:
        """Initialize dummy trainer.

        Args:
            base_model (BaseMlModel): Base model to use.
            config_factory (ConfigFactory, optional):
                Training configuration. Defaults to ConfigFactory().
        """
        super().__init__(base_model=base_model, config_factory=config_factory)

    def _optimize_objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Implement optimization objective.

        Args:
            trial (optuna.Trial): Optuna trial object.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            float: Objective score.
        """

        return 0.5  # Dummy score


def test_base_trainer_initialization(dummy_model: Any) -> None:
    """Test BaseMlTrainer initialization."""

    trainer = DummyTrainer(base_model=dummy_model)

    trainer.config.n_splits = 2
    trainer.config.random_state = 42
    trainer.config.experiment_name = "test_classification"
    trainer.optimize_metric = "f1"

    trainer.n_splits = trainer.config.n_splits
    trainer.random_state = trainer.config.random_state
    trainer.experiment_name = trainer.config.experiment_name
    trainer.optimize_metric = trainer.config.optimize_metric

    assert trainer.base_model == dummy_model
    assert trainer.n_splits == 2
    assert trainer.random_state == 42
    assert trainer.experiment_name == "test_classification"
    assert trainer.optimize_metric == "f1"
    assert trainer.best_model is None
    assert trainer.feature_names is None


def test_log_training_info(dummy_model: Any, mock_mlflow: Any) -> None:
    """Test training info logging.

    Args:
        mock_mlflow: MLflow mock fixture.
    """

    trainer = DummyTrainer(base_model=dummy_model)

    trainer._log_training_info(n_trials=100)

    mock_mlflow["log_params"].assert_called_once()
    params = mock_mlflow["log_params"].call_args[0][0]
    assert params["model_name"] == "Dummy Model"
    assert params["n_splits"] == 5  # Default value
    assert params["n_trials"] == 100


def test_log_dataset_info(dummy_model: Any, mock_mlflow: Any) -> None:
    """Test dataset info logging.

    Args:
        mock_mlflow: MLflow mock fixture.
    """

    trainer = DummyTrainer(base_model=dummy_model)

    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    feature_names = ["feature1", "feature2"]
    trainer.feature_names = feature_names

    trainer._log_dataset_info(X, y)

    mock_mlflow["log_params"].assert_called_once()
    params = mock_mlflow["log_params"].call_args[0][0]
    assert params["n_samples"] == 2
    assert params["n_features"] == 2


def test_log_fold_results(dummy_model: Any, mock_mlflow: Any) -> None:
    """Test fold results logging.

    Args:
        mock_mlflow: MLflow mock fixture.
    """

    trainer = DummyTrainer(base_model=dummy_model)

    metrics = {"accuracy": 0.9, "f1": 0.85}
    trainer._log_fold_results(fold=1, metrics=metrics)

    expected_calls = [
        ("accuracy_fold_1", 0.9),
        ("f1_fold_1", 0.85),
    ]

    mock_mlflow["log_metric"].assert_called()
    actual_calls = [args for args, _ in mock_mlflow["log_metric"].call_args_list]
    assert actual_calls == expected_calls


def test_calc_feature_importance(dummy_model: Any) -> None:
    """Test feature importance calculation."""

    trainer = DummyTrainer(base_model=dummy_model)

    # Test with feature_importances_ attribute
    importance_array = np.array([0.7, 0.3])
    model_with_importance = SimpleNamespace(feature_importances_=importance_array)
    importance = trainer._calc_feature_importance(model_with_importance)
    assert importance is not None
    np.testing.assert_array_almost_equal(importance, importance_array)

    # Test with coef_ attribute (single target)
    coef_array = np.array([0.5, -0.5])
    model_with_coef = SimpleNamespace(coef_=coef_array)
    importance = trainer._calc_feature_importance(model_with_coef)
    assert importance is not None
    np.testing.assert_array_almost_equal(importance, np.abs(coef_array))

    # Test with coef_ attribute (multi target)
    multi_coef_array = np.array([[0.5, -0.5], [0.3, 0.7]])
    model_with_multi_coef = SimpleNamespace(coef_=multi_coef_array)
    importance = trainer._calc_feature_importance(model_with_multi_coef)
    assert importance is not None
    expected = np.mean(np.abs(multi_coef_array), axis=0)
    np.testing.assert_array_almost_equal(importance, expected)

    # Test model with both attributes
    model_with_both = SimpleNamespace(
        feature_importances_=np.array([0.7, 0.3]), coef_=np.array([0.5, -0.5])
    )
    importance = trainer._calc_feature_importance(model_with_both)
    assert importance is not None
    np.testing.assert_array_almost_equal(importance, np.array([0.7, 0.3]))

    # Test with empty arrays
    model_empty = SimpleNamespace(feature_importances_=np.array([]))
    importance = trainer._calc_feature_importance(model_empty)
    assert isinstance(importance, np.ndarray)
    assert len(importance) == 0

    # Test with different array dtypes
    float32_array = np.array([0.7, 0.3], dtype=np.float32)
    model_float32 = SimpleNamespace(feature_importances_=float32_array)
    importance = trainer._calc_feature_importance(model_float32)
    assert importance is not None
    assert importance.dtype == np.float32
    np.testing.assert_array_almost_equal(importance, float32_array)

    # Test with no importance attributes
    model_without_importance = SimpleNamespace()
    importance = trainer._calc_feature_importance(model_without_importance)
    assert importance is None


@pytest.mark.parametrize("n_splits", [3, 5, 10])
def test_trainer_with_different_cv_splits(dummy_model: Any, n_splits: int) -> None:
    """Test trainer with different CV split configurations.

    Args:
        dummy_model (DummyModel): Dummy model instance.
        n_splits (int): Number of CV splits.
    """

    trainer = DummyTrainer(base_model=dummy_model)
    trainer.config.n_splits = n_splits
    trainer.n_splits = trainer.config.n_splits

    assert trainer.n_splits == n_splits


def test_metric_config_better_score() -> None:
    """Test MetricConfig better_score functionality."""

    # Test default better_score (maximize)
    config_factory = ConfigFactory()
    configs = config_factory.get_config()
    config = configs.metric
    assert config.better_score(5, 3)
    assert not config.better_score(3, 5)
    assert not config.better_score(5, 5)
    assert config.better_score is gt

    # Test minimize better_score
    config = configs.metric
    config.better_score = lt
    assert config.better_score(3, 5)
    assert not config.better_score(5, 3)
    assert not config.better_score(5, 5)
    assert config.better_score is lt


def test_default_metric_configs() -> None:
    """Test default metric configurations."""

    configs = get_default_metric_configs()

    # Test presence of all metrics from evaluation.py
    expected_metrics = {
        # Classification metrics
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "matthews_correlation_coefficient",
        "quadratic_weighted_kappa",
        "jaccard",
        "hamming_loss",
        # "d2_log_loss",
        "zero_one_loss",
        "log_loss",
        "roc_auc",
        "brier_score",
        # Regression metrics
        "explained_variance",
        "max_error",
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
        "median_absolute_error",
        "r2",
        "mean_absolute_percentage_error",
        "d2_absolute_error",
        "d2_pinball",
        "d2_tweedie",
        # "mean_squared_log_error",
        # "root_mean_squared_log_error",
        # "mean_poisson_deviance",
        # "mean_gamma_deviance",
    }

    assert set(configs.keys()) == expected_metrics

    # Test configuration correctness by metric type
    maximize_metrics = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "matthews_correlation_coefficient",
        "quadratic_weighted_kappa",
        "jaccard",
        "roc_auc",
        "explained_variance",
        "r2",
        # "d2_log_loss",
        "d2_absolute_error",
        "d2_pinball",
        "d2_tweedie",
    }

    for metric, config in configs.items():
        # Test direction
        expected_direction = "maximize" if metric in maximize_metrics else "minimize"
        assert config.direction == expected_direction, f"Wrong direction for {metric}"

        # Test initial score
        if config.direction == "maximize":
            assert config.initial_score == -float("inf"), f"Wrong initial score for {metric}"
        else:
            assert config.initial_score == float("inf"), f"Wrong initial score for {metric}"

        # Test better_score function
        if config.direction == "maximize":
            assert config.better_score == gt, f"Wrong comparison for {metric}"
        else:
            assert config.better_score == lt, f"Wrong comparison for {metric}"
