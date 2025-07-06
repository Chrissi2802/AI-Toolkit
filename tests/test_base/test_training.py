from operator import gt, lt
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import optuna
import pytest

from ai_toolkit.base.models import BaseMlModel
from ai_toolkit.base.training import (
    BaseMlTrainer,
    MetricConfig,
    MlTrainerConfig,
    get_default_metric_configs,
)


class DummyTrainer(BaseMlTrainer):
    """Dummy trainer implementation for testing."""

    def __init__(self, base_model: BaseMlModel, config: MlTrainerConfig = MlTrainerConfig()):
        """Initialize dummy trainer.

        Args:
            base_model (BaseMlModel): Base model to use.
            config (MlTrainerConfig, optional):
                Training configuration. Defaults to MlTrainerConfig().
        """
        super().__init__(base_model=base_model, config=config)

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


def test_base_trainer_initialization(dummy_model):
    """Test BaseMlTrainer initialization."""

    config = MlTrainerConfig(
        N_SPLITS=5,
        RANDOM_STATE=28,
        N_TRAILS=2,
        EXPERIMENT_NAME="test_classification",
        OPTIMIZE_METRIC="f1",
    )

    trainer = DummyTrainer(base_model=dummy_model, config=config)

    assert trainer.base_model == dummy_model
    assert trainer.n_splits == 5
    assert trainer.random_state == 28
    assert trainer.experiment_name == "test_classification"
    assert trainer.optimize_metric == "f1"
    assert trainer.best_model is None
    assert trainer.feature_names is None


def test_log_training_info(dummy_model, mock_mlflow):
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


def test_log_dataset_info(dummy_model, mock_mlflow):
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


def test_log_fold_results(dummy_model, mock_mlflow):
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


def test_calc_feature_importance(dummy_model):
    """Test feature importance calculation."""

    trainer = DummyTrainer(base_model=dummy_model)

    # Test with feature_importances_ attribute
    importance_array = np.array([0.7, 0.3])
    model_with_importance = SimpleNamespace(feature_importances_=importance_array)
    importance = trainer._calc_feature_importance(model_with_importance)
    np.testing.assert_array_almost_equal(importance, importance_array)

    # Test with coef_ attribute (single target)
    coef_array = np.array([0.5, -0.5])
    model_with_coef = SimpleNamespace(coef_=coef_array)
    importance = trainer._calc_feature_importance(model_with_coef)
    np.testing.assert_array_almost_equal(importance, np.abs(coef_array))

    # Test with coef_ attribute (multi target)
    multi_coef_array = np.array([[0.5, -0.5], [0.3, 0.7]])
    model_with_multi_coef = SimpleNamespace(coef_=multi_coef_array)
    importance = trainer._calc_feature_importance(model_with_multi_coef)
    expected = np.mean(np.abs(multi_coef_array), axis=0)
    np.testing.assert_array_almost_equal(importance, expected)

    # Test model with both attributes
    model_with_both = SimpleNamespace(
        feature_importances_=np.array([0.7, 0.3]), coef_=np.array([0.5, -0.5])
    )
    importance = trainer._calc_feature_importance(model_with_both)
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
    assert importance.dtype == np.float32
    np.testing.assert_array_almost_equal(importance, float32_array)

    # Test with no importance attributes
    model_without_importance = SimpleNamespace()
    importance = trainer._calc_feature_importance(model_without_importance)
    assert importance is None


@pytest.mark.parametrize("n_splits", [3, 5, 10])
def test_trainer_with_different_cv_splits(dummy_model, n_splits: int):
    """Test trainer with different CV split configurations.

    Args:
        dummy_model (DummyModel): Dummy model instance.
        n_splits (int): Number of CV splits.
    """

    config = MlTrainerConfig(N_SPLITS=n_splits)
    trainer = DummyTrainer(base_model=dummy_model, config=config)

    assert trainer.n_splits == n_splits


def test_trainer_error_handling():
    """Test error handling in trainer."""

    # Test with invalid n_splits
    with pytest.raises(ValueError):
        MlTrainerConfig(N_SPLITS=1)

    # Test with invalid random state
    with pytest.raises(ValueError):
        MlTrainerConfig(RANDOM_STATE=-1)


@pytest.mark.parametrize(
    "config_params",
    [
        {"N_SPLITS": 5, "RANDOM_STATE": 28},
        {"OPTIMIZE_METRIC": "accuracy"},
        {"EXPERIMENT_NAME": "test_classification"},
    ],
)
def test_trainer_config_variations(dummy_model, config_params: Dict[str, Any]):
    """Test trainer with different configurations.

    Args:
        dummy_model (DummyModel): Dummy model instance.
        config_params (Dict[str, Any]): Configuration parameters.
    """

    config = MlTrainerConfig(**config_params)
    trainer = DummyTrainer(base_model=dummy_model, config=config)

    for param, value in config_params.items():
        assert getattr(trainer, param.lower()) == value


def test_metric_config_validation():
    """Test MetricConfig validation."""

    # Test valid configurations
    valid_configs = [
        MetricConfig(),  # Default values
        MetricConfig(DIRECTION="maximize", INITIAL_SCORE=0.0),
        MetricConfig(DIRECTION="minimize", INITIAL_SCORE=float("inf")),
        MetricConfig(BETTER_SCORE=lambda x, y: x < y),
    ]

    for config in valid_configs:
        assert config.DIRECTION in ["maximize", "minimize"]
        assert callable(config.BETTER_SCORE)

    # Test invalid configurations
    with pytest.raises(ValueError, match="DIRECTION must be"):
        MetricConfig(DIRECTION="invalid")

    with pytest.raises(ValueError, match="INITIAL_SCORE must be"):
        MetricConfig(INITIAL_SCORE="invalid")

    with pytest.raises(ValueError, match="BETTER_SCORE must be"):
        MetricConfig(BETTER_SCORE="invalid")


def test_metric_config_better_score():
    """Test MetricConfig better_score functionality."""

    # Test default better_score (maximize)
    config = MetricConfig()
    assert config.BETTER_SCORE(5, 3)
    assert not config.BETTER_SCORE(3, 5)
    assert not config.BETTER_SCORE(5, 5)
    assert config.BETTER_SCORE is gt

    # Test minimize better_score
    config = MetricConfig(BETTER_SCORE=lt)
    assert config.BETTER_SCORE(3, 5)
    assert not config.BETTER_SCORE(5, 3)
    assert not config.BETTER_SCORE(5, 5)
    assert config.BETTER_SCORE is lt


def test_default_metric_configs():
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
        assert config.DIRECTION == expected_direction, f"Wrong direction for {metric}"

        # Test initial score
        if config.DIRECTION == "maximize":
            assert config.INITIAL_SCORE == -float("inf"), f"Wrong initial score for {metric}"
        else:
            assert config.INITIAL_SCORE == float("inf"), f"Wrong initial score for {metric}"

        # Test better_score function
        if config.DIRECTION == "maximize":
            assert config.BETTER_SCORE == gt, f"Wrong comparison for {metric}"
        else:
            assert config.BETTER_SCORE == lt, f"Wrong comparison for {metric}"


def test_metric_config_integration():
    """Test MetricConfig integration with MlTrainerConfig."""

    # Test with valid metric
    config = MlTrainerConfig(OPTIMIZE_METRIC="accuracy")
    metric_config = config.METRIC_CONFIGS
    assert metric_config.DIRECTION == "maximize"
    assert metric_config.INITIAL_SCORE == -float("inf")
    assert metric_config.BETTER_SCORE == gt

    # Test with invalid metric
    with pytest.raises(ValueError, match="not supported"):
        MlTrainerConfig(OPTIMIZE_METRIC="invalid_metric")
