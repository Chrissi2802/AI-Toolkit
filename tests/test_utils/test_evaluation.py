from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from ai_toolkit.utils.evaluation import (
    ClassificationMetrics,
    CrossValidationMetrics,
    RegressionMetrics,
)


@pytest.fixture
def cv_metrics_data() -> List[Dict[str, float]]:
    """Create cross-validation metrics data.

    Returns:
        List[Dict[str, float]]: List of dictionaries containing evaluation metrics.
    """

    return [
        {"accuracy": 0.95, "precision": 0.94, "recall": 0.93},
        {"accuracy": 0.92, "precision": 0.91, "recall": 0.90},
        {"accuracy": 0.89, "precision": 0.88, "recall": 0.87},
    ]


class TestClassificationMetrics:
    """Test suite for ClassificationMetrics."""

    def test_basic_metrics(
        self, classification_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test basic classification metrics calculation.

        Args:
            classification_predictions (Tuple[np.ndarray, np.ndarray, np.ndarray]):
                Tuple of true labels, predicted labels, and predicted probabilities.
        """

        y_true, y_pred, y_pred_proba = classification_predictions

        metrics = ClassificationMetrics.calculate_basic_metrics(y_true, y_pred, y_pred_proba)

        # Test presence of all metrics
        expected_metrics = {
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
        }
        assert all(metric in metrics for metric in expected_metrics)

        # Test metric values against sklearn implementations
        np.testing.assert_almost_equal(metrics["accuracy"], accuracy_score(y_true, y_pred))
        np.testing.assert_almost_equal(
            metrics["precision"], precision_score(y_true, y_pred, average="weighted")
        )
        np.testing.assert_almost_equal(
            metrics["recall"], recall_score(y_true, y_pred, average="weighted")
        )
        np.testing.assert_almost_equal(metrics["f1"], f1_score(y_true, y_pred, average="weighted"))
        np.testing.assert_almost_equal(
            metrics["quadratic_weighted_kappa"],
            cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        )

    def test_metrics_without_probabilities(
        self, classification_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test metrics calculation without probability predictions.

        Args:
            classification_predictions (Tuple[np.ndarray, np.ndarray, np.ndarray]):
                Tuple of true labels, predicted labels, and predicted probabilities.
        """

        y_true, y_pred, _ = classification_predictions

        metrics = ClassificationMetrics.calculate_basic_metrics(y_true, y_pred)

        # Probability-based metrics should not be present
        assert "log_loss" not in metrics
        assert "roc_auc" not in metrics
        assert "brier_score" not in metrics

    def test_confusion_matrix(
        self, classification_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test confusion matrix calculation.

        Args:
            classification_predictions (Tuple[np.ndarray, np.ndarray, np.ndarray]):
                Tuple of true labels, predicted labels, and predicted probabilities.
        """

        y_true, y_pred, _ = classification_predictions

        # Test different normalization options
        cm = ClassificationMetrics.get_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)  # Binary classification

        cm_normalized = ClassificationMetrics.get_confusion_matrix(y_true, y_pred, normalize="true")
        assert np.allclose(cm_normalized.sum(axis=1), 1)  # Row sums should be 1

    def test_edge_cases(self) -> None:
        """Test edge cases for classification metrics."""

        # Perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = ClassificationMetrics.calculate_basic_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

        # Worst predictions
        y_pred = 1 - y_true
        metrics = ClassificationMetrics.calculate_basic_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["f1"] == 0.0

    def test_input_validation(self) -> None:
        """Test input validation for classification metrics."""

        # Invalid input shapes
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])
        with pytest.raises(RuntimeError):
            ClassificationMetrics.calculate_basic_metrics(y_true, y_pred)

        # Invalid input types
        y_pred = np.array([0, 1, 1])
        y_pred_proba = np.array([0, 1])
        with pytest.raises(RuntimeError):
            ClassificationMetrics.calculate_basic_metrics(y_true, y_pred, y_pred_proba)


class TestRegressionMetrics:
    """Test suite for RegressionMetrics."""

    def test_basic_metrics(self, regression_predictions: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic regression metrics calculation.

        Args:
            regression_predictions (Tuple[np.ndarray, np.ndarray]):
                Tuple of true and predicted regression values.
        """

        y_true, y_pred = regression_predictions

        metrics = RegressionMetrics.calculate_basic_metrics(y_true, y_pred)

        # Test presence of all metrics
        expected_metrics = {
            "explained_variance",
            "max_error",
            "mean_absolute_error",
            "mean_squared_error",
            "root_mean_squared_error",
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
        assert all(metric in metrics for metric in expected_metrics)

        # Test metric values against sklearn implementations
        np.testing.assert_almost_equal(
            metrics["explained_variance"], explained_variance_score(y_true, y_pred)
        )
        np.testing.assert_almost_equal(
            metrics["mean_squared_error"], mean_squared_error(y_true, y_pred)
        )
        np.testing.assert_almost_equal(metrics["r2"], r2_score(y_true, y_pred))

    def test_residuals(self, regression_predictions: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test residuals calculation.

        Args:
            regression_predictions (Tuple[np.ndarray, np.ndarray]):
                Tuple of true and predicted regression values.
        """

        y_true, y_pred = regression_predictions

        residuals = RegressionMetrics.calculate_residuals(y_true, y_pred)
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) == len(y_true)
        assert np.allclose(residuals, y_true - y_pred)

    def test_edge_cases(self) -> None:
        """Test edge cases for regression metrics."""

        # Perfect predictions
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = RegressionMetrics.calculate_basic_metrics(y_true, y_pred)
        assert metrics["mean_squared_error"] == 0.0
        assert metrics["r2"] == 1.0

        # Constant predictions
        y_pred = np.array([2.0, 2.0, 2.0])
        metrics = RegressionMetrics.calculate_basic_metrics(y_true, y_pred)
        assert metrics["r2"] < 1.0

    def test_input_validation(self) -> None:
        """Test input validation for regression metrics."""

        # Invalid input shapes
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])
        with pytest.raises(RuntimeError):
            RegressionMetrics.calculate_basic_metrics(y_true, y_pred)


class TestCrossValidationMetrics:
    """Test suite for CrossValidationMetrics."""

    def test_metrics_aggregation(self, cv_metrics_data: List[Dict[str, float]]) -> None:
        """Test aggregation of cross-validation metrics.

        Args:
            cv_metrics_data (List[Dict[str, float]]):
                List of dictionaries containing evaluation metrics.
        """

        aggregated = CrossValidationMetrics.aggregate_cv_metrics(cv_metrics_data)

        # Test structure
        assert isinstance(aggregated, dict)
        assert all(metric in aggregated for metric in ["accuracy", "precision", "recall"])

        # Test aggregated values
        for metric in aggregated:
            stats = aggregated[metric]
            assert isinstance(stats, dict)
            assert all(key in stats for key in ["mean", "std", "median", "min", "max"])
            assert isinstance(stats["mean"], float)

            # Test specific values for accuracy
            if metric == "accuracy":
                np.testing.assert_almost_equal(stats["mean"], 0.92)
                np.testing.assert_almost_equal(stats["min"], 0.89)
                np.testing.assert_almost_equal(stats["max"], 0.95)

    def test_input_validation(self) -> None:
        """Test input validation for cross-validation metrics aggregation."""

        with pytest.raises(RuntimeError):
            CrossValidationMetrics.aggregate_cv_metrics([])

    def test_inconsistent_metrics(self) -> None:
        """Test aggregation with inconsistent metrics."""

        inconsistent_metrics = [
            {"metric1": 0.9, "metric2": 0.8},
            {"metric1": 0.7},  # Missing metric2
            {"metric1": 0.8, "metric2": 0.6},
        ]

        with pytest.raises(RuntimeError):
            CrossValidationMetrics.aggregate_cv_metrics(inconsistent_metrics)


@pytest.mark.parametrize(
    "metric_name,expected_range",
    [
        ("accuracy", (0, 1)),
        ("precision", (0, 1)),
        ("recall", (0, 1)),
        ("f1", (0, 1)),
        ("roc_auc", (0, 1)),
    ],
)
def test_classification_metric_ranges(
    classification_predictions: Any, metric_name: Any, expected_range: Any
) -> None:
    """Test that classification metrics stay within expected ranges."""

    y_true, y_pred, y_pred_proba = classification_predictions
    metrics = ClassificationMetrics.calculate_basic_metrics(y_true, y_pred, y_pred_proba)

    if metric_name in metrics:
        assert expected_range[0] <= metrics[metric_name] <= expected_range[1]


@pytest.mark.parametrize("n_folds", [3, 5, 10])
def test_cv_metrics_with_different_folds(n_folds: Any) -> None:
    """Test cross-validation metrics with different numbers of folds."""

    metrics = [{"accuracy": 0.9 + i / 100, "precision": 0.85 + i / 100} for i in range(n_folds)]

    aggregated = CrossValidationMetrics.aggregate_cv_metrics(metrics)
    assert isinstance(aggregated["accuracy"]["mean"], float)
    assert isinstance(aggregated["precision"]["mean"], float)
    assert not np.isnan(aggregated["accuracy"]["mean"])
    assert not np.isnan(aggregated["precision"]["mean"])
    assert aggregated["accuracy"]["mean"] > 0.9
    assert aggregated["precision"]["mean"] > 0.85
