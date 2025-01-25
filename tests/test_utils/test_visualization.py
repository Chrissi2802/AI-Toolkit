import pytest
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ai_toolkit.utils.visualization import (
    ClassificationPlots,
    RegressionPlots,
    ModelAnalysisPlots,
)


class TestClassificationPlots:
    """Test suite for ClassificationPlots."""

    def test_roc_curve(
        self, classification_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """Test ROC curve plotting.

        Args:
            classification_predictions:
                Tuple of true labels, predicted labels, and predicted probabilities.
        """

        y_true, _, y_pred_proba = classification_predictions

        fig = ClassificationPlots.plot_roc_curve(y_true, y_pred_proba, "Test ROC Curve")

        # Test figure properties
        assert isinstance(fig, Figure)
        ax = fig.gca()

        # Test plot elements
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"
        assert ax.get_title() == "Test ROC Curve"

        # Test line properties
        lines = ax.get_lines()
        assert len(lines) == 2  # ROC curve and random line

        # Clean up
        plt.close(fig)

    def test_confusion_matrix(
        self, classification_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """Test confusion matrix plotting.

        Args:
            classification_predictions:
                Tuple of true labels, predicted labels, and predicted probabilities
        """

        y_true, y_pred, _ = classification_predictions

        fig = ClassificationPlots.plot_confusion_matrix(
            y_true, y_pred, "Test Confusion Matrix"
        )

        # Test figure properties
        assert isinstance(fig, Figure)

        # Count main subplot axes (excluding colorbars)
        main_axes = [
            ax for ax in fig.axes if not ax.get_label().startswith("<colorbar>")
        ]
        assert len(main_axes) == 2

        # Test titles and labels
        for ax in main_axes:
            assert "Confusion Matrix" in ax.get_title()
            assert ax.get_xlabel() == "Predicted"
            assert ax.get_ylabel() == "True"

        plt.close(fig)

    @pytest.mark.parametrize("figsize", [(8, 6), (10, 8)])
    def test_plot_sizes(self, classification_predictions, figsize):
        """Test different plot sizes."""

        y_true, y_pred, y_pred_proba = classification_predictions

        # Test ROC curve
        fig = ClassificationPlots.plot_roc_curve(
            y_true, y_pred_proba, "Test ROC Curve", figsize=figsize
        )

        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)

        # Test confusion matrix
        fig = ClassificationPlots.plot_confusion_matrix(
            y_true, y_pred, "Test Confusion Matrix", figsize=figsize
        )
        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)


class TestRegressionPlots:
    """Test suite for RegressionPlots."""

    def test_residuals(self, regression_predictions: Tuple[np.ndarray, np.ndarray]):
        """Test residual plots.

        Args:
            regression_predictions: Tuple of true and predicted values.
        """

        y_true, y_pred = regression_predictions

        fig = RegressionPlots.plot_residuals(y_true, y_pred, "Test Residual Plot")

        # Test figure properties
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Should have two subplots

        # Test subplot properties
        ax1, ax2 = fig.axes
        assert ax1.get_xlabel() == "Predicted Values"
        assert ax1.get_ylabel() == "Residuals"
        assert "Residual Distribution" in ax2.get_title()

        # Clean up
        plt.close(fig)

    def test_prediction_scatter(
        self, regression_predictions: Tuple[np.ndarray, np.ndarray]
    ):
        """Test prediction scatter plot.

        Args:
            regression_predictions: Tuple of true and predicted values.
        """

        y_true, y_pred = regression_predictions

        fig = RegressionPlots.plot_prediction_scatter(
            y_true, y_pred, "Test Scatter Plot"
        )

        # Test figure properties
        assert isinstance(fig, Figure)
        ax = fig.gca()

        # Test plot elements
        assert ax.get_xlabel() == "Actual Values"
        assert ax.get_ylabel() == "Predicted Values"
        assert ax.get_title() == "Test Scatter Plot"

        # Should have scatter points and diagonal line
        children = ax.get_children()
        assert any(isinstance(child, plt.Line2D) for child in children)

        # Clean up
        plt.close(fig)

    @pytest.mark.parametrize("figsize", [(16, 6), (12, 8)])
    def test_plot_sizes(self, regression_predictions, figsize):
        """Test different plot sizes."""

        y_true, y_pred = regression_predictions

        # Test residual plot
        fig = RegressionPlots.plot_residuals(
            y_true, y_pred, "Test Residual Plot", figsize=figsize
        )
        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)

        # Test scatter plot
        fig = RegressionPlots.plot_prediction_scatter(
            y_true, y_pred, "Test Scatter Plot", figsize=(8, 6)
        )
        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, (8, 6))
        plt.close(fig)


class TestModelAnalysisPlots:
    """Test suite for ModelAnalysisPlots."""

    def test_feature_importance(self, feature_importance_data: Tuple[np.ndarray, list]):
        """Test feature importance plot.

        Args:
            feature_importance_data: Tuple of importance scores and feature names.
        """

        importance_scores, feature_names = feature_importance_data

        fig = ModelAnalysisPlots.plot_feature_importance(
            importance_scores, feature_names, "Test Feature Importance"
        )

        # Test figure properties
        assert isinstance(fig, Figure)
        ax = fig.gca()

        # Test plot elements
        assert ax.get_xlabel() == "Features"
        assert ax.get_ylabel() == "Importance Score"
        assert ax.get_title() == "Test Feature Importance"

        # Test number of bars
        assert len(ax.patches) == len(feature_names)

        # Clean up
        plt.close(fig)

    def test_feature_importance_sorting(
        self, feature_importance_data: Tuple[np.ndarray, list]
    ):
        """Test feature importance sorting.

        Args:
            feature_importance_data: Tuple of importance scores and feature names.
        """

        importance_scores, feature_names = feature_importance_data

        fig = ModelAnalysisPlots.plot_feature_importance(
            importance_scores, feature_names, "Test Feature Importance"
        )

        # Get heights of bars
        heights = [patch.get_height() for patch in fig.gca().patches]

        # Test that bars are sorted in descending order
        assert heights == sorted(heights, reverse=True)

        # Clean up
        plt.close(fig)

    @pytest.mark.parametrize("figsize", [(10, 6), (12, 8)])
    def test_plot_sizes(self, feature_importance_data, figsize):
        """Test different plot sizes."""

        importance_scores, feature_names = feature_importance_data

        fig = ModelAnalysisPlots.plot_feature_importance(
            importance_scores, feature_names, "Test Feature Importance", figsize=figsize
        )
        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)


def test_plot_style_consistency():
    """Test plot style consistency across all plots."""

    # Create test data
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.6, 0.9])
    importance_scores = np.array([0.3, 0.7])
    feature_names = ["Feature_1", "Feature_2"]

    # Create all types of plots
    plots = [
        ClassificationPlots.plot_roc_curve(y_true, y_pred_proba, "ROC"),
        ClassificationPlots.plot_confusion_matrix(y_true, y_pred, "Confusion"),
        RegressionPlots.plot_residuals(y_true, y_pred, "Residuals"),
        RegressionPlots.plot_prediction_scatter(y_true, y_pred, "Scatter"),
        ModelAnalysisPlots.plot_feature_importance(
            importance_scores, feature_names, "Importance"
        ),
    ]

    # Test style consistency
    for fig in plots:
        assert isinstance(fig, Figure)
        for ax in fig.axes:
            if not ax.get_label().startswith("<colorbar>"):
                assert ax.get_title() != ""  # Should have a title
                assert ax.get_xlabel() != ""  # Should have x-label
                assert ax.get_ylabel() != ""  # Should have y-label
        plt.close(fig)


@pytest.mark.parametrize(
    "plot_func,args",
    [
        (ClassificationPlots.plot_roc_curve, (np.array([]), np.array([]), "Empty")),
        (RegressionPlots.plot_residuals, (np.array([]), np.array([]), "Empty")),
        (ModelAnalysisPlots.plot_feature_importance, (np.array([]), [], "Empty")),
    ],
)
def test_empty_data_handling(plot_func, args):
    """Test handling of empty data."""

    with pytest.raises(ValueError):
        fig = plot_func(*args)
        plt.close(fig)
