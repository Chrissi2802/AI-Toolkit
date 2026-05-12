from typing import Any, Dict, Generator, List, Tuple
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from ai_toolkit.utils.visualization import (
    ClassificationPlots,
    ModelAnalysisPlots,
    RegressionPlots,
)


@pytest.fixture
def mock_shap() -> Generator[Dict[str, Any], None, None]:
    """Create mock SHAP objects and functions."""

    with patch("shap.Explainer") as mock_explainer:
        with patch("shap.summary_plot") as mock_summary_plot:
            # Configure mock explanation with proper numpy arrays
            mock_values = np.random.randn(100, 5).astype(np.float32)
            mock_explanation = Mock()
            mock_explanation.values = mock_values
            mock_explainer.return_value.return_value = mock_explanation

            # Configure summary_plot to do nothing
            mock_summary_plot.return_value = None

            yield {
                "explainer": mock_explainer,
                "summary_plot": mock_summary_plot,
            }


class TestClassificationPlots:
    """Test suite for ClassificationPlots."""

    def test_roc_curve(
        self, classification_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test ROC curve plotting.

        Args:
            classification_predictions:
                Tuple of true labels, predicted labels, and predicted probabilities.
        """

        y_true, _, y_pred_proba = classification_predictions

        fig = ClassificationPlots.plot_roc_curve(y_true, y_pred_proba[:, 1], "Test ROC Curve")

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
    ) -> None:
        """Test confusion matrix plotting.

        Args:
            classification_predictions:
                Tuple of true labels, predicted labels, and predicted probabilities
        """

        y_true, y_pred, _ = classification_predictions
        feature_names = [f"class_{c}" for c in np.unique(y_true)]

        fig = ClassificationPlots.plot_confusion_matrix(
            y_true, y_pred, feature_names, "Test Confusion Matrix"
        )

        # Test figure properties
        assert isinstance(fig, Figure)

        # Count main subplot axes (excluding colorbars)
        main_axes = [ax for ax in fig.axes if not str(ax.get_label()).startswith("<colorbar>")]
        assert len(main_axes) == 2

        # Test titles and labels
        assert "Confusion Matrix" in fig._suptitle.get_text()  # type: ignore[attr-defined]

        for ax in main_axes:
            assert ax.get_xlabel() == "Predicted"
            assert ax.get_ylabel() == "True"

        plt.close(fig)

    @pytest.mark.parametrize("figsize", [(8, 6), (10, 8)])
    def test_plot_sizes(self, classification_predictions: Any, figsize: Any) -> None:
        """Test different plot sizes."""

        y_true, y_pred, y_pred_proba = classification_predictions
        feature_names = [f"class_{c}" for c in np.unique(y_true)]

        # Test ROC curve
        fig = ClassificationPlots.plot_roc_curve(
            y_true, y_pred_proba[:, 1], "Test ROC Curve", figsize=figsize
        )

        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)

        # Test confusion matrix
        fig = ClassificationPlots.plot_confusion_matrix(
            y_true, y_pred, feature_names, "Test Confusion Matrix", figsize=figsize
        )
        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)


class TestRegressionPlots:
    """Test suite for RegressionPlots."""

    def test_residuals(self, regression_predictions: Tuple[np.ndarray, np.ndarray]) -> None:
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
    ) -> None:
        """Test prediction scatter plot.

        Args:
            regression_predictions: Tuple of true and predicted values.
        """

        y_true, y_pred = regression_predictions

        fig = RegressionPlots.plot_prediction_scatter(y_true, y_pred, "Test Scatter Plot")

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
    def test_plot_sizes(self, regression_predictions: Any, figsize: Any) -> None:
        """Test different plot sizes."""

        y_true, y_pred = regression_predictions

        # Test residual plot
        fig = RegressionPlots.plot_residuals(y_true, y_pred, "Test Residual Plot", figsize=figsize)
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

    def test_feature_importance(self, feature_importance_data: Tuple[np.ndarray, list]) -> None:
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
    ) -> None:
        """Test feature importance sorting.

        Args:
            feature_importance_data: Tuple of importance scores and feature names.
        """

        importance_scores, feature_names = feature_importance_data

        fig = ModelAnalysisPlots.plot_feature_importance(
            importance_scores, feature_names, "Test Feature Importance"
        )

        # Get heights of bars
        heights = [patch.get_height() for patch in fig.gca().patches]  # type: ignore[attr-defined]

        # Test that bars are sorted in descending order
        assert heights == sorted(heights, reverse=True)

        # Clean up
        plt.close(fig)

    def test_shapley_plot_tree_model(self, mock_shap: Any, feature_importance_data: Any) -> None:
        """Test Shapley plot with tree-based model."""

        # Create mock tree-based model
        class TreeModel:

            def predict(self, X: Any) -> np.ndarray:
                return np.zeros(len(X))

            def apply(self, X: Any) -> np.ndarray:
                return np.zeros(len(X))

        mock_model = TreeModel()

        # Get test data
        _, feature_names = feature_importance_data
        X = np.random.randn(100, len(feature_names)).astype(np.float32)

        # Create plot
        fig = ModelAnalysisPlots.plot_shapley_values(mock_model, X, feature_names)

        # Verify Explainer was called
        mock_shap["explainer"].assert_called_once()

        # Verify plot creation
        mock_shap["summary_plot"].assert_called_once()

        # Test figure properties
        assert isinstance(fig, Figure)
        ax = fig.gca()
        assert ax.get_title() == "SHAP Feature Importance"

        plt.close(fig)

    def test_shapley_plot_kernel_model(self, mock_shap: Any, feature_importance_data: Any) -> None:
        """Test Shapley plot with kernel-based model."""

        # Create mock kernel-based model
        class KernelModel:

            def predict(self, X: Any) -> np.ndarray:
                return np.zeros(len(X))

        mock_model = KernelModel()

        # Get test data
        _, feature_names = feature_importance_data
        X = np.random.randn(100, len(feature_names)).astype(np.float32)

        # Create plot
        fig = ModelAnalysisPlots.plot_shapley_values(mock_model, X, feature_names)

        # Verify Explainer was called
        mock_shap["explainer"].assert_called_once()

        # Verify plot creation
        mock_shap["summary_plot"].assert_called_once()

        # Test figure properties
        assert isinstance(fig, Figure)
        ax = fig.gca()
        assert ax.get_title() == "SHAP Feature Importance"

        plt.close(fig)

    def test_shapley_plot_empty_data(self, mock_shap: Any) -> None:
        """Test Shapley plot with empty data."""

        mock_model = Mock()
        X = np.array([])
        feature_names: List[str] = []

        with pytest.raises(RuntimeError, match="Plot creation failed"):
            fig = ModelAnalysisPlots.plot_shapley_values(mock_model, X, feature_names)
            plt.close(fig)

    def test_shapley_plot_sample_reduction(
        self, mock_shap: Any, feature_importance_data: Any
    ) -> None:
        """Test sample size reduction for large datasets in Shapley plot."""

        # Create mock kernel-based model
        class KernelModel:

            def predict(self, X: Any) -> np.ndarray:
                return np.zeros(len(X))

        mock_model = KernelModel()

        # Create large test dataset
        _, feature_names = feature_importance_data
        X = np.random.randn(1000, len(feature_names))

        with patch("shap.sample") as mock_sample:
            mock_sample.return_value = X[:100]

            fig = ModelAnalysisPlots.plot_shapley_values(mock_model, X, feature_names)

            # Verify sampling was called
            mock_sample.assert_called_once()
            args, kwargs = mock_sample.call_args
            assert args[0].shape == X.shape
            assert args[1] == 200  # Sample size
            assert kwargs["random_state"] == 28

            plt.close(fig)

    @pytest.mark.parametrize("figsize", [(10, 6), (12, 8)])
    def test_plot_sizes(self, feature_importance_data: Any, mock_shap: Any, figsize: Any) -> None:
        """Test different plot sizes."""

        importance_scores, feature_names = feature_importance_data

        # Test feature importance plot
        fig = ModelAnalysisPlots.plot_feature_importance(
            importance_scores, feature_names, "Test Feature Importance", figsize=figsize
        )
        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)

        # Test Shapley plot with proper mock model
        class TreeModel:

            def predict(self, X: Any) -> np.ndarray:
                return np.zeros(len(X))

            def apply(self, X: Any) -> np.ndarray:
                return np.zeros(len(X))

        mock_model = TreeModel()
        X = np.random.randn(100, len(feature_names)).astype(np.float32)

        fig = ModelAnalysisPlots.plot_shapley_values(mock_model, X, feature_names, figsize=figsize)
        size_inches = fig.get_size_inches()
        assert np.allclose(size_inches, figsize)
        plt.close(fig)


def test_plot_style_consistency() -> None:
    """Test plot style consistency across all plots."""

    # Create test data
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.6, 0.9])
    importance_scores = np.array([0.3, 0.7])
    feature_names = ["Feature_1", "Feature_2"]
    X = np.random.randn(4, 2)  # Sample data for Shapley plot

    # Create proper mock model for Shapley plot
    class TreeModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.zeros(len(X))

        def apply(self, X: Any) -> np.ndarray:
            return np.zeros(len(X))

    mock_model = TreeModel()

    # Mock SHAP functions
    with patch("shap.Explainer") as mock_explainer:
        with patch("shap.summary_plot") as mock_summary_plot:
            # Configure mock with proper numpy arrays
            mock_values = np.random.randn(4, 2).astype(np.float32)
            mock_explanation = Mock()
            mock_explanation.values = mock_values
            mock_explainer.return_value.return_value = mock_explanation
            mock_summary_plot.return_value = None

            # Create all types of plots
            plots = [
                ClassificationPlots.plot_roc_curve(y_true, y_pred_proba, "ROC"),
                ClassificationPlots.plot_confusion_matrix(
                    y_true, y_pred, feature_names, "Confusion"
                ),
                RegressionPlots.plot_residuals(y_true, y_pred, "Residuals"),
                RegressionPlots.plot_prediction_scatter(y_true, y_pred, "Scatter"),
                ModelAnalysisPlots.plot_feature_importance(
                    importance_scores, feature_names, "Importance"
                ),
                ModelAnalysisPlots.plot_shapley_values(
                    mock_model,
                    X,
                    feature_names,
                ),
            ]

            # Test style consistency
            for fig in plots:
                assert isinstance(fig, Figure)
                for ax in fig.axes:
                    if not str(ax.get_label()).startswith("<colorbar>"):
                        assert ax.get_title() != ""  # Should have a title

                        # Check if not Shapley plot
                        if "SHAP" not in ax.get_title():
                            assert ax.get_xlabel() != ""  # Should have x-label

                        assert ax.get_ylabel() != ""  # Should have y-label

                        # Test font sizes are consistent
                        title_size = ax.title.get_fontsize()
                        assert float(title_size) > 0

                        # Test axis visibility
                        assert ax.get_xaxis().get_visible()
                        assert ax.get_yaxis().get_visible()
                plt.close(fig)


@pytest.mark.parametrize(
    "plot_func,args",
    [
        (ClassificationPlots.plot_roc_curve, (np.array([]), np.array([]), "Empty")),
        (
            ClassificationPlots.plot_confusion_matrix,
            (np.array([]), np.array([]), [], "Empty"),
        ),
        (RegressionPlots.plot_residuals, (np.array([]), np.array([]), "Empty")),
        (
            RegressionPlots.plot_prediction_scatter,
            (np.array([]), np.array([]), "Empty"),
        ),
        (ModelAnalysisPlots.plot_feature_importance, (np.array([]), [], "Empty")),
        (ModelAnalysisPlots.plot_shapley_values, (None, np.array([]), [])),
    ],
)
def test_empty_data_handling(plot_func: Any, args: Any) -> None:
    """Test handling of empty data."""

    with pytest.raises(RuntimeError, match="Plot creation failed"):
        fig = plot_func(*args)
        plt.close(fig)
