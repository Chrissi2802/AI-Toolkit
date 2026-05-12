from typing import List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_curve

from ai_toolkit.utils.logging import get_logger


sns.set_style("darkgrid")


class ClassificationPlots:
    """Visualization utilities for classification models."""

    logger = get_logger("ClassificationPlots")

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve",
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot ROC curve and return figure.

        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            title (str, optional): Plot title. Defaults to "ROC Curve".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 6).

        Returns:
            plt.Figure: Matplotlib figure object
        """

        try:
            ClassificationPlots.logger.debug(
                "Creating ROC curve plot", data_shape=y_true.shape, figsize=figsize
            )

            # Check y_true and y_pred_proba
            if len(y_true) == 0 or y_pred_proba is None:
                raise ValueError("y_true and y_pred_proba cannot be empty.")

            fig = plt.figure(figsize=figsize)
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

            plt.plot(fpr, tpr, color="royalblue", label="Model")
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.grid()
            plt.legend()
            plt.tight_layout()

            ClassificationPlots.logger.debug("ROC curve plot created successfully")

            return fig

        except Exception as e:
            ClassificationPlots.logger.error("Failed to create ROC curve plot", error=e)
            raise RuntimeError("Plot creation failed") from e

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot confusion matrix and return figure.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            feature_names (Optional[List[str]], optional): Class names for axis labels.
                Defaults to None (uses unique values from y_true).
            title (str, optional): Plot title. Defaults to "Confusion Matrix".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

        Returns:
            plt.Figure: Matplotlib figure object
        """

        try:
            ClassificationPlots.logger.debug(
                "Creating confusion matrix plot",
                data_shape=y_true.shape,
                feature_names=feature_names,
                figsize=figsize,
            )

            # Check y_true and y_pred
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("y_true and y_pred cannot be empty.")

            labels = feature_names if feature_names is not None else list(np.unique(y_true))

            # Create figure and subplots
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = cast(Tuple[plt.Axes, plt.Axes], axes)
            fig.suptitle(title)

            # Calculate confusion matrices
            cm = confusion_matrix(y_true, y_pred)
            cm_percent = confusion_matrix(y_true, y_pred, normalize="true") * 100

            # Plot absolute numbers
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax1,
                cbar=True,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                linewidths=1,
                linecolor="gray",
            )
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("True")
            ax1.set_title("Absolute Numbers")

            # Plot percentages
            sns.heatmap(
                cm_percent,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                ax=ax2,
                cbar=True,
                square=True,
                xticklabels=labels,
                yticklabels=labels,
                linewidths=1,
                linecolor="gray",
            )
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("True")
            ax2.set_title("Normalized (Percentages %)")

            # Add % symbol to annotations in the percentage plot
            for t in ax2.texts:
                t.set_text(t.get_text() + "%")

            # Adjust layout
            plt.tight_layout()

            ClassificationPlots.logger.debug("Confusion matrix plot created successfully")

            return fig

        except Exception as e:
            ClassificationPlots.logger.error("Failed to create confusion matrix plot", error=e)
            raise RuntimeError("Plot creation failed") from e


class RegressionPlots:
    """Visualization utilities for regression models."""

    logger = get_logger("RegressionPlots")

    @staticmethod
    def plot_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Plot",
        figsize: Tuple[int, int] = (16, 6),
    ) -> plt.Figure:
        """Create residual plots.

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str, optional): Plot title. Defaults to "Residual Plot".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (16, 6).

        Returns:
            plt.Figure: Matplotlib figure object
        """

        try:
            RegressionPlots.logger.debug(
                "Creating residual plot", data_shape=y_true.shape, figsize=figsize
            )

            # Check y_true and y_pred
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("y_true and y_pred cannot be empty.")

            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = cast(Tuple[plt.Axes, plt.Axes], axes)

            residuals = y_true - y_pred

            # Residuals vs Predicted
            ax1.scatter(y_pred, residuals, alpha=0.5, color="royalblue")
            ax1.axhline(y=0, color="r", linestyle="--")
            ax1.set_xlabel("Predicted Values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residuals vs Predicted Values")
            ax1.grid()

            # Residual Distribution
            sns.histplot(residuals, kde=True, ax=ax2, color="royalblue")
            ax2.set_title("Residual Distribution")
            ax2.set_xlabel("Residual Value")

            plt.suptitle(title)
            plt.tight_layout()

            RegressionPlots.logger.debug("Residual plot created successfully")

            return fig

        except Exception as e:
            RegressionPlots.logger.error("Failed to create residual plot", error=e)
            raise RuntimeError("Plot creation failed") from e

    @staticmethod
    def plot_prediction_scatter(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted",
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Create actual vs predicted scatter plot.

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str, optional): Plot title. Defaults to "Actual vs Predicted".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 6).

        Returns:
            plt.Figure: Matplotlib figure object
        """

        try:
            RegressionPlots.logger.debug(
                "Creating actual vs predicted scatter plot",
                data_shape=y_true.shape,
                figsize=figsize,
            )

            # Check y_true and y_pred
            if len(y_true) == 0 or len(y_pred) == 0:
                raise ValueError("y_true and y_pred cannot be empty.")

            fig = plt.figure(figsize=figsize)
            plt.scatter(y_true, y_pred, alpha=0.5, color="royalblue")
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(title)
            plt.grid()

            RegressionPlots.logger.debug("Actual vs predicted scatter plot created successfully")

            return fig

        except Exception as e:
            RegressionPlots.logger.error(
                "Failed to create actual vs predicted scatter plot", error=e
            )
            raise RuntimeError("Plot creation failed") from e


class ModelAnalysisPlots:
    """General model analysis visualization utilities."""

    logger = get_logger("ModelAnalysisPlots")

    @staticmethod
    def plot_feature_importance(
        importance_scores: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Importance",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot feature importance scores.

        Args:
            importance_scores (np.ndarray): Feature importance scores
            feature_names (List[str]): Feature names
            title (str, optional): Plot title. Defaults to "Feature Importance".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

        Returns:
            plt.Figure: Matplotlib figure object
        """

        try:
            ModelAnalysisPlots.logger.debug(
                "Creating feature importance plot",
                n_features=len(feature_names),
                figsize=figsize,
            )

            # Check importance_scores and feature_names
            if len(importance_scores) == 0 or len(feature_names) == 0:
                raise ValueError("importance_scores and feature_names cannot be empty.")

            # Sort features by importance
            indices = np.argsort(importance_scores)[::-1]

            fig = plt.figure(figsize=figsize)
            plt.bar(
                range(len(importance_scores)),
                importance_scores[indices],
                color="royalblue",
            )
            plt.xticks(
                range(len(importance_scores)),
                [feature_names[i] for i in indices],
                rotation=45,
                ha="right",
            )
            plt.xlabel("Features")
            plt.ylabel("Importance Score")
            plt.title(title)
            plt.tight_layout()

            ModelAnalysisPlots.logger.debug("Feature importance plot created successfully")

            return fig

        except Exception as e:
            ModelAnalysisPlots.logger.error("Failed to create feature importance plot", error=e)
            raise RuntimeError("Plot creation failed") from e

    @staticmethod
    def plot_shapley_values(
        model: object,
        X: np.ndarray,
        feature_names: List[str],
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot SHAP values.

        Args:
            model (object): Trained model.
            X (np.ndarray): Feature matrix.
            feature_names (List[str]): Feature names.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 6).

        Returns:
            plt.Figure: Matplotlib figure object
        """

        try:
            ModelAnalysisPlots.logger.debug(
                "Creating SHAP feature importance plot", figsize=figsize
            )

            # Check feature_names and shap_values
            if X is None or len(feature_names) == 0:
                raise ValueError("X and feature_names cannot be empty.")

            # Check number of features
            if X.shape[1] != len(feature_names):
                raise ValueError("Number of features in X and feature_names do not match.")

            # Limit samples
            sample_size = 200
            if len(X) > sample_size:
                X = shap.sample(X, sample_size, random_state=28)

            # Use the unified SHAP explainer
            explainer = shap.Explainer(model, X)
            explanation = explainer(X)

            if hasattr(explanation, "values"):
                shap_values = explanation.values
            else:
                shap_values = np.stack([e.values for e in explanation], axis=2)

            # Reduce dimensionality if necessary (e.g. multi-class classification)
            if shap_values.ndim == 3:
                shap_values = np.mean(np.abs(shap_values), axis=2)

            # Create figure
            fig = plt.figure(figsize=figsize)
            shap.summary_plot(shap_values, feature_names, plot_type="violin", show=False)
            plt.title("SHAP Feature Importance")
            plt.ylabel("Feature")
            plt.tight_layout()

            ModelAnalysisPlots.logger.debug("SHAP feature importance plot created successfully")

            return fig

        except Exception as e:
            ModelAnalysisPlots.logger.error(
                "Failed to create SHAP feature importance plot", error=e
            )
            raise RuntimeError("Plot creation failed") from e


if __name__ == "__main__":
    pass
