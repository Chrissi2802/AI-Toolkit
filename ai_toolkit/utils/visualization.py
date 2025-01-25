from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

sns.set_style("darkgrid")


class ClassificationPlots:
    """Visualization utilities for classification models."""

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

        # Check y_true and y_pred_proba
        if len(y_true) == 0 or len(y_pred_proba) == 0:
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

        return fig

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot confusion matrix and return figure.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            title (str, optional): Plot title. Defaults to "Confusion Matrix".
            figsize (Tuple[int, int], optional): Figure size. Defaults to (8, 6).

        Returns:
            plt.Figure: Matplotlib figure object
        """

        # Check y_true and y_pred
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("y_true and y_pred cannot be empty.")

        # Create figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Calculate confusion matrices
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = confusion_matrix(y_true, y_pred, normalize="true") * 100

        # Plot absolute numbers
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax1, cbar=True, square=True
        )
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")
        ax1.set_title(f"{title}\n(Absolute Numbers)")

        # Plot percentages
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            ax=ax2,
            cbar=True,
            square=True,
        )
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title(f"{title}\n(Percentages %)")

        # Add % symbol to annotations in the percentage plot
        for t in ax2.texts:
            t.set_text(t.get_text() + "%")

        # Adjust layout
        plt.tight_layout()

        return fig


class RegressionPlots:
    """Visualization utilities for regression models."""

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

        # Check y_true and y_pred
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("y_true and y_pred cannot be empty.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

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

        return fig

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

        return fig


class ModelAnalysisPlots:
    """General model analysis visualization utilities."""

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

        # Check importance_scores and feature_names
        if len(importance_scores) == 0 or len(feature_names) == 0:
            raise ValueError("importance_scores and feature_names cannot be empty.")

        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]

        fig = plt.figure(figsize=figsize)
        plt.bar(
            range(len(importance_scores)), importance_scores[indices], color="royalblue"
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

        return fig


if __name__ == "__main__":
    pass
