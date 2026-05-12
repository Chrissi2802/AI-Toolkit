from typing import Dict, List, Optional, cast

import numpy as np
from sklearn import metrics

from ai_toolkit.utils.logging import get_logger


class ClassificationMetrics:
    """Utility class for classification metrics calculation."""

    logger = get_logger("ClassificationMetrics")

    @staticmethod
    def calculate_basic_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: Optional[str] = "weighted",
    ) -> Dict[str, float]:
        """Calculate all required metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities. Defaults to None.
            average (str): Averaging strategy for multiclass classification. Defaults to 'weighted'.

        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """

        try:
            # Basic validation
            if y_true.shape != y_pred.shape:
                raise ValueError("y_true and y_pred must have the same shape.")

            if y_pred_proba is not None and y_true.shape[0] != y_pred_proba.shape[0]:
                raise ValueError("y_true and y_pred_proba must have the same length.")

            ClassificationMetrics.logger.debug(
                "Calculating classification metrics",
                data_shape=y_true.shape,
                has_probabilities=y_pred_proba is not None,
            )

            result = {
                "accuracy": metrics.accuracy_score(y_true, y_pred),
                "balanced_accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
                "precision": metrics.precision_score(y_true, y_pred, average=average),
                "recall": metrics.recall_score(y_true, y_pred, average=average),
                "f1": metrics.f1_score(y_true, y_pred, average=average),
                "matthews_correlation_coefficient": metrics.matthews_corrcoef(y_true, y_pred),
                "quadratic_weighted_kappa": metrics.cohen_kappa_score(
                    y_true, y_pred, weights="quadratic"
                ),
                "jaccard": metrics.jaccard_score(y_true, y_pred, average=average),
                "hamming_loss": metrics.hamming_loss(y_true, y_pred),
                # "d2_log_loss": metrics.d2_log_loss_score(y_true, y_pred),
                "zero_one_loss": metrics.zero_one_loss(y_true, y_pred),
            }

            # Add additional metrics if probabilities are provided
            if y_pred_proba is not None:

                result.update(
                    {
                        "log_loss": metrics.log_loss(y_true, y_pred_proba),
                    }
                )

                if y_pred_proba.shape[1] == 2:
                    # Binary classification
                    result.update(
                        {
                            "roc_auc": metrics.roc_auc_score(y_true, y_pred_proba[:, 1]),
                            "brier_score": metrics.brier_score_loss(y_true, y_pred_proba[:, 1]),
                        }
                    )
                else:
                    # Multiclass classification
                    result.update(
                        {
                            "roc_auc": metrics.roc_auc_score(
                                y_true, y_pred_proba, average=average, multi_class="ovr"
                            ),
                            "brier_score": np.mean(
                                [
                                    metrics.brier_score_loss(
                                        (y_true == i).astype(int), y_pred_proba[:, i]
                                    )
                                    for i in range(y_pred_proba.shape[1])
                                ]
                            ),
                        }
                    )

            ClassificationMetrics.logger.debug("Metrics calculated successfully", metrics=result)

            return result

        except Exception as e:
            ClassificationMetrics.logger.error("Failed to calculate metrics", error=e)
            raise RuntimeError("Metrics calculation failed") from e

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray, normalize: Optional[str] = None
    ) -> np.ndarray:
        """Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization strategy ('true', 'pred', 'all', or None)

        Returns:
            np.ndarray: Confusion matrix
        """

        return cast(np.ndarray, metrics.confusion_matrix(y_true, y_pred, normalize=normalize))


class RegressionMetrics:
    """Utility class for regression metrics calculation."""

    logger = get_logger("RegressionMetrics")

    @staticmethod
    def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.

        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict[str, float]: Dictionary of metric names and values
        """

        try:
            # Basic validation
            if y_true.shape != y_pred.shape:
                raise ValueError("y_true and y_pred must have the same shape.")

            RegressionMetrics.logger.debug(
                "Calculating regression metrics", data_shape=y_true.shape
            )

            result = {
                "explained_variance": metrics.explained_variance_score(y_true, y_pred),
                "max_error": metrics.max_error(y_true, y_pred),
                "mean_absolute_error": metrics.mean_absolute_error(y_true, y_pred),
                "mean_squared_error": metrics.mean_squared_error(y_true, y_pred),
                "root_mean_squared_error": metrics.root_mean_squared_error(y_true, y_pred),
                "median_absolute_error": metrics.median_absolute_error(y_true, y_pred),
                "r2": metrics.r2_score(y_true, y_pred),
                "mean_absolute_percentage_error": metrics.mean_absolute_percentage_error(
                    y_true, y_pred
                ),
                "d2_absolute_error": metrics.d2_absolute_error_score(y_true, y_pred),
                "d2_pinball": metrics.d2_pinball_score(y_true, y_pred),
                "d2_tweedie": metrics.d2_tweedie_score(y_true, y_pred),
                # No negative input values are possible, this leads to errors.
                # "mean_squared_log_error": metrics.mean_squared_log_error(y_true, y_pred),
                # "root_mean_squared_log_error": metrics.root_mean_squared_log_error(
                #     y_true, y_pred
                # ),
                # "mean_poisson_deviance": metrics.mean_poisson_deviance(y_true, y_pred),
                # "mean_gamma_deviance": metrics.mean_gamma_deviance(y_true, y_pred),
            }

            RegressionMetrics.logger.debug("Metrics calculated successfully", metrics=result)

            return result

        except Exception as e:
            RegressionMetrics.logger.error("Failed to calculate metrics", error=e)
            raise RuntimeError("Metrics calculation failed") from e

    @staticmethod
    def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate prediction residuals.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            np.ndarray: Residuals array
        """

        return cast(np.ndarray, y_true - y_pred)


class CrossValidationMetrics:
    """Utility class for cross-validation metrics."""

    logger = get_logger("CrossValidationMetrics")

    @staticmethod
    def aggregate_cv_metrics(
        fold_metrics: List[Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across CV folds.

        Args:
            fold_metrics: List of metric dictionaries from each fold

        Returns:
            Dictionary with mean and std for each metric
        """

        try:
            # Basic validation
            if fold_metrics is None or len(fold_metrics) == 0:
                raise ValueError("fold_metrics must not be empty.")

            if not all(len(fold) == len(fold_metrics[0]) for fold in fold_metrics):
                raise ValueError("All fold metrics must have the same number of elements.")

            all_metrics = {}

            CrossValidationMetrics.logger.debug(
                "Aggregating cross-validation metrics",
                n_folds=len(fold_metrics),
            )

            # Get all unique metric names
            metric_names = set().union(*[d.keys() for d in fold_metrics])

            for metric in metric_names:
                values = [d[metric] for d in fold_metrics if metric in d]
                all_metrics[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

            CrossValidationMetrics.logger.debug(
                "Metrics aggregated successfully", metrics=all_metrics
            )

            return all_metrics

        except Exception as e:
            CrossValidationMetrics.logger.error(
                "Failed to aggregate cross-validation metrics", error=e
            )
            raise RuntimeError("Metrics aggregation failed") from e


if __name__ == "__main__":
    pass
