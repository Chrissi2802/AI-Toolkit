from abc import ABC
from dataclasses import dataclass, field
from operator import gt, lt
from typing import Any, Callable, Dict, List, Union

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema, TensorSpec

from ai_toolkit.base.models import BaseMlModel
from ai_toolkit.utils.evaluation import CrossValidationMetrics
from ai_toolkit.utils.logging import get_logger


@dataclass
class MetricConfig:
    """Configuration for optimization metrics."""

    DIRECTION: str = field(
        default="maximize",
        metadata={"description": "Direction of optuna optimization: maximize or minimize."},
    )
    INITIAL_SCORE: float = field(
        default=float("-inf"),
        metadata={"description": "Initial score for optuna optimization."},
    )
    BETTER_SCORE: Callable[[float, float], bool] = field(
        default=gt,
        metadata={"description": "Function to compare new and old scores."},
    )

    def __post_init__(self):
        """Post initialization checks for configuration."""

        if self.DIRECTION not in ["maximize", "minimize"]:
            raise ValueError("DIRECTION must be 'maximize' or 'minimize'")

        if self.INITIAL_SCORE is not None and not isinstance(self.INITIAL_SCORE, (int, float)):
            raise ValueError("INITIAL_SCORE must be a number")

        if not callable(self.BETTER_SCORE):
            raise ValueError("BETTER_SCORE must be a callable function")


def get_default_metric_configs() -> Dict[str, MetricConfig]:
    """Get default metric configurations.

    Returns:
        Dict[str, MetricConfig]: Dictionary of metric configurations.
    """

    dict_metric_configs = {
        # Classification metrics
        "accuracy": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "balanced_accuracy": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "precision": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "recall": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "f1": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "matthews_correlation_coefficient": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "jaccard": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "hamming_loss": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        # "d2_log_loss": MetricConfig(
        #     DIRECTION="maximize",
        #     INITIAL_SCORE=float("-inf"),
        #     BETTER_SCORE=gt,
        # ),
        "zero_one_loss": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "log_loss": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "roc_auc": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "brier_score": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        # Regression metrics
        "explained_variance": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "max_error": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "mean_absolute_error": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "mean_squared_error": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "root_mean_squared_error": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "median_absolute_error": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "r2": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "mean_absolute_percentage_error": MetricConfig(
            DIRECTION="minimize",
            INITIAL_SCORE=float("inf"),
            BETTER_SCORE=lt,
        ),
        "d2_absolute_error": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "d2_pinball": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        "d2_tweedie": MetricConfig(
            DIRECTION="maximize",
            INITIAL_SCORE=float("-inf"),
            BETTER_SCORE=gt,
        ),
        # "mean_squared_log_error": MetricConfig(
        #     DIRECTION="minimize",
        #     INITIAL_SCORE=float("inf"),
        #     BETTER_SCORE=lt,
        # ),
        # "root_mean_squared_log_error": MetricConfig(
        #     DIRECTION="minimize",
        #     INITIAL_SCORE=float("inf"),
        #     BETTER_SCORE=lt,
        # ),
        # "mean_poisson_deviance": MetricConfig(
        #     DIRECTION="minimize",
        #     INITIAL_SCORE=float("inf"),
        #     BETTER_SCORE=lt,
        # ),
        # "mean_gamma_deviance": MetricConfig(
        #     DIRECTION="minimize",
        #     INITIAL_SCORE=float("inf"),
        #     BETTER_SCORE=lt,
        # ),
    }

    return dict_metric_configs


@dataclass
class MlTrainerConfig:
    """Configuration for ML training."""

    N_SPLITS: int = field(default=5, metadata={"description": "Number of cross-validation splits."})
    RANDOM_STATE: int = field(
        default=28, metadata={"description": "Random state for reproducibility."}
    )
    N_TRAILS: int = field(default=100, metadata={"description": "Number of optimization trials."})
    EXPERIMENT_NAME: str = field(
        default="ml_classification",
        metadata={"description": "Name of the MLflow experiment to log results."},
    )
    OPTIMIZE_METRIC: str = field(
        default="f1",
        metadata={"description": "Metric to optimize during hyperparameter optimization."},
    )
    USE_SMOTE: bool = field(
        default=True,
        metadata={
            "description": "Whether to use SMOTE for imbalanced datasets. Only for classification."
        },
    )
    SMOTE_RATIO: float = field(
        default=1.0,
        metadata={"description": "Ratio of minority to majority class after SMOTE."},
    )

    METRIC_CONFIGS: MetricConfig = field(
        init=False,
        metadata={"description": "Configuration for the chosen optimization metric."},
    )

    def __post_init__(self):
        """Post initialization checks for configuration."""

        if self.N_SPLITS < 2:
            raise ValueError("N_SPLITS must be >= 2")

        if self.RANDOM_STATE < 0:
            raise ValueError("RANDOM_STATE must be >= 0")

        if self.N_TRAILS < 1:
            raise ValueError("N_TRAILS must be >= 1")

        if not self.EXPERIMENT_NAME:
            raise ValueError("EXPERIMENT_NAME cannot be empty")

        if not self.OPTIMIZE_METRIC:
            raise ValueError("OPTIMIZE_METRIC cannot be empty")

        if not isinstance(self.USE_SMOTE, bool):
            raise ValueError("USE_SMOTE must be boolean")

        if not self.SMOTE_RATIO:
            raise ValueError("SMOTE_RATIO cannot be empty")

        default_metric_configs = get_default_metric_configs()

        if self.OPTIMIZE_METRIC not in default_metric_configs:
            raise ValueError(
                f"OPTIMIZE_METRIC '{self.OPTIMIZE_METRIC}' not supported. "
                f"Supported metrics: {list(default_metric_configs.keys())}"
            )

        self.METRIC_CONFIGS = default_metric_configs[self.OPTIMIZE_METRIC]


class BaseMlTrainer(ABC):
    """Abstract base class for all ml trainers."""

    def __init__(
        self,
        base_model: BaseMlModel,
        config: MlTrainerConfig = MlTrainerConfig(),
    ) -> None:
        """Initialize the base ml trainer.

        Args:
            base_model (BaseMlModel): Base model class to be trained.
            config (MlTrainerConfig, optional):
                Configuration for ML training. Defaults to MlTrainerConfig.
        """

        self.base_model = base_model
        self.config = config
        self.n_splits = self.config.N_SPLITS
        self.random_state = self.config.RANDOM_STATE
        self.experiment_name = self.config.EXPERIMENT_NAME
        self.optimize_metric = self.config.OPTIMIZE_METRIC
        self.metric_configs = self.config.METRIC_CONFIGS

        self.best_model = None
        self.feature_names = None

        # Depends on the metric to optimize
        self.best_score = self.metric_configs.INITIAL_SCORE

        mlflow.set_experiment(self.experiment_name)

        # Initialize logger
        self.logger = get_logger(f"{self.__class__.__name__}_{base_model.model_name}")
        self.logger.info(
            "Initializing trainer",
            model_name=base_model.model_name,
            config=config.__dict__,
        )

    def _log_training_info(self, n_trials: int) -> None:
        """Log training parameters to MLflow.

        Args:
            n_trials (int): Number of optimization trials
        """

        mlflow.log_params(
            {
                "model_name": self.base_model.model_name,
                "n_splits": self.n_splits,
                "random_state": self.random_state,
                "n_trials": n_trials,
                "optimize_metric": self.optimize_metric,
            }
        )

    def _log_dataset_info(self, X: np.ndarray, y: np.ndarray) -> None:
        """Log dataset characteristics to MLflow.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
        """

        y = y.astype(int)

        mlflow.log_params(
            {
                "n_samples": len(X),
                "n_features": X.shape[1],
            }
        )

        mlflow.log_table(data=pd.DataFrame(self.feature_names), artifact_file="feature_names.json")

    def _log_final_metrics(self, metrics_stats: Dict[str, Dict[str, float]]) -> None:
        """Log stats of all metrics across all folds.

        Args:
            metrics_stats (Dict[str, Dict[str, float]]): Metric stats dictionary
        """

        for metric_name, stats in metrics_stats.items():
            for stat_name, value in stats.items():
                mlflow.log_metric(f"{metric_name}_{stat_name}", value)

    def _log_fold_results(self, fold: int, metrics: Dict[str, float]) -> None:
        """Log metrics and plots for a specific fold.

        Args:
            fold (int): Fold number
            metrics (Dict[str, float]): Metrics dictionary
        """

        # Log fold metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{metric_name}_fold_{fold}", value)

    def _log_final_results(
        self,
        all_metrics: List[Dict[str, float]],
        all_predictions: Dict[str, np.ndarray],
        X_array: Union[np.ndarray, tf.Tensor],
    ) -> Dict[str, float]:
        """Log final results and save in MLflow.

        Args:
            all_metrics (List[Dict[str, float]]): All metrics across all folds.
            all_predictions (Dict[str, np.ndarray]): All predictions across all folds.
            X_array Union[np.ndarray, tf.Tensor]: Feature matrix.

        Returns:
            Dict[str, float]: Mean metrics across all folds.
        """

        all_metrics_stats = CrossValidationMetrics.aggregate_cv_metrics(all_metrics)

        mean_metrics = {metric: values["mean"] for metric, values in all_metrics_stats.items()}

        self._log_final_metrics(all_metrics_stats)

        # Handle different lengths of predictions
        # Find maximum length of predictions
        max_length = max(len(v) for v in all_predictions.values())

        # Fill up shorter predictions with NaNs
        all_predictions_padded = {
            k: np.pad(
                v.flatten().astype(float),
                (0, max_length - len(v)),
                constant_values=np.nan,
            )
            for k, v in all_predictions.items()
        }

        # Save predictions in MLflow
        mlflow.log_table(
            data=pd.DataFrame(all_predictions_padded),
            artifact_file="predictions.json",
        )

        output_schema = Schema([ColSpec("double", "target")])

        # Create input example
        if isinstance(X_array, np.ndarray):
            # Create model signature
            signature = ModelSignature(
                inputs=Schema([ColSpec("double", name) for name in self.feature_names]),
                outputs=output_schema,
            )

            # Create input example
            input_example = pd.DataFrame(X_array[:5], columns=self.feature_names)

            # Log the best model
            mlflow.sklearn.log_model(
                self.best_model,
                "model",
                signature=signature,
                input_example=input_example,
            )
        elif isinstance(X_array, tf.Tensor):
            # Create model signature
            signature = ModelSignature(
                inputs=Schema(
                    [
                        TensorSpec(
                            shape=(-1,) + tuple(X_array.shape[1:].as_list()),
                            type=np.dtype(X_array.dtype.as_numpy_dtype),
                            name=self.best_model.input_names[0],
                        )
                    ]
                ),
                outputs=output_schema,
            )

            # Create input example
            input_example = X_array[:5].numpy()

            # Log the best model
            mlflow.tensorflow.log_model(
                self.best_model,
                "model",
                signature=signature,
                input_example=input_example,
                keras_model_kwargs={"save_format": "tf", "save_traces": True},
            )

        # Log best_params
        mlflow.log_table(
            data=pd.DataFrame([self.base_model.best_params]),
            artifact_file="best_params.json",
        )

        # Print results
        print(f"\n# Model: {self.base_model.model_name}")
        print("\n## Best Hyperparameters:", self.base_model.best_params)
        print(f"\n## Optimize metric '{self.optimize_metric}' for each fold:")
        for i, metrics in enumerate(all_metrics):
            print(f"Fold {i+1} Score: {metrics[self.optimize_metric]:.4f}")
        print("\n## Mean Metrics across all folds:")
        for metric, value in mean_metrics.items():
            print(f"{metric}: {value:.4f}")

        return mean_metrics

    def _calc_feature_importance(self, model: Any) -> np.ndarray:
        """Calculate feature importance scores for a trained model.

        Args:
            model (Any): Trained model.

        Returns:
            np.ndarray: Feature importance scores.
        """

        # Models with feature_importances_ attribute
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_

        # Linear models with coefficients
        elif hasattr(model, "coef_"):
            coef = model.coef_

            # Handle multi-target case
            if coef.ndim > 1:
                return np.mean(np.abs(coef), axis=0)

            return np.abs(coef)

        # Models with no feature importance
        else:
            return None

    def array_indexing(
        self,
        data: Union[np.ndarray, tf.Tensor],
        indices: Union[np.ndarray, tf.Tensor, list],
    ) -> Union[np.ndarray, tf.Tensor]:
        """Flexibly indexing either NumPy arrays or TensorFlow tensors.

        Args:
            data (Union[np.ndarray, tf.Tensor]):
                Input data as either NumPy array or TensorFlow tensor
            indices (Union[np.ndarray, tf.Tensor, list]): Indices to use for indexing

        Returns:
            Union[np.ndarray, tf.Tensor]:  Indexed data in the same format as input
        """

        if isinstance(data, np.ndarray):
            # Convert indices to numpy array if needed
            return data[indices]

        elif isinstance(data, tf.Tensor):
            # Convert indices to tensorflow tensor if needed
            if not isinstance(indices, tf.Tensor):
                indices = tf.convert_to_tensor(indices, dtype=tf.int32)
            return tf.gather(data, indices)


if __name__ == "__main__":
    pass
