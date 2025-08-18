from abc import ABC
from typing import Any, Dict, List, Union

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema, TensorSpec

from ai_toolkit.base.config import ConfigFactory
from ai_toolkit.base.models import BaseMlModel
from ai_toolkit.utils.evaluation import CrossValidationMetrics
from ai_toolkit.utils.logging import get_logger


class BaseMlTrainer(ABC):
    """Abstract base class for all ml trainers."""

    def __init__(
        self, base_model: BaseMlModel, config_factory: ConfigFactory = ConfigFactory()
    ) -> None:
        """Initialize the base ml trainer.

        Args:
            base_model (BaseMlModel): Base model class to be trained.
            config_factory (ConfigFactory, optional):
                Configuration for ML training. Defaults to ConfigFactory.
        """

        self.base_model = base_model
        self.config_factory = config_factory
        self.config = self.config_factory.get_config().training
        self.n_splits = self.config.n_splits
        self.random_state = self.config.random_state
        self.experiment_name = self.config.experiment_name
        self.optimize_metric = self.config.optimize_metric
        self.metric_configs = self.config.get_metric_configs()

        self.best_model = None
        self.feature_names = None

        # Depends on the metric to optimize
        self.best_score = self.metric_configs.initial_score

        mlflow.set_experiment(self.experiment_name)

        # Initialize logger
        self.logger = get_logger(f"{self.__class__.__name__}_{base_model.model_name}")
        self.logger.info(
            "Initializing trainer",
            model_name=base_model.model_name,
            config=self.config.__dict__,
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
