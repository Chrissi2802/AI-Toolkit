from datetime import datetime
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm

from ai_toolkit.base.models import BaseMlEnsembleModel, BaseMlModel
from ai_toolkit.base.training import BaseMlTrainer, MlTrainerConfig
from ai_toolkit.utils.evaluation import RegressionMetrics
from ai_toolkit.utils.logging import get_logger
from ai_toolkit.utils.visualization import ModelAnalysisPlots, RegressionPlots


class RegressionModelTrainer(BaseMlTrainer):
    """Handles regression model training and optimization."""

    def __init__(
        self,
        base_model: Union[BaseMlModel, BaseMlEnsembleModel],
        config: MlTrainerConfig = MlTrainerConfig(),
    ) -> None:
        """Initialize the RegressionModelTrainer.

        Args:
            base_model (Union[BaseMlModel, BaseMlEnsembleModel]):
                Base model or ensemble model class to be trained.
            config (MlTrainerConfig, optional):
                Configuration for ML training. Defaults to MlTrainerConfig.
        """

        super().__init__(
            base_model=base_model,
            config=config,
        )

    def _log_dataset_info(self, X: np.ndarray, y: np.ndarray) -> None:
        """Log dataset characteristics to MLflow.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
        """

        super()._log_dataset_info(X, y)

        mlflow.log_params(
            {
                "target_mean": np.mean(y),
                "target_std": np.std(y),
                "target_median": np.median(y),
                "target_min": np.min(y),
                "target_max": np.max(y),
            }
        )

    def _log_fold_results(
        self,
        fold: int,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
    ) -> None:
        """Log metrics and plots for a specific fold.

        Args:
            fold (int): Fold number
            metrics (Dict[str, float]): Metrics dictionary
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model (Any): Trained model.
        """

        # Log fold metrics
        super()._log_fold_results(fold, metrics)

        # Create and log plots for residuals
        residuals_fig = RegressionPlots.plot_residuals(
            y_true, y_pred, f"Residual Analysis - Fold: {fold}"
        )
        mlflow.log_figure(residuals_fig, f"fold_{fold}_residuals.png")
        plt.close(residuals_fig)

        # Create and log plots for prediction scatter
        scatter_fig = RegressionPlots.plot_prediction_scatter(
            y_true, y_pred, f"Actual vs Predicted - Fold: {fold}"
        )
        mlflow.log_figure(scatter_fig, f"fold_{fold}_scatter.png")
        plt.close(scatter_fig)

        # Calculate feature importance
        importance_scores = self._calc_feature_importance(model)

        if importance_scores is not None:
            # Create and log plot for feature importance
            fi_fig = ModelAnalysisPlots.plot_feature_importance(
                importance_scores,
                self.feature_names,
                f"Feature Importance - Fold: {fold}",
            )
            mlflow.log_figure(fi_fig, f"fold_{fold}_feature_importance.png")
            plt.close(fi_fig)

    def _optimize_objective(
        self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Optimize objective function for Optuna hyperparameter search.

        Args:
            trial (optuna.Trial): Optuna trial object
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector

        Returns:
            float: Mean score across all folds
        """

        self.logger.debug(f"Starting optimization trial {trial.number}")

        try:
            params = self.base_model.get_param_space(trial)
            model = self.base_model.create_model(params)

            # Cross-validation evaluation
            kf = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Calculate metrics
                metrics = RegressionMetrics.calculate_basic_metrics(y_val, y_pred)
                scores.append(metrics[self.optimize_metric])

                self.logger.debug(
                    f"Fold {fold+1} score: {metrics[self.optimize_metric]:.4f}"
                )

            mean_score = np.mean(scores)

            self.logger.debug(
                "Trial completed",
                trial_number=trial.number,
                mean_score=mean_score,
                params=params,
            )
            return mean_score

        except Exception as e:
            self.logger.error("Trial failed", trial_number=trial.number, error=e)
            raise RuntimeError("Optimization trial failed") from e

    def train_and_optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train and optimize a ml model for regression.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            n_trials (int, optional): Number of optimization trials. Defaults to 100.

        Returns:
            Tuple[Any, Dict[str, float]]: Best model and mean metrics
        """

        with mlflow.start_run(
            run_name=f"{self.base_model.model_name}_{datetime.now()}"
        ):

            # Store feature names and convert to numpy arrays
            self.feature_names = list(X.columns)
            X_array = X.values
            y_array = y.values

            # Log information
            self._log_training_info(n_trials)
            self._log_dataset_info(X_array, y_array)

            # Optimize hyperparameters
            study = optuna.create_study(
                direction=self.metric_configs.DIRECTION,  # Depends on the metric to be optimized
                study_name=self.base_model.model_name + " optimization",
            )
            study.optimize(
                lambda trial: self._optimize_objective(trial, X_array, y_array),
                n_trials=n_trials,
            )

            # Log optimization history
            mlflow.log_table(
                data=study.trials_dataframe(), artifact_file="optimization_history.json"
            )

            # Log best parameters
            self.base_model.best_params = study.best_params
            mlflow.log_params({k: v for k, v in study.best_params.items()})

            # Cross-validation evaluation
            kf = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            all_metrics = []
            all_predictions = {}

            for fold, (train_idx, val_idx) in enumerate(
                tqdm(kf.split(X_array), total=self.n_splits, desc="Cross-validation")
            ):
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]

                # Create and train model
                model = self.base_model.create_model(self.base_model.best_params)
                X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
                model.fit(X_train_df, y_train)

                # Make predictions
                X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
                y_pred = model.predict(X_val_df)

                # Store predictions
                all_predictions[f"fold_{fold}"] = y_pred

                # Calculate metrics
                metrics = RegressionMetrics.calculate_basic_metrics(y_val, y_pred)
                all_metrics.append(metrics)

                # Log fold results
                self._log_fold_results(fold, metrics, y_val, y_pred, model)

                # Track best model based on specified metric
                if self.metric_configs.BETTER_SCORE(
                    metrics[self.optimize_metric], self.best_score
                ):  # Depends on the metric to be optimized
                    self.best_score = metrics[self.optimize_metric]
                    self.best_model = model

                    # Log best model plots
                    mlflow.log_metric("best_score", self.best_score)

                    # Create and log plots for residuals
                    residuals_fig = RegressionPlots.plot_residuals(
                        y_val, y_pred, f"Residual Analysis - Best Model Fold: {fold}"
                    )
                    mlflow.log_figure(residuals_fig, "best_residuals.png")
                    plt.close(residuals_fig)

                    # Create and log plots for prediction scatter
                    scatter_fig = RegressionPlots.plot_prediction_scatter(
                        y_val, y_pred, f"Actual vs Predicted - Best Model Fold: {fold}"
                    )
                    mlflow.log_figure(scatter_fig, "best_scatter.png")
                    plt.close(scatter_fig)

                    # Calculate feature importance
                    importance_scores = self._calc_feature_importance(self.best_model)

                    if importance_scores is not None:
                        # Create and log plot for feature importance
                        fi_fig = ModelAnalysisPlots.plot_feature_importance(
                            importance_scores,
                            self.feature_names,
                            f"Feature Importance - Best Model Fold: {fold}",
                        )
                        mlflow.log_figure(fi_fig, "best_feature_importance.png")
                        plt.close(fi_fig)

                    # Create and log plot for SHAP values
                    shap_fig = ModelAnalysisPlots.plot_shapley_values(
                        self.best_model,
                        X_val,
                        self.feature_names,
                    )
                    mlflow.log_figure(shap_fig, "best_shap_values.png")
                    plt.close(shap_fig)

            mean_metrics = self._log_final_results(
                all_metrics, all_predictions, X_array
            )

            return self.best_model, mean_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model.

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.ndarray: Predicted values
        """

        try:
            self.logger.info("Making predictions", X_shape=X.shape)

            if self.best_model is None:
                raise ValueError(
                    "No model trained yet. Please call train_and_optimize first."
                )

            y_pred = self.best_model.predict(X)

            self.logger.info("Predictions completed", predictions_shape=y_pred.shape)

            return y_pred

        except Exception as e:
            self.logger.error("Prediction failed", error=e)
            raise RuntimeError("Prediction failed") from e


def lazypredict_regression(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 28
) -> pd.DataFrame:
    """Run cross-validated regression with LazyRegressor.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 28.

    Returns:
        pd.DataFrame: Mean metrics across all folds.
    """

    logger = get_logger("Lazypredict Regression")
    logger.info(
        "Starting lazypredict regression",
        X_shape=X.shape,
        y_shape=y.shape,
        n_splits=n_splits,
    )

    try:
        X_array = X.values
        y_array = y.values

        # Initialize KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Store results for each fold
        fold_results = []

        # Run cross-validation
        for fold, (train_idx, val_idx) in enumerate(
            tqdm(kf.split(X_array), total=n_splits, desc="Cross-Validation")
        ):

            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            # Create and train LazyRegressor
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models, _ = reg.fit(X_train, X_val, y_train, y_val)

            # Add fold number to results
            models["fold"] = fold
            fold_results.append(models)

        # Combine all fold results
        all_results = pd.concat(fold_results, axis=0)

        # Calculate mean metrics across folds
        mean_results = (
            all_results.groupby(all_results.index)
            .agg(
                {
                    "Adjusted R-Squared": "mean",
                    "R-Squared": "mean",
                    "RMSE": "mean",
                    "Time Taken": "mean",
                }
            )
            .round(4)
        ).sort_values("Adjusted R-Squared", ascending=False)

        # Add standard deviation of R-Squared as additional information
        ar2_std = (
            all_results.groupby(all_results.index)["Adjusted R-Squared"].std().round(4)
        )
        mean_results["Adjusted R-Squared Std"] = ar2_std

        logger.info(
            "Lazypredict completed",
            n_models=len(mean_results),
            best_model=mean_results.index[0],
            best_accuracy=mean_results["Adjusted R-Squared"].iloc[0],
        )

        return mean_results

    except Exception as e:
        logger.error("Lazypredict failed", error=e)
        raise RuntimeError("Lazypredict regression failed") from e


if __name__ == "__main__":
    pass
