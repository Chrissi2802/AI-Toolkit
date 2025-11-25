from datetime import datetime
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ai_toolkit.base.config import ConfigFactory
from ai_toolkit.base.models import BaseMlEnsembleModel, BaseMlModel
from ai_toolkit.base.training import BaseMlTrainer
from ai_toolkit.utils.evaluation import ClassificationMetrics
from ai_toolkit.utils.logging import get_logger
from ai_toolkit.utils.visualization import ClassificationPlots, ModelAnalysisPlots


class ClassificationModelTrainer(BaseMlTrainer):
    """Handles classification model training and optimization."""

    def __init__(
        self,
        base_model: Union[BaseMlModel, BaseMlEnsembleModel],
        config_factory: ConfigFactory = ConfigFactory(),
    ) -> None:
        """Initialize the ModelTrainer.

        Args:
            base_model (Union[BaseMlModel, BaseMlEnsembleModel]):
                Base model or ensemble model class to be trained.
            config_factory (ConfigFactory, optional):
                Configuration for ML training. Defaults to ConfigFactory().
        """

        super().__init__(
            base_model=base_model,
            config_factory=config_factory,
        )

        self.use_smote = self.config.use_smote
        self.smote_ratio = self.config.smote_ratio
        self.smote = None

    def _log_training_info(self, n_trials: int) -> None:
        """Log training parameters to MLflow.

        Args:
            n_trials (int): Number of optimization trials
        """

        super()._log_training_info(n_trials)
        mlflow.log_params(
            {
                "use_smote": self.use_smote,
                "smote_ratio": self.smote_ratio,
            }
        )

    def _log_dataset_info(self, X: np.ndarray, y: np.ndarray) -> None:
        """Log dataset characteristics to MLflow.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
        """

        y = y.astype(int)

        super()._log_dataset_info(X, y)

        params = {"class_distribution": str(pd.Series(y).value_counts().sort_index().to_dict())}

        if len(np.unique(y)) == 2:
            params["class_ratio"] = f"{np.bincount(y)[0]}/{np.bincount(y)[1]}"

        mlflow.log_params(params)

    def _log_fold_results(
        self,
        fold: int,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model: Any,
    ) -> None:
        """Log metrics and plots for a specific fold.

        Args:
            fold (int): Fold number
            metrics (Dict[str, float]): Metrics dictionary
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model (Any): Trained model.
        """

        # Log fold metrics
        super()._log_fold_results(fold, metrics)

        if y_pred_proba is not None and np.unique(y_true).shape[0] == 2:
            # Create and log plot for ROC curve
            roc_fig = ClassificationPlots.plot_roc_curve(
                y_true,
                y_pred_proba[:, 1],
                f"ROC Curve - Fold: {fold}",
            )
            mlflow.log_figure(roc_fig, f"fold_{fold}_roc_curve.png")
            plt.close(roc_fig)

        # Create and log plot for confusion matrix
        cm_fig = ClassificationPlots.plot_confusion_matrix(
            y_true,
            y_pred,
            f"Confusion Matrix - Fold: {fold}",
        )
        mlflow.log_figure(cm_fig, f"fold_{fold}_confusion_matrix.png")
        plt.close(cm_fig)

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

    def _optimize_objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
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

            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            scores = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train = self.array_indexing(X, train_idx)
                X_val = self.array_indexing(X, val_idx)
                y_train = self.array_indexing(y, train_idx)
                y_val = self.array_indexing(y, val_idx)

                # Apply SMOTE if enabled
                if self.use_smote:
                    X_train, y_train = self.smote.fit_resample(X_train, y_train)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_val)
                else:
                    y_pred_proba = None

                # Calculate metrics
                metrics = ClassificationMetrics.calculate_basic_metrics(y_val, y_pred, y_pred_proba)
                scores.append(metrics[self.optimize_metric])

                self.logger.debug(f"Fold {fold+1} score: {metrics[self.optimize_metric]:.4f}")

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

    def _setup_smote(self, y: pd.Series) -> None:
        """Setup SMOTE for handling class imbalance.

        Args:
            y (pd.Series): Target vector
        """

        max_count = max(y.value_counts())
        sampling_strategy = {
            cls: (
                count
                if int(max_count * self.smote_ratio) <= count
                else int(max_count * self.smote_ratio)
            )
            for cls, count in y.value_counts().sort_index().items()
        }

        # Synthetic Minority Over-sampling Technique (SMOTE)
        self.smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)

    def _track_best_model(
        self,
        X_val: np.ndarray,
        y_array: np.ndarray,
        y_val: np.ndarray,
        y_pred: np.ndarray,
        fold: int,
        y_pred_proba: Union[np.ndarray, None] = None,
    ) -> None:
        """Track the best model and log relevant information.

        Args:
            X_val (np.ndarray): Validation feature matrix
            y_array (np.ndarray): Target vector
            y_val (np.ndarray): Validation target vector
            y_pred (np.ndarray): Predicted labels
            fold (int): Fold number
            y_pred_proba (Union[np.ndarray, None], optional): Predicted probabilities.
                Defaults to None.
        """

        # Log best model plots
        mlflow.log_metric("best_score", self.best_score)

        if y_pred_proba is not None and np.unique(y_array).shape[0] == 2:
            # Create and log plot for ROC curve
            roc_fig = ClassificationPlots.plot_roc_curve(
                y_val,
                y_pred_proba[:, 1],
                f"ROC Curve - Best Model Fold: {fold}",
            )
            mlflow.log_figure(roc_fig, "best_roc_curve.png")
            plt.close(roc_fig)

        # Create and log plot for confusion matrix
        cm_fig = ClassificationPlots.plot_confusion_matrix(
            y_val,
            y_pred,
            f"Confusion Matrix - Best Model Fold: {fold}",
        )
        mlflow.log_figure(cm_fig, "best_confusion_matrix.png")
        plt.close(cm_fig)

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

        if isinstance(X_val, np.ndarray):
            # Create and log plot for SHAP values
            shap_fig = ModelAnalysisPlots.plot_shapley_values(
                self.best_model,
                X_val,
                self.feature_names,
            )
            mlflow.log_figure(shap_fig, "best_shap_values.png")
            plt.close(shap_fig)

    def train_and_optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train and optimize a ml model for classification.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            n_trials (int, optional): Number of optimization trials. Defaults to None.

        Returns:
            Tuple[Any, Dict[str, float]]: Best model and mean metrics
        """

        if n_trials is None:
            n_trials = self.config.n_trials

        if self.use_smote:
            self._setup_smote(y)

        # Set number of classes for the model
        self.base_model.set_num_classes(y.nunique())

        with mlflow.start_run(run_name=f"{self.base_model.model_name}_{datetime.now()}"):
            # Store feature names and convert to numpy arrays
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
                X_array = X.values
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X_array = X

            y_array = y.values

            # Log information
            self._log_training_info(n_trials)
            self._log_dataset_info(X_array, y_array)

            # Encode target labels
            self.encoder = LabelEncoder()
            y_array = self.encoder.fit_transform(y_array)

            # Optimize hyperparameters
            study = optuna.create_study(
                direction=self.metric_configs.direction,  # Depends on the metric to be optimized
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
            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            all_metrics = []
            all_predictions = {}

            for fold, (train_idx, val_idx) in enumerate(
                tqdm(
                    skf.split(X_array, y_array),
                    total=self.n_splits,
                    desc="Cross-validation",
                )
            ):
                X_train = self.array_indexing(X_array, train_idx)
                X_val = self.array_indexing(X_array, val_idx)
                y_train = self.array_indexing(y_array, train_idx)
                y_val = self.array_indexing(y_array, val_idx)

                if self.use_smote:
                    X_train, y_train = self.smote.fit_resample(X_train, y_train)

                # Create and train model
                model = self.base_model.create_model(self.base_model.best_params)

                # Check and convert to DataFrame if necessary
                if X_train.ndim == 2:
                    X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
                    X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
                else:
                    # X_train.ndim > 2
                    X_train_df = X_train
                    X_val_df = X_val

                # Fit the model
                model.fit(X_train_df, y_train)

                # Make predictions
                y_pred = model.predict(X_val_df)

                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_val_df)
                else:
                    y_pred_proba = None

                # Inverse transform target labels
                y_val = self.encoder.inverse_transform(y_val)
                y_pred = self.encoder.inverse_transform(y_pred)

                # Store predictions
                all_predictions[f"fold_{fold}"] = y_pred

                if y_pred_proba is not None:
                    all_predictions[f"fold_{fold}_proba"] = y_pred_proba[:, 1]

                # Calculate metrics
                metrics = ClassificationMetrics.calculate_basic_metrics(y_val, y_pred, y_pred_proba)
                all_metrics.append(metrics)

                # Log fold results
                self._log_fold_results(fold, metrics, y_val, y_pred, y_pred_proba, model)

                # Track best model based on specified metric
                if self.metric_configs.better_score(
                    metrics[self.optimize_metric], self.best_score
                ):  # Depends on the metric to be optimized
                    self.best_score = metrics[self.optimize_metric]
                    self.best_model = model

                    self._track_best_model(
                        X_val=X_val,
                        y_array=y_array,
                        y_val=y_val,
                        y_pred=y_pred,
                        fold=fold,
                        y_pred_proba=y_pred_proba,
                    )

            mean_metrics = self._log_final_results(all_metrics, all_predictions, X_array)

            return self.best_model, mean_metrics

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the best model.

        Args:
            X (pd.DataFrame): Features

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions and predicted probabilities.
        """

        try:
            self.logger.info("Making predictions", X_shape=X.shape)

            if self.best_model is None:
                raise ValueError("No model trained yet. Please call train_and_optimize first.")

            y_pred = self.best_model.predict(X)

            if hasattr(self.best_model, "predict_proba"):
                y_pred_proba = self.best_model.predict_proba(X)
            else:
                y_pred_proba = None

            # Inverse transform target labels
            y_pred = self.encoder.inverse_transform(y_pred)

            self.logger.info("Predictions completed", predictions_shape=y_pred.shape)

            return y_pred, y_pred_proba

        except Exception as e:
            self.logger.error("Prediction failed", error=e)
            raise RuntimeError("Prediction failed") from e


def lazypredict_classification(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 28,
) -> pd.DataFrame:
    """Run LazyPredict classification on the dataset.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 28.

    Returns:
        pd.DataFrame: Mean metrics across all folds.
    """

    logger = get_logger("Lazypredict classification")
    logger.info(
        "Starting lazypredict classification",
        X_shape=X.shape,
        y_shape=y.shape,
        n_splits=n_splits,
    )

    try:
        X_array = X.values
        y_array = y.values

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Store results for each fold
        fold_results = []

        # Run cross-validation
        for fold, (train_idx, val_idx) in enumerate(
            tqdm(skf.split(X_array, y_array), total=n_splits, desc="Cross-Validation")
        ):
            logger.debug(f"Processing fold {fold+1}")

            X_train, X_val = X_array[train_idx], X_array[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]

            # Create and train LazyClassifier
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, _ = clf.fit(X_train, X_val, y_train, y_val)

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
                    "Accuracy": "mean",
                    "Balanced Accuracy": "mean",
                    "ROC AUC": "mean",
                    "F1 Score": "mean",
                    "Time Taken": "mean",
                }
            )
            .round(4)
        ).sort_values("Balanced Accuracy", ascending=False)

        # Add standard deviation of accuracy as additional information
        accuracy_std = all_results.groupby(all_results.index)["Accuracy"].std().round(4)
        mean_results["Accuracy Std"] = accuracy_std

        logger.info(
            "Lazypredict completed",
            n_models=len(mean_results),
            best_model=mean_results.index[0],
            best_accuracy=mean_results["Accuracy"].iloc[0],
        )

        return mean_results

    except Exception as e:
        logger.error("Lazypredict failed", error=e)
        raise RuntimeError("Lazypredict classification failed") from e


if __name__ == "__main__":
    pass
