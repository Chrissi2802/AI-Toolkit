from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold
import optuna
from imblearn.over_sampling import SMOTE
import mlflow
from mlflow.types.schema import Schema, ColSpec
from mlflow.models.signature import ModelSignature
from lazypredict.Supervised import LazyClassifier

from ai_toolkit.base.model import BaseMlModel
from ai_toolkit.utils.evaluation import ClassificationMetrics
from ai_toolkit.utils.visualization import ClassificationPlots


class ClassificationModelTrainer:
    """Handles classification model training and optimization."""

    def __init__(
        self,
        base_model: BaseMlModel,
        n_splits: int = 5,
        random_state: int = 28,
        experiment_name: str = "ml_classification",
        optimize_metric: str = "f1",
        use_smote: bool = True,
        smote_ratio: float = 1.0,
    ) -> None:
        """Initialize the ModelTrainer.

        Args:
            base_model (BaseModel): Base model class to be trained.
            n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
            random_state (int, optional): Random state for reproducibility. Defaults to 28.
            experiment_name (str, optional): MLflow experiment name.
                Defaults to "ml_classification".
            optimize_metric (str, optional): Metric to optimize during hyperparameter search.
                Defaults to "f1".
            use_smote (bool, optional): Whether to use SMOTE. Defaults to True.
            smote_ratio (float, optional): SMOTE sampling ratio. Defaults to 1.0.
        """

        self.base_model = base_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.experiment_name = experiment_name
        self.optimize_metric = optimize_metric
        self.use_smote = use_smote
        self.smote_ratio = smote_ratio

        self.smote = (
            SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
            if use_smote
            else None
        )  # Synthetic Minority Over-sampling Technique (SMOTE)

        self.best_model = None
        self.best_score = -1  # ! Depends on the metric to optimize
        self.feature_names = None

        mlflow.set_experiment(self.experiment_name)

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

        mlflow.log_params(
            {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "class_distribution": str(np.bincount(y)),
                "class_ratio": f"{np.bincount(y)[0]}/{np.bincount(y)[1]}",
            }
        )

        mlflow.log_dict(self.feature_names, "feature_names.json")

    def _log_final_metrics(self, mean_metrics: Dict[str, float]) -> None:
        """Log mean metrics across all folds.

        Args:
            mean_metrics (Dict[str, float]): Mean metrics dictionary
        """

        for metric_name, value in mean_metrics.items():
            mlflow.log_metric(f"mean_{metric_name}", value)

    def _log_fold_results(
        self,
        fold: int,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> None:
        """Log metrics and plots for a specific fold.

        Args:
            fold (int): Fold number
            metrics (Dict[str, float]): Metrics dictionary
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
        """

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"fold_{fold}_{metric_name}", value)

        # Create and log plot for ROC curve
        roc_fig = ClassificationPlots.plot_roc_curve(
            y_true,
            y_pred_proba,
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

        params = self.base_model.get_param_space(trial)
        model = self.base_model.create_model(params)

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Apply SMOTE if enabled
            if self.use_smote:
                X_train, y_train = self.smote.fit_resample(X_train, y_train)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Calculate metrics
            metrics = ClassificationMetrics.calculate_basic_metrics(y_val, y_pred)
            scores.append(metrics[self.optimize_metric])

        return np.mean(scores)

    def train_and_optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train and optimize a ml model for classification.

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
                direction="maximize",  # ! Depends on the metric to be optimized
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
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train, y_val = y_array[train_idx], y_array[val_idx]

                if self.use_smote:
                    X_train, y_train = self.smote.fit_resample(X_train, y_train)

                # Create and train model
                model = self.base_model.create_model(self.base_model.best_params)
                X_train_df = pd.DataFrame(X_train, columns=self.feature_names)
                model.fit(X_train_df, y_train)

                # Make predictions
                X_val_df = pd.DataFrame(X_val, columns=self.feature_names)
                y_pred = model.predict(X_val_df)
                y_pred_proba = model.predict_proba(X_val_df)[:, 1]

                # Store predictions
                all_predictions[f"fold_{fold}"] = y_pred
                all_predictions[f"fold_{fold}_proba"] = y_pred_proba

                # Calculate metrics
                metrics = ClassificationMetrics.calculate_basic_metrics(
                    y_val, y_pred, y_pred_proba
                )
                all_metrics.append(metrics)

                # Log fold results
                self._log_fold_results(fold, metrics, y_val, y_pred, y_pred_proba)

                # Track best model based on specified metric
                if (
                    metrics[self.optimize_metric] > self.best_score
                ):  # ! Depends on the metric to be optimized
                    self.best_score = metrics[self.optimize_metric]
                    self.best_model = model

                    # Log best model plots
                    mlflow.log_metric("best_score", self.best_score)

                    # Create and log plot for ROC curve
                    roc_fig = ClassificationPlots.plot_roc_curve(
                        y_val,
                        y_pred_proba,
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

            # Calculate and log mean metrics
            mean_metrics = {
                metric: np.mean([m[metric] for m in all_metrics])
                for metric in all_metrics[0].keys()
            }
            self._log_final_metrics(mean_metrics)

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

            # Create and log model signature
            signature = ModelSignature(
                inputs=Schema([ColSpec("double", name) for name in self.feature_names]),
                outputs=Schema([ColSpec("double", "target")]),
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

            # Print results
            print(f"\n# Model: {self.base_model.model_name}")
            print("\n## Best Hyperparameters:", self.base_model.best_params)
            print(f"\n## Optimize metric '{self.optimize_metric}' for each fold:")
            for i, metrics in enumerate(all_metrics):
                print(f"Fold {i+1} Score: {metrics[self.optimize_metric]:.4f}")
            print("\n## Mean Metrics across all folds:")
            for metric, value in mean_metrics.items():
                print(f"{metric}: {value:.4f}")

            return self.best_model, mean_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model.

        Args:
            X (pd.DataFrame): Features

        Returns:
            np.ndarray: Predicted labels
        """

        if self.best_model is None:
            raise ValueError(
                "No model trained yet. Please call train_and_optimize first."
            )

        y_pred = self.best_model.predict(X)

        return y_pred


def lazypredict_classification(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 28,
) -> pd.DataFrame:
    """

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 28.

    Returns:
        pd.DataFrame: Mean metrics across all folds.
    """

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

    return mean_results


if __name__ == "__main__":
    pass
