from typing import Any, Dict, Sequence, Tuple, cast

import lightgbm as lgb
import optuna
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.applications import MobileNetV3Small

from ai_toolkit.base.models import BaseMlEnsembleModel, BaseMlModel
from ai_toolkit.utils.logging import get_logger


class LogisticRegressionModel(BaseMlModel):
    """Linear based: Logistic Regression Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    def __init__(self) -> None:
        """Initialize the Logistic Regression model."""

        super().__init__("Logistic Regression Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the Logistic Regression model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        # Base parameters
        params = {
            "penalty": penalty,
            "solver": trial.suggest_categorical("solver", ["saga"]),
            "C": trial.suggest_float("C", 1e-5, 100, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 2000),  # Increased for convergence
            "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state": trial.suggest_categorical("random_state", [28]),
        }

        # Add l1_ratio only for elasticnet
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)

        return params

    def create_model(self, params: Dict[str, Any]) -> LogisticRegression:
        """Create a Logistic Regression model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            LogisticRegression: A Logistic Regression model.
        """

        try:
            self.logger.info("Creating logistic regression model", params=params)
            self.model = LogisticRegression(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create logistic regression model", error=e)
            raise RuntimeError("Model creation failed") from e


class SVCModel(BaseMlModel):
    """Kernel based: Support Vector Machine Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """

    def __init__(self) -> None:
        """Initialize the Support Vector Machine model."""

        super().__init__("Support Vector Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the SVM model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])

        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "probability": trial.suggest_categorical("probability", [True]),
            "random_state": trial.suggest_categorical("random_state", [28]),
        }

        # Add kernel-specific parameters
        if kernel in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)

        return params

    def create_model(self, params: Dict[str, Any]) -> SVC:
        """Create a SVM model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            SVC: A Support Vector Machine model.
        """

        try:
            self.logger.info("Creating support vector classifier model", params=params)
            self.model = SVC(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create support vector classifier model", error=e)
            raise RuntimeError("Model creation failed") from e


class KNNModel(BaseMlModel):
    """Instance based: K-Nearest Neighbors Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    def __init__(self) -> None:
        """Initialize the K-Nearest Neighbors model."""

        super().__init__("K-Nearest Neighbors Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the KNN model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50),
            "p": trial.suggest_int("p", 1, 2),  # 1 for manhattan_distance, 2 for euclidean_distance
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> KNeighborsClassifier:
        """Create a KNN model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            KNeighborsClassifier: A K-Nearest Neighbors model.
        """

        try:
            self.logger.info("Creating K-Nearest Neighbors model", params=params)
            self.model = KNeighborsClassifier(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create K-Nearest Neighbors model", error=e)
            raise RuntimeError("Model creation failed") from e


class NaiveBayesModel(BaseMlModel):
    """Bayesian based: Naive Bayes Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    """

    def __init__(self) -> None:
        """Initialize the Naive Bayes model."""

        super().__init__("Naive Bayes Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the Naive Bayes model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-8, log=True),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> GaussianNB:
        """Create a Naive Bayes model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            GaussianNB: A Gaussian Naive Bayes model.
        """

        try:
            self.logger.info("Creating Naive Bayes model", params=params)
            self.model = GaussianNB(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create Naive Bayes model", error=e)
            raise RuntimeError("Model creation failed") from e


class DecisionTreeModel(BaseMlModel):
    """Tree based: Decision Tree Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    def __init__(self) -> None:
        """Initialize the Decision Tree model."""

        super().__init__("Decision Tree Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the Decision Tree model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state": trial.suggest_categorical("random_state", [28]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 1.0),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> DecisionTreeClassifier:
        """Create a Decision Tree model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            DecisionTreeClassifier: A Decision Tree model.
        """

        try:
            self.logger.info("Creating decision tree model", params=params)
            self.model = DecisionTreeClassifier(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create decision tree model", error=e)
            raise RuntimeError("Model creation failed") from e


class RandomForestModel(BaseMlModel):
    """Ensemble based: Random Forest Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(self) -> None:
        """Initialize the Random Forest model."""

        super().__init__("Random Forest Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the Random Forest model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]
            ),
            "random_state": trial.suggest_categorical("random_state", [28]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> RandomForestClassifier:
        """Create a Random Forest model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            RandomForestClassifier: A Random Forest model.
        """

        try:
            self.logger.info("Creating random forest model", params=params)
            self.model = RandomForestClassifier(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create random forest model", error=e)
            raise RuntimeError("Model creation failed") from e


class XGBoostModel(BaseMlModel):
    """Gradient Boosting: Extreme Gradient Boosting Model.
    https://xgboost.readthedocs.io/en/stable/parameter.html
    """

    def __init__(self) -> None:
        """Initialize the XGBoost model."""

        super().__init__("XGBoost Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the XGBoost model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": trial.suggest_categorical("random_state", [28]),
        }

        # For multi-class classification
        if self.is_multiclass:
            params["objective"] = trial.suggest_categorical("objective", ["multi:softmax"])
            params["num_class"] = trial.suggest_categorical("num_class", [self.num_classes])
        else:
            params["objective"] = trial.suggest_categorical("objective", ["binary:logistic"])

        return params

    def create_model(self, params: Dict[str, Any]) -> xgb.XGBClassifier:
        """Create an XGBoost model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            xgb.XGBClassifier: An XGBoost model.
        """

        try:
            self.logger.info("Creating XGBoost model", params=params)
            self.model = xgb.XGBClassifier(**params)
            return cast(xgb.XGBClassifier, self.model)
        except Exception as e:
            self.logger.error("Failed to create XGBoost model", error=e)
            raise RuntimeError("Model creation failed") from e


class LightGBMModel(BaseMlModel):
    """Gradient Boosting: Light Gradient-Boosting Machine (LightGBM) Model.
    https://lightgbm.readthedocs.io/en/latest/Parameters.html
    """

    def __init__(self) -> None:
        """Initialize the LightGBM model."""

        super().__init__("LightGBM Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the LightGBM model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            # Core Parameters
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            # Learning Parameters
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            # Sampling Parameters
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # Regularization Parameters
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0),
            # Performance Parameters
            "n_jobs": trial.suggest_categorical("n_jobs", [-1]),  # use all CPU cores
            "random_state": trial.suggest_categorical("random_state", [28]),
            "verbose": trial.suggest_categorical("verbose", [-1]),  # suppress messages
            # Class Weight Parameters
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        }

        # DART-specific parameters
        if params["boosting_type"] == "dart":
            params.update(
                {
                    "drop_rate": trial.suggest_float("drop_rate", 0.1, 0.5),
                    "skip_drop": trial.suggest_float("skip_drop", 0.1, 0.5),
                    "max_drop": trial.suggest_int("max_drop", 10, 50),
                }
            )

        # For multi-class classification
        if self.is_multiclass:
            params.update(
                {
                    "objective": trial.suggest_categorical("objective", ["multiclass"]),
                    "num_class": trial.suggest_categorical("num_class", [self.num_classes]),
                    "metric": trial.suggest_categorical("metric", ["multi_logloss"]),
                }
            )
        else:  # For binary classification
            params.update(
                {
                    "objective": trial.suggest_categorical("objective", ["binary"]),
                    "metric": trial.suggest_categorical("metric", ["binary_logloss"]),
                }
            )

        return params

    def create_model(self, params: Dict[str, Any]) -> lgb.LGBMClassifier:
        """Create a LightGBM model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            lgb.LGBMClassifier: A LightGBM model.
        """

        try:
            self.logger.info("Creating LightGBM model", params=params)
            self.model = lgb.LGBMClassifier(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create LightGBM model", error=e)
            raise RuntimeError("Model creation failed") from e


class MobileNetV3SmallModel(BaseMlModel):
    """MobileNetV3Small Model.
    https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small
    """

    def __init__(self) -> None:
        """Initialize the MobileNetV3Small model."""

        super().__init__("MobileNetV3 Small Classifier")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the MobileNetV3Small model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            # Model parameters
            "input_shape": trial.suggest_categorical("input_shape", cast(Any, [(224, 224, 3)])),
            "weights": trial.suggest_categorical("weights", ["imagenet"]),
            "minimalistic": trial.suggest_categorical("minimalistic", [False, True]),
            "include_top": trial.suggest_categorical("include_top", [False]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            # Training parameters
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 10, 50),
            "loss": trial.suggest_categorical("loss", ["sparse_categorical_crossentropy"]),
            "metrics": trial.suggest_categorical("metrics", cast(Any, [["accuracy"]])),
            "validation_split": trial.suggest_categorical("validation_split", [0.2]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """Create a MobileNetV3Small model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            Model: A MobileNetV3Small model.
        """

        try:
            self.logger.info("Creating MobileNetV3Small model", params=params)

            base_model = MobileNetV3Small(
                input_shape=params["input_shape"],
                weights=params["weights"],
                minimalistic=params["minimalistic"],
                include_top=params["include_top"],
                dropout_rate=params["dropout_rate"],
            )
            base_model.trainable = False  # Freeze the base model

            # Add custom layers on top of the base model
            model = tf.keras.Sequential(
                [
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(params["dropout_rate"]),
                    tf.keras.layers.Dense(self.num_classes, activation="softmax"),
                ]
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
                loss=params["loss"],
                metrics=params["metrics"],
            )

            # Override the models fit method
            original_fit = model.fit
            original_predict = model.predict

            def custom_fit(x: Any, y: Any, **kwargs: Any) -> Any:
                return original_fit(
                    x,
                    y,
                    batch_size=params["batch_size"],
                    epochs=params["epochs"],
                    validation_split=params["validation_split"],
                    **kwargs,
                )

            def custom_predict(x: Any, **kwargs: Any) -> Any:
                # Only return the class with the highest probability
                return original_predict(x, **kwargs).argmax(axis=1)

            model.fit = custom_fit
            model.predict = custom_predict
            model.predict_proba = original_predict

            self.model = model
            return self.model
        except Exception as e:
            self.logger.error("Failed to create MobileNetV3Small model", error=e)
            raise RuntimeError("Model creation failed") from e


def get_all_classification_models() -> Dict[str, BaseMlModel]:
    """Get all classification models.

    Returns:
        Dict[str, BaseMlModel]: A dictionary containing all classification models.
    """

    logger = get_logger("All Classification Models")
    logger.info("Creating all classification models")

    try:
        models = {
            "Logistic Regression Classifier": LogisticRegressionModel(),
            "Support Vector Classifier": SVCModel(),
            "K-Nearest Neighbors Classifier": KNNModel(),
            "Naive Bayes Classifier": NaiveBayesModel(),
            "Decision Tree Classifier": DecisionTreeModel(),
            "Random Forest Classifier": RandomForestModel(),
            "XGBoost Classifier": XGBoostModel(),
            "LightGBM Classifier": LightGBMModel(),
        }

        logger.info(
            "Successfully created all models",
            model_count=len(models),
            model_names=list(models.keys()),
        )
        return models
    except Exception as e:
        logger.error("Failed to create all models", error=e)
        raise RuntimeError("Model initialization failed") from e


class EnsembleVotingClassifierModel(BaseMlEnsembleModel):
    """Ensemble voting classifier model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
    """

    def __init__(self, models: Sequence[Tuple[BaseMlModel, str]]) -> None:
        """Initialize the ensemble voting classifier model.

        Args:
            models (Sequence[Tuple[BaseMlModel, str]]):
                List of ml models for ensemble and MLflow run ids for best parameters.
        """

        super().__init__(model_name="Ensemble Voting Classifier", models=list(models))

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the ensemble voting classifier model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "estimators": trial.suggest_categorical(
                "estimators",
                cast(Any, [[(model.model_name, model.model) for model, _ in self.models]]),
            ),
            "voting": trial.suggest_categorical("voting", ["hard", "soft"]),
            "weights": [
                trial.suggest_float(f"weight_{i}", 0.0, 1.0) for i in range(self.num_models)
            ],
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> VotingClassifier:
        """Create a ensemble voting classifier model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            VotingClassifier: A ensemble voting classifier model.
        """

        try:
            self.logger.info("Creating ensemble voting classifier model", params=params)
            params = self._del_weight_keys(params)
            self.model = VotingClassifier(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create ensemble voting classifier model", error=e)
            raise RuntimeError("Model creation failed") from e


class EnsembleStackingClassifierModel(BaseMlEnsembleModel):
    """Ensemble stacking classifier model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
    """

    def __init__(self, models: Sequence[Tuple[BaseMlModel, str]], meta_model: BaseMlModel) -> None:
        """Initialize the ensemble stacking classifier model.

        Args:
            models (Sequence[Tuple[BaseMlModel, str]]):
                List of ml models for ensemble and MLflow run ids for best parameters.
            meta_model (BaseMlModel): A meta model for stacking.
        """

        super().__init__(model_name="Ensemble Stacking Classifier", models=list(models))
        self.meta_model = meta_model

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the ensemble stacking classifier model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "estimators": trial.suggest_categorical(
                "estimators",
                cast(Any, [[(model.model_name, model.model) for model, _ in self.models]]),
            ),
            "final_estimator": self.meta_model.create_model(self.meta_model.get_param_space(trial)),
            "stack_method": trial.suggest_categorical("stack_method", ["auto"]),
            "passthrough": trial.suggest_categorical("passthrough", [False, True]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> StackingClassifier:
        """Create a ensemble stacking classifier model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            StackingClassifier: A ensemble stacking classifier model.
        """

        try:
            self.logger.info("Creating ensemble stacking classifier model", params=params)
            stacking_params, meta_params = self._extract_meta_params(params)

            # Create meta model
            meta_model = self.meta_model.create_model(meta_params)

            # Create stacking model
            stacking_params["final_estimator"] = meta_model
            self.model = StackingClassifier(**stacking_params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create ensemble stacking classifier model", error=e)
            raise RuntimeError("Model creation failed") from e

    def set_num_classes(self, num_classes: int) -> None:
        """Set the number of classes for classification.

        Args:
            num_classes (int): Number of classes (2 for binary, > 2 for multi-class)
        """

        # Set for ensemble and base models
        super().set_num_classes(num_classes)

        # Set for meta model
        self.meta_model.set_num_classes(num_classes)


if __name__ == "__main__":
    pass
