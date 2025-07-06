from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import optuna
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from ai_toolkit.base.models import BaseMlEnsembleModel, BaseMlModel
from ai_toolkit.utils.logging import get_logger


class RidgeRegressionModel(BaseMlModel):
    """Linear based: Ridge Regression Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    """

    def __init__(self):
        """Initialize the Ridge Regression model."""

        super().__init__("Ridge Regressor")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the Ridge Regression model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 100, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "solver": trial.suggest_categorical(
                "solver",
                ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            ),
            "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
            "random_state": trial.suggest_categorical("random_state", [28]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> Ridge:
        """Create a Ridge Regression model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            Ridge: A Ridge Regression model.
        """

        try:
            self.logger.info("Creating ridge regression model", params=params)
            self.model = Ridge(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create ridge regression model", error=e)
            raise RuntimeError("Model creation failed") from e


class BayesianRidgeRegressionModel(BaseMlModel):
    """Bayesian based: Bayesian Ridge Regression Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
    """

    def __init__(self):
        """Initialize the Bayesian Ridge Regression model."""

        super().__init__("Bayesian Ridge Regressor")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the Bayesian Ridge model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
            "alpha_1": trial.suggest_float("alpha_1", 1e-7, 1e-4, log=True),
            "alpha_2": trial.suggest_float("alpha_2", 1e-7, 1e-4, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-7, 1e-4, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-7, 1e-4, log=True),
            "compute_score": trial.suggest_categorical("compute_score", [True]),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> BayesianRidge:
        """Create a Bayesian Ridge model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            BayesianRidge: A Bayesian Ridge Regression model.
        """

        try:
            self.logger.info("Creating bayesian ridge regression model", params=params)
            self.model = BayesianRidge(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create bayesian ridge regression model", error=e)
            raise RuntimeError("Model creation failed") from e


class SVRModel(BaseMlModel):
    """Kernel based: Support Vector Regression Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """

    def __init__(self):
        """Initialize the Support Vector Regression model."""

        super().__init__("Support Vector Regressor")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the SVR model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])

        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),
            "cache_size": trial.suggest_categorical("cache_size", [2000]),
        }

        # Add kernel-specific parameters
        if kernel in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["coef0"] = trial.suggest_float("coef0", 0, 1)

        if kernel == "sigmoid":
            params["coef0"] = trial.suggest_float("coef0", 0, 1)

        return params

    def create_model(self, params: Dict[str, Any]) -> SVR:
        """Create a SVR model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            SVR: A Support Vector Regression model.
        """

        try:
            self.logger.info("Creating support vector regression model", params=params)
            self.model = SVR(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create support vector regression model", error=e)
            raise RuntimeError("Model creation failed") from e


class KNNRegressorModel(BaseMlModel):
    """Instance based: K-Nearest Neighbors Regression Model.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """

    def __init__(self):
        """Initialize the K-Nearest Neighbors Regression model."""

        super().__init__("K-Nearest Neighbors Regressor")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the KNN Regressor model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50),
            "p": trial.suggest_int("p", 1, 2),  # 1 for manhattan, 2 for euclidean
            "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> KNeighborsRegressor:
        """Create a KNN Regressor model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            KNeighborsRegressor: A K-Nearest Neighbors Regression model.
        """

        try:
            self.logger.info("Creating K-Nearest Neighbors regression model", params=params)
            self.model = KNeighborsRegressor(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create K-Nearest Neighbors regression model", error=e)
            raise RuntimeError("Model creation failed") from e


class XGBoostRegressorModel(BaseMlModel):
    """Gradient Boosting: Extreme Gradient Boosting Model.
    https://xgboost.readthedocs.io/en/stable/parameter.html
    """

    def __init__(self):
        """Initialize the XGBoost Regressor model."""

        super().__init__("XGBoost Regressor")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the XGBoost model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "objective": trial.suggest_categorical("objective", ["reg:squarederror"]),
            "tree_method": trial.suggest_categorical("tree_method", ["hist"]),
            "device": trial.suggest_categorical("device", ["cpu"]),
            "random_state": trial.suggest_categorical("random_state", [28]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> xgb.XGBRegressor:
        """Create an XGBoost model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            xgb.XGBRegressor: An XGBoost Regressor model.
        """

        try:
            self.logger.info("Creating XGBoost regression model", params=params)
            self.model = xgb.XGBRegressor(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create XGBoost regression model", error=e)
            raise RuntimeError("Model creation failed") from e


class LightGBMRegressorModel(BaseMlModel):
    """Gradient Boosting: Light Gradient-Boosting Machine (LightGBM) Model.
    https://lightgbm.readthedocs.io/en/latest/Parameters.html
    """

    def __init__(self):
        """Initialize the LightGBM Regressor model."""

        super().__init__("LightGBM Regressor")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the LightGBM model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "objective": trial.suggest_categorical("objective", ["regression"]),
            "metric": trial.suggest_categorical("metric", ["rmse"]),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "device_type": trial.suggest_categorical("device_type", ["cpu"]),
            "verbose": trial.suggest_categorical("verbose", [-1]),
            "seed": trial.suggest_categorical("seed", [28]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> lgb.LGBMRegressor:
        """Create a LightGBM model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            lgb.LGBMRegressor: A LightGBM Regressor model.
        """

        try:
            self.logger.info("Creating LightGBM regression model", params=params)
            self.model = lgb.LGBMRegressor(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create LightGBM regression model", error=e)
            raise RuntimeError("Model creation failed") from e


class CatBoostRegressorModel(BaseMlModel):
    """Gradient Boosting: CatBoost Regression Model.
    https://catboost.ai/docs/en/concepts/python-reference_catboostregressor
    """

    def __init__(self):
        """Initialize the CatBoost Regressor model."""

        super().__init__("CatBoost Regressor")

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the CatBoost model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.01, 10.0),
            "od_type": trial.suggest_categorical("od_type", ["Iter"]),
            "od_wait": trial.suggest_int("od_wait", 10, 50),
            "verbose": trial.suggest_categorical("verbose", [False]),
            "random_seed": trial.suggest_categorical("random_seed", [28]),
            "task_type": trial.suggest_categorical("task_type", ["CPU"]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> CatBoostRegressor:
        """Create a CatBoost model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            CatBoostRegressor: A CatBoost Regressor model.
        """

        try:
            self.logger.info("Creating CatBoost regression model", params=params)
            self.model = CatBoostRegressor(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create CatBoost regression model", error=e)
            raise RuntimeError("Model creation failed") from e


def get_all_regression_models() -> Dict[str, BaseMlModel]:
    """Get all regression models.

    Returns:
        Dict[str, BaseMlModel]: A dictionary containing all regression models.
    """

    logger = get_logger("All Regression Models")
    logger.info("Creating all classification models")

    try:
        models = {
            "Ridge Regressor": RidgeRegressionModel(),
            "Bayesian Ridge Regressor": BayesianRidgeRegressionModel(),
            "Support Vector Regressor": SVRModel(),
            "K-Nearest Neighbors Regressor": KNNRegressorModel(),
            "XGBoost Regressor": XGBoostRegressorModel(),
            "LightGBM Regressor": LightGBMRegressorModel(),
            "CatBoost Regressor": CatBoostRegressorModel(),
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


class EnsembleVotingRegressorModel(BaseMlEnsembleModel):
    """Ensemble voting regressor model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
    """

    def __init__(self, models: List[Tuple[BaseMlModel, str]]) -> None:
        """Initialize the ensemble voting regressor model.

        Args:
            models (List[Tuple[BaseMlModel, str]]):
                List of ml models for ensemble and MLflow run ids for best parameters.
        """

        super().__init__(model_name="Ensemble Voting Regressor", models=models)

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the ensemble voting regressor model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "estimators": trial.suggest_categorical(
                "estimators",
                [[(model.model_name, model.model) for model, _ in self.models]],
            ),
            "weights": [
                trial.suggest_float(f"weight_{i}", 0.0, 1.0) for i in range(self.num_models)
            ],
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> VotingRegressor:
        """Create a ensemble voting regressor model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            VotingRegressor: A ensemble voting regressor model.
        """

        try:
            self.logger.info("Creating ensemble voting regressor model", params=params)
            params = self._del_weight_keys(params)
            self.model = VotingRegressor(**params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create ensemble voting regressor model", error=e)
            raise RuntimeError("Model creation failed") from e


class EnsembleStackingRegressorModel(BaseMlEnsembleModel):
    """Ensemble stacking regressor model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
    """

    def __init__(self, models: List[Tuple[BaseMlModel, str]], meta_model: BaseMlModel) -> None:
        """Initialize the ensemble stacking regressor model.

        Args:
            models (List[Tuple[BaseMlModel, str]]):
                List of ml models for ensemble and MLflow run ids for best parameters.
            meta_model (BaseMlModel): A meta model for stacking.
        """

        super().__init__(model_name="Ensemble Stacking Regressor", models=models)
        self.meta_model = meta_model

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the ensemble stacking regressor model.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """

        params = {
            "estimators": trial.suggest_categorical(
                "estimators",
                [[(model.model_name, model.model) for model, _ in self.models]],
            ),
            "final_estimator": self.meta_model.create_model(self.meta_model.get_param_space(trial)),
            "passthrough": trial.suggest_categorical("passthrough", [False, True]),
        }

        return params

    def create_model(self, params: Dict[str, Any]) -> StackingRegressor:
        """Create a ensemble stacking regressor model with the given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            StackingRegressor: A ensemble stacking regressor model.
        """

        try:
            self.logger.info("Creating ensemble stacking regressor model", params=params)
            stacking_params, meta_params = self._extract_meta_params(params)

            # Create meta model
            meta_model = self.meta_model.create_model(meta_params)

            # Create stacking model
            stacking_params["final_estimator"] = meta_model
            self.model = StackingRegressor(**stacking_params)
            return self.model
        except Exception as e:
            self.logger.error("Failed to create ensemble stacking regressor model", error=e)
            raise RuntimeError("Model creation failed") from e


if __name__ == "__main__":
    pass
