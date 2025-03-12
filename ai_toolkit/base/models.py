from abc import ABC, abstractmethod
from ast import literal_eval
from typing import Any, Dict, List, Tuple

import mlflow
import optuna


class BaseMlModel(ABC):
    """Abstract base class for all ml models."""

    def __init__(self, model_name: str) -> None:
        """Initialize the base ml model.

        Args:
            model_name (str): The name of the model.
        """

        self.model_name = model_name
        self.model = None
        self.best_params = None

    @abstractmethod
    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the parameter space for optimization.

        Args:
            trial (optuna.Trial): An optuna trial object.

        Returns:
            Dict[str, Any]: A dictionary containing the hyperparameters.
        """
        pass

    @abstractmethod
    def create_model(self, params: Dict[str, Any]) -> Any:
        """Create a new model instance with given parameters.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            Any: A new model instance.
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """

        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "best_params": self.best_params,
        }

    def get_mlflow_best_params(self, run_id: str) -> Dict[str, Any]:
        """Load best parameters from MLflow runs based on models parameter space.

        Args:
            run_id (str): The run id of the MLflow run.
        Returns:
            Dict[str, Any]: Dictionary containing the best parameters.
        """

        # Get the run from MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        run = mlflow.get_run(run_id)

        # Get the parameters from the run
        params = run.data.params

        # Remove keys that are not in the parameter space
        keys = [
            "model_name",
            "n_splits",
            "n_trials",
            "optimize_metric",
            "n_samples",
            "n_features",
            "use_smote",
            "smote_ratio",
            "class_distribution",
            "class_ratio",
            "target_mean",
            "target_std",
            "target_median",
            "target_min",
            "target_max",
        ]

        for key in keys:
            params.pop(key, None)

        # Convert the values to the correct type
        best_params = {key: safe_convert(value) for key, value in params.items()}

        return best_params


def safe_convert(value: str) -> Any:
    """Converts a string to a Python object if possible. Otherwise, returns the string.

    Args:
        value (str): The string to convert.

    Returns:
        Any: The Python object.
    """

    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


class BaseMlEnsembleModel(BaseMlModel):
    """Abstract base class for all ml ensemble models."""

    def __init__(self, model_name: str, models: List[Tuple[BaseMlModel, str]]) -> None:
        """Initialize the base ml ensemble model.

        Args:
            model_name (str): Name of the ensemble model.
            models (List[Tuple[BaseMlModel, str]]):
                List of ml models for ensemble and MLflow run ids for best parameters.
        """

        super().__init__(model_name=model_name)
        self.models = models
        self.num_models = len(self.models)

        self._load_mlflow_best_params()

    def _load_mlflow_best_params(self) -> None:
        """Load best parameters from MLflow runs based on models parameter space."""

        for model, run_id in self.models:
            params = model.get_mlflow_best_params(run_id)
            model = model.create_model(params)

    def _del_weight_keys(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete weight keys from the dictionary and add weights key.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            Dict[str, Any]: A dictionary containing the parameters with weights key.
        """

        # Check if weights key are present
        if "weights" in params.keys():
            return params

        # Get all weight keys
        weight_keys = [key for key in params.keys() if key.startswith("weight_")]

        # Delete weight keys from the dictionary
        weight_values = []
        for key in weight_keys:
            weight_values.append(params[key])
            del params[key]

        # Add weights key to the dictionary
        if len(weight_values) == self.num_models:
            params["weights"] = weight_values

        return params

    def _extract_meta_params(
        self, params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract stacking and meta parameters from the whole parameter dictionary.

        Args:
            params (Dict[str, Any]): Parameters for the model.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing stacking and meta parameters.
        """

        stacking_keys = {"estimators", "passthrough"}
        sm = "stack_method"
        fe = "final_estimator"

        if sm in params.keys():
            stacking_keys.add(sm)

        if fe in params.keys():
            stacking_keys.add(fe)

        # Extract stacking and meta parameters
        stacking_params = {k: params[k] for k in stacking_keys}
        meta_params = {k: v for k, v in params.items() if k not in stacking_keys}

        return stacking_params, meta_params


if __name__ == "__main__":
    pass
