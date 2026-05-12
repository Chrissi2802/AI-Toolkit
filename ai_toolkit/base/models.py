from abc import ABC, abstractmethod
from ast import literal_eval
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import optuna

from ai_toolkit.utils.logging import get_logger


class BaseMlModel(ABC):
    """Abstract base class for all ml models."""

    def __init__(self, model_name: str) -> None:
        """Initialize the base ml model.

        Args:
            model_name (str): The name of the model.
        """

        self.model_name = model_name
        self.model: Any = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.num_classes: Optional[int] = None
        self.is_multiclass: Optional[bool] = None

        # Initialize logger
        self.logger = get_logger(f"{self.__class__.__name__}_{model_name}")
        self.logger.info("Initializing model", model_name=model_name)

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
            "num_classes": self.num_classes,
            "is_multiclass": self.is_multiclass,
        }

    def load_best_params(self, params_source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load best parameters from either MLflow run_id or manual dictionary.

        Args:
            params_source (Union[str, Dict[str, Any]]):
                Either MLflow run_id (str) or manual parameters (dict)

        Returns:
            Dict[str, Any]: Dictionary containing the best parameters.
        """

        if isinstance(params_source, str):
            self.logger.info("Loading parameters from MLflow", rund_id=params_source)
            return self.get_mlflow_best_params(params_source)
        elif isinstance(params_source, dict):
            self.logger.info("Using manual parameters", params=params_source)
            return self.set_manual_params(params_source)
        else:
            raise ValueError(
                f"params_source must be either str (MLflow run_id) or dict (manual params), "
                f"got {type(params_source)}"
            )

    def get_mlflow_best_params(self, run_id: str) -> Dict[str, Any]:
        """Load best parameters from MLflow runs based on models parameter space.

        Args:
            run_id (str): The run id of the MLflow run.
        Returns:
            Dict[str, Any]: Dictionary containing the best parameters.
        """

        try:
            self.logger.debug("Loading parameters from MLflow", run_id=run_id)

            # Get the run from MLflow
            mlflow.set_tracking_uri("http://localhost:5000")
            run = mlflow.get_run(run_id)

            # Get the parameters from the mlflow run
            mlflow_params = run.data.params

            # Create a new study and trial to get the parameter space
            study = optuna.create_study(study_name="Get best parameters from MLflow")
            trail = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))

            # Get the parameter space for the model
            model_params = self.get_param_space(trail).keys()

            # Remove keys that are not in the parameter space
            # Convert the values to the correct type
            best_params = {
                key: safe_convert(value)
                for key, value in mlflow_params.items()
                if key in model_params
            }

            self.logger.info("Loaded best parameters", run_id=run_id, params=best_params)

            return best_params

        except Exception as e:
            self.logger.error("Failed to load MLflow parameters", error=e, run_id=run_id)
            raise RuntimeError("Failed to load parameters from MLflow") from e

    def set_manual_params(self, manual_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set manual parameters based on models parameter space.

        Args:
            manual_params (Dict[str, Any]): Manual parameters dictionary

        Returns:
            Dict[str, Any]: Validated parameters dictionary
        """

        try:
            self.logger.debug("Processing manual parameters", params=manual_params)

            # Create a new study and trial to get the parameter space
            study = optuna.create_study(study_name="Validate manual parameters")
            trail = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))

            # Get the parameter space for the model
            model_params = self.get_param_space(trail).keys()

            # Validate that all provided parameters are in the parameter space
            invalid_params = set(manual_params.keys()) - set(model_params)
            if invalid_params:
                self.logger.warning(
                    "Some parameters are not in model parameter space",
                    invalid_params=list(invalid_params),
                    valid_params=list(model_params),
                )

            # Filter to only valid parameters
            validated_params = {
                key: value for key, value in manual_params.items() if key in model_params
            }

            self.logger.info("Validated manual parameters", params=validated_params)
            return validated_params

        except Exception as e:
            self.logger.error("Failed to process manual parameters", error=e, params=manual_params)
            raise RuntimeError("Failed to validate manual parameters") from e

    def set_num_classes(self, num_classes: int) -> None:
        """Set the number of classes for classification.

        Args:
            num_classes (int): Number of classes (2 for binary, > 2 for multi-class)
        """

        # Set for the model
        self.num_classes = num_classes
        self.is_multiclass = self.num_classes is not None and self.num_classes > 2
        self.logger.info("Set number of classes", num_classes=num_classes)


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

    def __init__(
        self, model_name: str, models: List[Tuple[BaseMlModel, Union[str, Dict[str, Any]]]]
    ) -> None:
        """Initialize the base ml ensemble model.

        Args:
            model_name (str): Name of the ensemble model.
            models (List[Tuple[BaseMlModel, Union[str, Dict[str, Any]]]]):
                List of ml models for ensemble and MLflow run ids for best parameters.
        """

        super().__init__(model_name=model_name)
        self.models = models
        self.num_models = len(self.models)

        self._load_best_params()

    def _load_best_params(self) -> None:
        """Load best parameters from parameter sources (MLflow runs or manual)
        based on models parameter space."""

        for model, params_source in self.models:
            try:
                # Automatic detection and loading
                params = model.load_best_params(params_source)
                model = model.create_model(params)
                model.best_params = params

            except Exception as e:
                self.logger.error(
                    "Failed to load best parameters for the model",
                    model_name=model.model_name,
                    error=e,
                )
                raise

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

    def _extract_meta_params(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    def set_num_classes(self, num_classes: int) -> None:
        """Set the number of classes for classification.

        Args:
            num_classes (int): Number of classes (2 for binary, > 2 for multi-class)
        """

        # Set for ensemble model
        super().set_num_classes(num_classes)

        # Set for all base models
        for model, _ in self.models:
            model.set_num_classes(num_classes)


if __name__ == "__main__":
    pass
