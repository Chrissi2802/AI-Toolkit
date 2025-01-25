from abc import ABC, abstractmethod
from typing import Dict, Any
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


if __name__ == "__main__":
    pass
