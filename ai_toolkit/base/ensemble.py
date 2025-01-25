from abc import abstractmethod
from typing import List
import numpy as np

from ai_toolkit.base.model import BaseMlModel


class BaseMlEnsembleModel(BaseMlModel):
    """Abstract base class for all ml ensemble models."""

    def __init__(self, model_name: str, models: List[BaseMlModel]) -> None:
        """Initialize the base ml ensemble model.

        Args:
            model_name (str): Name of the ensemble model.
            models (List[BaseMlModel]): List of ml models for ensemble.
        """

        super().__init__(model_name)
        self.models = models
        self.weights = None

    @abstractmethod
    def combine_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Combine predictions from multiple models.

        Args:
            predictions (List[np.ndarray]): List of predictions from each ml model.

        Returns:
            np.ndarray: Combined predictions.
        """
        pass


if __name__ == "__main__":
    pass
