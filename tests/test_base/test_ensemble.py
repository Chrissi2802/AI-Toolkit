import pytest
import optuna
from typing import Dict, Any, List
import numpy as np

from tests.test_base.test_model import DummyModel
from ai_toolkit.base.ensemble import BaseMlEnsembleModel


class DummyEnsembleModel(BaseMlEnsembleModel):
    """Dummy ensemble model implementation for testing."""

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the model.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """

        return {
            "weight1": trial.suggest_float("weight1", 0, 1),
            "weight2": trial.suggest_int("weight2", 1, 10),
        }

    def create_model(self, params: Dict[str, Any]) -> Any:
        """Create the model using the given hyperparameters.

        Args:
            params (Dict[str, Any]): Dictionary of hyperparameters.

        Returns:
            Any: Model object.
        """

        self.model = type("DummyEnsembleModel", (), params)()

        return self.model

    def combine_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Combine predictions from multiple models.

        Args:
            predictions (List[np.ndarray]): List of predictions from multiple models.

        Returns:
            np.ndarray: Combined predictions.
        """

        return np.mean(predictions, axis=0)


@pytest.fixture
def base_models():
    """Fixture providing list of base models."""

    return [DummyModel("model1"), DummyModel("model2")]


def test_ensemble_model_initialization():
    """Test that BaseMlEnsembleModel initialization raises an error."""

    with pytest.raises(TypeError):
        BaseMlEnsembleModel("Base Ensemble Model Name", [])


def test_dummy_ensemble_model_initialization(base_models: List[DummyModel]):
    """Test dummy ensemble model initialization.

    Args:
        base_models (List[BaseMlModel]): List of base models.
    """

    model = DummyEnsembleModel("Dummy Ensemble Model Name", base_models)

    assert model.model_name == "Dummy Ensemble Model Name"
    # assert model.models = base_models
    assert model.weights is None


def test_get_ensemble_model_info(base_models: List[DummyModel]):
    """Test ensemble model info retrieval."""

    model = DummyEnsembleModel("Dummy Ensemble Model Name", base_models)
    model.best_params = {"weight1": 0.2, "weight2": 8}

    info = model.get_model_info()
    assert info["model_name"] == "Dummy Ensemble Model Name"
    assert info["model_type"] == "DummyEnsembleModel"
    assert info["best_params"] == {"weight1": 0.2, "weight2": 8}


def test_ensemble_combine_predictions(base_models: List[DummyModel]):
    """Test prediction combination in ensemble."""

    model = DummyEnsembleModel("Dummy Ensemble Model Name", base_models)
    predictions = [np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])]

    combined = model.combine_predictions(predictions)
    expected = np.array([0.15, 0.25, 0.35])
    np.testing.assert_array_almost_equal(combined, expected)
