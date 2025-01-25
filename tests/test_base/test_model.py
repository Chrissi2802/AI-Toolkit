import pytest
import optuna
from typing import Dict, Any

from ai_toolkit.base.model import BaseMlModel


class DummyModel(BaseMlModel):
    """Dummy model implementation for testing."""

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the model.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """

        return {
            "param1": trial.suggest_float("param1", 0, 1),
            "param2": trial.suggest_int("param2", 1, 10),
        }

    def create_model(self, params: Dict[str, Any]) -> Any:
        """Create the model using the given hyperparameters.

        Args:
            params (Dict[str, Any]): Dictionary of hyperparameters.

        Returns:
            Any: Model object.
        """

        self.model = type("DummyModel", (), params)()

        return self.model


def test_base_model_initialization():
    """Test that BaseMlModel initialization raises an error."""

    with pytest.raises(TypeError):
        BaseMlModel("Base Model Name")


def test_dummy_model_initialization():
    """Test dummy model initialization."""

    model = DummyModel("Dummy Model Name")

    assert model.model_name == "Dummy Model Name"
    assert model.model is None
    assert model.best_params is None


def test_get_model_info():
    """Test model info retrieval."""

    model = DummyModel("Dummy Model Name")
    model.best_params = {"param1": 0.2, "param2": 8}

    info = model.get_model_info()
    assert info["model_name"] == "Dummy Model Name"
    assert info["model_type"] == "DummyModel"
    assert info["best_params"] == {"param1": 0.2, "param2": 8}


def test_model_creation():
    """Test model creation with parameters."""

    model = DummyModel("Dummy Model Name")
    test_params = {"param1": 0.2, "param2": 8}

    created_model = model.create_model(test_params)
    assert hasattr(created_model, "param1")
    assert hasattr(created_model, "param2")
    assert created_model.param1 == 0.2
    assert created_model.param2 == 8
