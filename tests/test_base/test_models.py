from typing import Any, Dict
from unittest.mock import Mock, patch

import optuna
import pytest

from ai_toolkit.base.models import BaseMlEnsembleModel, BaseMlModel


def test_base_model_initialization():
    """Test that BaseMlModel initialization raises an error."""

    with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
        BaseMlModel("Base Model Name")


def test_dummy_model_initialization(dummy_model):
    """Test dummy model initialization.

    Args:
        dummy_model (DummyModel): Dummy model instance.
    """

    assert dummy_model.model_name == "Dummy Model"
    assert dummy_model.model is None
    assert dummy_model.best_params is None


def test_get_model_info(dummy_model):
    """Test model info retrieval.

    Args:
        dummy_model (DummyModel): Dummy model instance.
    """

    dummy_model.best_params = {"param1": 0.2, "param2": 8}

    info = dummy_model.get_model_info()
    assert info["model_name"] == "Dummy Model"
    assert info["model_type"] == "DummyModel"
    assert info["best_params"] == {"param1": 0.2, "param2": 8}
    assert info["num_classes"] is None
    assert info["is_multiclass"] is None


def test_model_creation(dummy_model):
    """Test model creation with parameters.

    Args:
        dummy_model (DummyModel): Dummy model instance.
    """

    test_params = {"param1": 0.2, "param2": 8}

    created_model = dummy_model.create_model(test_params)
    assert hasattr(created_model, "param1")
    assert hasattr(created_model, "param2")
    assert created_model.param1 == 0.2
    assert created_model.param2 == 8


@pytest.mark.parametrize(
    "params",
    [
        {"param1": 0.0, "param2": 1},  # Minimum values
        {"param1": 1.0, "param2": 10},  # Maximum values
        {"param1": 0.5, "param2": 5},  # Middle values
    ],
)
def test_model_creation_parameter_ranges(dummy_model, params):
    """Test model creation with different parameter ranges.

    Args:
        dummy_model: Dummy model fixture.
        params: Test parameters.
    """

    model = dummy_model.create_model(params)
    for param_name, value in params.items():
        assert getattr(model, param_name) == value


def test_model_param_space(dummy_model, optuna_trial):
    """Test parameter space generation.

    Args:
        dummy_model: Dummy model fixture.
        optuna_trial: Optuna trial fixture.
    """

    param_space = dummy_model.get_param_space(optuna_trial)
    assert isinstance(param_space, dict)
    assert "param1" in param_space
    assert "param2" in param_space


@pytest.mark.parametrize(
    "invalid_params",
    [
        {},  # Empty params
        {"param1": 0.5},  # Missing param
        {"param1": 0.5, "param2": 5, "unknown": 1},  # Extra param
    ],
)
def test_model_creation_with_invalid_params(dummy_model, invalid_params):
    """Test model creation with invalid parameters.

    Args:
        dummy_model: Dummy model fixture.
        invalid_params: Invalid parameter configurations.
    """

    # For invalid but non-empty params, create_model should still work
    # but might have unexpected attributes
    model = dummy_model.create_model(invalid_params)
    for param_name, value in invalid_params.items():
        assert getattr(model, param_name) == value


def test_set_num_classes(dummy_model):
    """Test setting number of classes.

    Args:
        dummy_model: Dummy model fixture.
    """

    dummy_model.set_num_classes(2)
    assert dummy_model.num_classes == 2
    assert dummy_model.is_multiclass is False

    dummy_model.set_num_classes(3)
    assert dummy_model.num_classes == 3
    assert dummy_model.is_multiclass is True


class DummyEnsembleModel(BaseMlEnsembleModel):
    """Dummy ensemble model for testing."""

    def get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Get the hyperparameter space for the ensemble model.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """

        return {
            "weight_0": trial.suggest_float("weight_0", 0, 1),
            "weight_1": trial.suggest_float("weight_1", 0, 1),
        }

    def create_model(self, params: Dict[str, Any]) -> Any:
        """Create ensemble model with given parameters.

        Args:
            params (Dict[str, Any]): Model parameters.

        Returns:
            Any: Model instance.
        """

        self.model = type("DummyEnsembleModel", (), params)()
        return self.model


@pytest.fixture
def mock_mlflow_runs():
    """Create mock MLflow runs for testing.

    Returns:
        dict: Dictionary containing mock run data.
    """

    mock_run1 = Mock()
    mock_run1.data.params = {
        "param1": "0.5",
        "param2": "5",
        "model_name": "Dummy1",
        "n_splits": "5",
    }

    mock_run2 = Mock()
    mock_run2.data.params = {
        "param1": "0.3",
        "param2": "3",
        "model_name": "Dummy2",
        "n_splits": "5",
    }

    return {"mock_run_1": mock_run1, "mock_run_2": mock_run2}


@pytest.fixture
def ensemble_model(dummy_model, mock_mlflow_runs):
    """Create an ensemble model for testing.

    Args:
        dummy_model: Dummy model fixture.
        mock_mlflow_runs: Mock MLflow runs fixture.

    Returns:
        DummyEnsembleModel: A configured ensemble model.
    """

    with patch("mlflow.get_run") as mock_get_run:
        # Configure mock to return different runs based on run_id
        def get_mock_run(run_id):
            return mock_mlflow_runs[run_id]

        mock_get_run.side_effect = get_mock_run

        # Create ensemble model
        models = [(dummy_model, "mock_run_1"), (dummy_model, "mock_run_2")]
        return DummyEnsembleModel("Dummy Ensemble", models)


def test_ensemble_model_initialization(ensemble_model):
    """Test ensemble model initialization.

    Args:
        ensemble_model (DummyEnsembleModel): Dummy ensemble model instance.
    """

    assert ensemble_model.model_name == "Dummy Ensemble"
    assert len(ensemble_model.models) == 2
    assert ensemble_model.num_models == 2


@pytest.mark.parametrize(
    "weights,expected",
    [
        ({"weight_0": 0.3, "weight_1": 0.7}, [0.3, 0.7]),
        ({"weight_0": 0.5, "weight_1": 0.5}, [0.5, 0.5]),
        ({"weight_0": 1.0, "weight_1": 0.0}, [1.0, 0.0]),
    ],
)
def test_ensemble_weight_handling(ensemble_model, weights, expected):
    """Test ensemble model weight handling.

    Args:
        ensemble_model: Ensemble model fixture.
        weights: Test weight configurations.
        expected: Expected weight values.
    """

    processed_params = ensemble_model._del_weight_keys(weights)
    assert "weights" in processed_params
    assert processed_params["weights"] == expected


def test_ensemble_mlflow_integration(dummy_model, mock_mlflow_runs):
    """Test ensemble model MLflow integration.

    Args:
        dummy_model: Dummy model fixture.
        mock_mlflow_runs: Mock MLflow runs fixture.
    """

    with patch("mlflow.get_run") as mock_get_run:
        # Configure mock
        def get_mock_run(run_id):
            return mock_mlflow_runs[run_id]

        mock_get_run.side_effect = get_mock_run

        # Create ensemble and verify MLflow integration
        models = [(dummy_model, "mock_run_1"), (dummy_model, "mock_run_2")]
        ensemble = DummyEnsembleModel("Dummy Ensemble", models)

        # Verify that MLflow was called correctly
        assert mock_get_run.call_count == 2
        mock_get_run.assert_any_call("mock_run_1")
        mock_get_run.assert_any_call("mock_run_2")

        # Verify that base models were created
        for model, _ in ensemble.models:
            assert model.model is not None


def test_ensemble_with_invalid_run_id(dummy_model):
    """Test ensemble model with invalid MLflow run ID.

    Args:
        dummy_model: Dummy model fixture.
    """

    with patch("mlflow.get_run") as mock_get_run:
        mock_get_run.side_effect = Exception("Run ID not found")

        # Create ensemble with invalid run ID
        models = [(dummy_model, "invalid_run_id")]
        with pytest.raises(Exception):
            DummyEnsembleModel("Dummy Ensemble", models)


@pytest.mark.parametrize(
    "mock_params,expected_params",
    [
        (
            {"param1": "0.5", "param2": "5", "extra": "value"},
            {"param1": 0.5, "param2": 5},
        ),
        (
            {"param1": "0.1", "param2": "1", "model_name": "Test"},
            {"param1": 0.1, "param2": 1},
        ),
    ],
)
def test_ensemble_parameter_processing(dummy_model, mock_params, expected_params):
    """Test ensemble model parameter processing.

    Args:
        dummy_model: Dummy model fixture.
        mock_params: Mock MLflow parameters.
        expected_params: Expected processed parameters.
    """

    with patch("mlflow.get_run") as mock_get_run:
        # Configure mock run
        mock_run = Mock()
        mock_run.data.params = mock_params
        mock_get_run.return_value = mock_run

        # Create ensemble and verify parameter processing
        models = [(dummy_model, "mock_run")]
        ensemble = DummyEnsembleModel("Dummy Ensemble", models)

        # Verify that base models were created with correct parameters
        for model, _ in ensemble.models:
            assert model.model is not None
            for param_name, expected_value in expected_params.items():
                assert hasattr(model.model, param_name)
                assert getattr(model.model, param_name) == expected_value


def test_ensemble_model_creation(ensemble_model):
    """Test ensemble model creation with parameters.

    Args:
        ensemble_model: Ensemble model fixture.
    """

    # Test parameters for ensemble
    test_params = {"weights": [0.6, 0.4], "voting": "soft"}

    # Create model and verify
    created_model = ensemble_model.create_model(test_params)
    assert hasattr(created_model, "weights")
    assert hasattr(created_model, "voting")
    assert created_model.weights == [0.6, 0.4]
    assert created_model.voting == "soft"


def test_set_num_classes_ensemble(ensemble_model):
    """Test setting number of classes for ensemble model.

    Args:
        ensemble_model: Ensemble model fixture.
    """

    ensemble_model.set_num_classes(2)
    assert ensemble_model.num_classes == 2
    assert ensemble_model.is_multiclass is False

    for model, _ in ensemble_model.models:
        assert model.num_classes == 2
        assert model.is_multiclass is False

    ensemble_model.set_num_classes(3)
    assert ensemble_model.num_classes == 3
    assert ensemble_model.is_multiclass is True

    for model, _ in ensemble_model.models:
        assert model.num_classes == 3
        assert model.is_multiclass is True


def test_safe_convert():
    """Test safe conversion of parameter values."""

    from ai_toolkit.base.models import safe_convert

    # Test numeric conversions
    assert safe_convert("42") == 42
    assert safe_convert("3.14") == 3.14
    assert safe_convert("-1") == -1

    # Test boolean conversions
    assert safe_convert("True") is True
    assert safe_convert("False") is False

    # Test string (no conversion)
    assert safe_convert("hello") == "hello"

    # Test None
    assert safe_convert("None") is None

    # Test list/tuple
    assert safe_convert("[1, 2, 3]") == [1, 2, 3]
    assert safe_convert("(1, 2, 3)") == (1, 2, 3)

    # Test dict
    assert safe_convert("{'a': 1}") == {"a": 1}
