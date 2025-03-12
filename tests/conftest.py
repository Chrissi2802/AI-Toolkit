import gc
from typing import Any, Dict, Tuple
from unittest.mock import Mock, patch

import matplotlib
import mlflow
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from ai_toolkit.base.models import BaseMlModel


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


@pytest.fixture
def dummy_model() -> DummyModel:
    """Create a dummy model for testing.

    Returns:
        DummyModel: A dummy model instance.
    """

    return DummyModel("Dummy Model")


@pytest.fixture
def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic classification data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of features and target arrays.
    """

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=28,
    )

    return X, y


@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic regression data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of features and target arrays.
    """

    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=28,
    )

    # All target values must be positive
    y = y - np.min(y) + 1.0

    return X, y


@pytest.fixture
def classification_data_pd(
    classification_data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create synthetic classification data as pandas DataFrame and Series.

    Args:
        classification_data (Tuple[np.ndarray, np.ndarray]): A tuple of features and target arrays.
    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple of features and target pandas objects.
    """

    X, y = classification_data
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def regression_data_pd(
    regression_data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create synthetic regression data as pandas DataFrame and Series.

    Args:
        regression_data (Tuple[np.ndarray, np.ndarray]): A tuple of features and target arrays.
    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple of features and target pandas objects
    """

    X, y = regression_data
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def classification_predictions(
    classification_data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic classification predictions.

    Args:
        classification_data (Tuple[np.ndarray, np.ndarray]):
            A tuple containing true labels and predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Tuple containing true labels, predictions, and prediction probabilities.
    """

    _, y_true = classification_data

    # Create predictions with known characteristics
    y_pred = y_true.copy()
    y_pred[::5] = 1 - y_pred[::5]  # Flip every 5th prediction

    # Create probabilities
    y_pred_proba = np.random.random(len(y_true))
    y_pred_proba = np.column_stack((1 - y_pred_proba, y_pred_proba))

    return y_true, y_pred, y_pred_proba


@pytest.fixture
def regression_predictions(
    regression_data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic regression predictions.

    Args:
        regression_data (Tuple[np.ndarray, np.ndarray]):
            A tuple containing true labels and predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing true labels and predictions.
    """

    _, y_true = regression_data

    # Create predictions with known error
    noise = np.random.normal(0, 0.1, size=len(y_true))
    y_pred = y_true + noise

    return y_true, y_pred


@pytest.fixture
def optuna_trial() -> optuna.trial.Trial:
    """Create an Optuna trial object.

    Returns:
        optuna.trial.Trial: An Optuna trial object.
    """

    study = optuna.create_study()

    return optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))


@pytest.fixture
def feature_importance_data() -> Tuple[np.ndarray, list]:
    """Create feature importance data.

    Returns:
        Tuple[np.ndarray, list]: A tuple of feature importance scores and feature names
    """

    n_features = 5
    importance_scores = np.array([0.3, 0.2, 0.15, 0.25, 0.1])
    feature_names = [f"Feature_{i}" for i in range(n_features)]

    return importance_scores, feature_names


@pytest.fixture
def mock_mlflow():
    """Mock MLflow functionality."""

    with patch("mlflow.start_run") as mock_run:
        with patch("mlflow.log_params") as mock_log_params:
            with patch("mlflow.log_metric") as mock_log_metric:
                with patch("mlflow.log_artifacts") as mock_log_artifacts:
                    with patch("mlflow.log_table") as mock_log_table:
                        with patch("mlflow.end_run") as mock_end_run:
                            yield {
                                "run": mock_run,
                                "log_params": mock_log_params,
                                "log_metric": mock_log_metric,
                                "log_artifacts": mock_log_artifacts,
                                "log_table": mock_log_table,
                                "end_run": mock_end_run,
                            }


@pytest.fixture(autouse=True)
def mock_mlflow_experiment():
    """Mock MLflow set_experiment function."""

    with patch("mlflow.set_experiment") as mock_set_experiment:
        with patch(
            "mlflow.get_experiment_by_name", return_value=Mock(experiment_id="0")
        ):
            with patch("mlflow.set_tracking_uri"):  # Prevents connection attempts
                yield mock_set_experiment


@pytest.fixture(autouse=True)
def mlflow_cleanup():
    """Cleanup MLflow runs before and after each test."""

    mlflow.end_run()
    yield
    mlflow.end_run()


@pytest.fixture(autouse=True, scope="session")
def mpl_backend():
    """Configure matplotlib to use Agg backend for all tests."""

    matplotlib.use("Agg")
    assert matplotlib.get_backend() == "Agg"


@pytest.fixture(autouse=True, scope="package")  # package, module
def cleanup():
    """Cleanup memory after tests."""

    yield  # Run the tests
    gc.collect()
