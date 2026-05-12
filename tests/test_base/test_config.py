import json
import logging
import tempfile
from operator import gt, lt
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from ai_toolkit.base.config import (
    AIToolkitConfig,
    ConfigFactory,
    DataConfig,
    LoggingConfig,
    MetricConfig,
    TrainingConfig,
    get_default_metric_configs,
)


class TestDataConfig:
    """Test cases for DataConfig."""

    def test_data_config_defaults(self) -> None:
        """Test DataConfig default values."""

        config = DataConfig()

        assert config.categorical_fill_strategy == "mode"
        assert config.numerical_fill_strategy == "median"
        assert config.categorical_preprocessing_strategy == "OneHotEncoder"
        assert config.numerical_preprocessing_strategy == "StandardScaler"
        assert config.array_length_strategy == "zero"
        assert config.statistical_feature_set == "minimal"
        assert config.feature_selection is False

    @pytest.mark.parametrize(
        "config_params",
        [
            {"categorical_fill_strategy": "missing"},
            {"numerical_fill_strategy": "mean"},
            {"categorical_preprocessing_strategy": "LabelEncoder"},
            {"numerical_preprocessing_strategy": "MinMaxScaler"},
            {"array_length_strategy": "mean"},
            {"statistical_feature_set": "comprehensive"},
            {
                "categorical_fill_strategy": "missing",
                "numerical_fill_strategy": "zero",
                "categorical_preprocessing_strategy": "all",
                "numerical_preprocessing_strategy": "RobustScaler",
                "array_length_strategy": "median",
                "statistical_feature_set": "efficient",
            },
        ],
    )
    def test_data_config_custom(self, config_params: Any) -> None:
        """Test DataConfig with custom parameters."""

        config = DataConfig(**config_params)

        for param, value in config_params.items():
            assert getattr(config, param) == value

    def test_data_config_invalid_values(self) -> None:
        """Test DataConfig with invalid values."""

        with pytest.raises(ValueError):
            DataConfig(categorical_fill_strategy="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            DataConfig(numerical_fill_strategy="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            DataConfig(categorical_preprocessing_strategy="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            DataConfig(numerical_preprocessing_strategy="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            DataConfig(array_length_strategy="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            DataConfig(statistical_feature_set="invalid")  # type: ignore[arg-type]


class TestMetricConfig:
    """Test cases for MetricConfig."""

    def test_metric_config_defaults(self) -> None:
        """Test MetricConfig default values."""

        config = MetricConfig()

        assert config.direction == "maximize"
        assert config.initial_score == float("-inf")
        assert config.better_score == gt

    @pytest.mark.parametrize(
        "config_params",
        [
            {"direction": "minimize", "initial_score": float("inf"), "better_score": lt},
            {"direction": "maximize", "initial_score": 0.0, "better_score": gt},
            {"initial_score": -1.0},
            {"better_score": lt},
        ],
    )
    def test_metric_config_custom(self, config_params: Any) -> None:
        """Test MetricConfig with custom parameters."""

        config = MetricConfig(**config_params)

        for param, value in config_params.items():
            assert getattr(config, param) == value

    def test_metric_config_validation(self) -> None:
        """Test MetricConfig field validation."""

        # Test valid values
        config = MetricConfig(initial_score=1.5, better_score=gt)
        assert config.initial_score == 1.5
        assert config.better_score == gt

        # Test invalid initial_score
        with pytest.raises(ValidationError, match="Input should be a valid number"):
            MetricConfig(initial_score="invalid")  # type: ignore[arg-type]

        # Test invalid better_score
        with pytest.raises(ValidationError, match="Input should be callable"):
            MetricConfig(better_score="not_callable")  # type: ignore[arg-type]

    def test_metric_config_invalid_values(self) -> None:
        """Test MetricConfig with invalid values."""

        with pytest.raises(ValueError):
            MetricConfig(direction="invalid")  # type: ignore[arg-type]


class TestTrainingConfig:
    """Test cases for TrainingConfig."""

    def test_training_config_defaults(self) -> None:
        """Test TrainingConfig default values."""

        config = TrainingConfig()

        assert config.n_splits == 5
        assert config.random_state == 28
        assert config.n_trials == 2
        assert config.experiment_name == "ai_toolkit_classification_experiment"
        assert config.optimize_metric == "f1"
        assert config.use_smote is True
        assert config.smote_ratio == 1.0

    @pytest.mark.parametrize(
        "config_params",
        [
            {"n_splits": 10},
            {"random_state": 42},
            {"n_trials": 200},
            {"experiment_name": "custom_experiment"},
            {"optimize_metric": "accuracy"},
            {"use_smote": False},
            {"smote_ratio": 0.5},
            {
                "n_splits": 8,
                "random_state": 123,
                "n_trials": 50,
                "experiment_name": "test_experiment",
                "optimize_metric": "precision",
                "use_smote": False,
                "smote_ratio": 0.8,
            },
        ],
    )
    def test_training_config_custom(self, config_params: Any) -> None:
        """Test TrainingConfig with custom parameters."""

        config = TrainingConfig(**config_params)

        for param, value in config_params.items():
            assert getattr(config, param) == value

    def test_training_config_validation(self) -> None:
        """Test TrainingConfig field validation."""

        # Test valid optimize_metric
        config = TrainingConfig(optimize_metric="accuracy")
        assert config.optimize_metric == "accuracy"

        # Test invalid optimize_metric
        with pytest.raises(ValueError, match="optimize_metric 'invalid' not supported."):
            TrainingConfig(optimize_metric="invalid")

    def test_training_config_constraints(self) -> None:
        """Test TrainingConfig field constraints."""

        # Test n_splits constraints
        with pytest.raises(ValueError):
            TrainingConfig(n_splits=1)  # Below minimum

        with pytest.raises(ValueError):
            TrainingConfig(n_splits=11)  # Above maximum

        # Test random_state constraints
        with pytest.raises(ValueError):
            TrainingConfig(random_state=-1)  # Below minimum

        # Test n_trials constraints
        with pytest.raises(ValueError):
            TrainingConfig(n_trials=0)  # Below minimum

        with pytest.raises(ValueError):
            TrainingConfig(n_trials=1001)  # Above maximum

        # Test smote_ratio constraints
        with pytest.raises(ValueError):
            TrainingConfig(smote_ratio=-0.1)  # Below minimum

        with pytest.raises(ValueError):
            TrainingConfig(smote_ratio=1.1)  # Above maximum

    def test_get_metric_configs(self) -> None:
        """Test get_metric_configs method."""

        config = TrainingConfig(optimize_metric="accuracy")
        metric_config = config.get_metric_configs()

        assert isinstance(metric_config, MetricConfig)
        assert metric_config.direction == "maximize"
        assert metric_config.initial_score == float("-inf")
        assert metric_config.better_score == gt


class TestLoggingConfig:
    """Test cases for LoggingConfig."""

    def test_logging_config_defaults(self) -> None:
        """Test LoggingConfig default values."""

        config = LoggingConfig()

        assert config.name == "logger"
        assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.level == logging.INFO
        assert config.dir == Path("logs")
        assert config.file == ".log"
        assert config.enable_console is True
        assert config.enable_file is True

    @pytest.mark.parametrize(
        "config_params",
        [
            {"name": "custom_logger"},
            {"format": "%(message)s"},
            {"level": logging.INFO},
            {"dir": Path("custom_logs")},
            {"file": "custom.log"},
            {"enable_console": False},
            {"enable_file": False},
            {
                "name": "test_logger",
                "format": "%(name)s: %(message)s",
                "level": logging.DEBUG,
                "dir": Path("test_logs"),
                "file": "test.log",
                "enable_console": False,
                "enable_file": True,
            },
        ],
    )
    def test_logging_config_custom(self, config_params: Any) -> None:
        """Test LoggingConfig with custom parameters."""

        config = LoggingConfig(**config_params)

        for param, value in config_params.items():
            assert getattr(config, param) == value

    def test_logging_config_validation(self) -> None:
        """Test LoggingConfig field validation."""

        # Test valid level
        config = LoggingConfig(level=logging.ERROR)
        assert config.level == logging.ERROR

        # Test invalid level types
        with pytest.raises(ValidationError, match="Input should be a valid integer"):
            LoggingConfig(level="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Logging level must be an integer between 0 and 50."):
            LoggingConfig(level=-1)

        with pytest.raises(ValueError, match="Logging level must be an integer between 0 and 50."):
            LoggingConfig(level=51)


class TestAIToolkitConfig:
    """Test cases for AIToolkitConfig."""

    def test_ai_toolkit_config_defaults(self) -> None:
        """Test AIToolkitConfig default values."""

        config = AIToolkitConfig()

        assert config.version == "0.1.0"  # From get_version()
        assert isinstance(config.created_at, str)
        assert isinstance(config.project_root, Path)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.metric, MetricConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.environment == "development"

    def test_ai_toolkit_config_custom(self) -> None:
        """Test AIToolkitConfig with custom parameters."""

        custom_data = DataConfig(categorical_fill_strategy="missing")
        custom_training = TrainingConfig(n_trials=50)
        custom_logging = LoggingConfig(name="custom_logger")

        config = AIToolkitConfig(
            environment="production",
            data=custom_data,
            training=custom_training,
            logging=custom_logging,
        )

        assert config.environment == "production"
        assert config.data.categorical_fill_strategy == "missing"
        assert config.training.n_trials == 50
        assert config.logging.name == "custom_logger"

    def test_ai_toolkit_config_invalid_values(self) -> None:
        """Test AIToolkitConfig with invalid values."""

        with pytest.raises(ValueError):
            AIToolkitConfig(environment="invalid")  # type: ignore[arg-type]

    def test_from_yaml(self) -> None:
        """Test loading config from YAML file."""

        yaml_data = {
            "environment": "production",
            "data": {"categorical_fill_strategy": "missing"},
            "training": {"n_trials": 50},
            "logging": {"name": "yaml_logger"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_data, f)
            temp_path = f.name

        try:
            config = AIToolkitConfig.from_yaml(temp_path)
            assert config.environment == "production"
            assert config.data.categorical_fill_strategy == "missing"
            assert config.training.n_trials == 50
            assert config.logging.name == "yaml_logger"
        finally:
            Path(temp_path).unlink()

    def test_from_json(self) -> None:
        """Test loading config from JSON file."""

        json_data = {
            "environment": "production",
            "data": {"categorical_fill_strategy": "missing"},
            "training": {"n_trials": 50},
            "logging": {"name": "json_logger"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            config = AIToolkitConfig.from_json(temp_path)
            assert config.environment == "production"
            assert config.data.categorical_fill_strategy == "missing"
            assert config.training.n_trials == 50
            assert config.logging.name == "json_logger"
        finally:
            Path(temp_path).unlink()

    def test_from_env(self) -> None:
        """Test loading config from environment variables."""

        with patch.dict(
            "os.environ",
            {
                "AI_TOOLKIT_ENVIRONMENT": "production",
                "AI_TOOLKIT_DATA__CATEGORICAL_FILL_STRATEGY": "missing",
                "AI_TOOLKIT_TRAINING__N_TRIALS": "50",
                "AI_TOOLKIT_LOGGING__NAME": "env_logger",
            },
        ):
            config = AIToolkitConfig.from_env()
            assert config.environment == "production"
            assert config.data.categorical_fill_strategy == "missing"
            assert config.training.n_trials == 50
            assert config.logging.name == "env_logger"

    def test_save_yaml(self) -> None:
        """Test saving config to YAML file."""

        config = AIToolkitConfig(environment="production")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            config.save_yaml(temp_path)

            # Load and verify
            loaded_config = AIToolkitConfig.from_yaml(temp_path)
            assert loaded_config.environment == "production"
        finally:
            Path(temp_path).unlink()

    def test_save_json(self) -> None:
        """Test saving config to JSON file."""

        config = AIToolkitConfig(environment="production")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            config.save_json(temp_path)

            # Load and verify
            loaded_config = AIToolkitConfig.from_json(temp_path)
            assert loaded_config.environment == "production"
        finally:
            Path(temp_path).unlink()

    def test_merge_with(self) -> None:
        """Test merging configurations."""

        base_config = AIToolkitConfig(
            environment="development",
            data=DataConfig(categorical_fill_strategy="mode"),
            training=TrainingConfig(n_trials=100),
        )

        other_config = AIToolkitConfig(
            environment="production",
            data=DataConfig(categorical_fill_strategy="missing"),
            logging=LoggingConfig(name="merged_logger"),
        )

        merged = base_config.merge_with(other_config)

        assert merged.environment == "production"  # Overridden
        assert merged.data.categorical_fill_strategy == "missing"  # Overridden
        assert merged.training.n_trials == 100  # Kept from base
        assert merged.logging.name == "merged_logger"  # From other


class TestConfigFactory:
    """Test cases for ConfigFactory."""

    def test_get_config_singleton(self) -> None:
        """Test ConfigFactory singleton pattern."""

        ConfigFactory._instance = None  # Reset singleton

        config1 = ConfigFactory.get_config()
        config2 = ConfigFactory.get_config()

        assert config1 is config2
        assert isinstance(config1, AIToolkitConfig)

    def test_get_config_reload(self) -> None:
        """Test ConfigFactory reload functionality."""

        ConfigFactory._instance = None  # Reset singleton

        config1 = ConfigFactory.get_config()
        config2 = ConfigFactory.get_config(reload=True)

        assert config1 is not config2
        assert isinstance(config2, AIToolkitConfig)

    def test_create_config_default(self) -> None:
        """Test ConfigFactory create_config with defaults."""

        config = ConfigFactory.create_config()

        assert isinstance(config, AIToolkitConfig)
        assert config.environment == "development"

    def test_create_config_with_environment(self) -> None:
        """Test ConfigFactory create_config with environment."""

        # Create environment-specific config file
        env_config_data = {
            "environment": "production",
            "data": {"categorical_fill_strategy": "missing"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(env_config_data, f)
            env_config_path = Path(f.name)

        # Rename to expected environment config name
        expected_path = env_config_path.parent / "config_production.yaml"
        env_config_path.rename(expected_path)

        try:
            # Change to temp directory for test
            temp_dir = expected_path.parent
            with patch("pathlib.Path.cwd", return_value=temp_dir):
                config = ConfigFactory.create_config(environment="production")

            assert config.environment == "production"
            assert config.data.categorical_fill_strategy == "missing"
        finally:
            expected_path.unlink()

    def test_create_config_with_config_file(self) -> None:
        """Test ConfigFactory create_config with config file."""

        config_data = {
            "environment": "production",
            "training": {"n_trials": 200},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = ConfigFactory.create_config(config_file=temp_path)
            assert config.environment == "production"
            assert config.training.n_trials == 200
        finally:
            Path(temp_path).unlink()

    def test_load_config_file_invalid_format(self) -> None:
        """Test ConfigFactory with invalid config file format."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("invalid content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                ConfigFactory._load_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_save_config_file_invalid_format(self) -> None:
        """Test ConfigFactory save with invalid format."""

        ConfigFactory._instance = AIToolkitConfig()

        with pytest.raises(ValueError, match="Unsupported config file format"):
            ConfigFactory.save_config_file("config.txt")

    def test_getattr_access(self) -> None:
        """Test ConfigFactory __getattr__ access."""

        ConfigFactory._instance = None  # Reset singleton
        factory = ConfigFactory()

        # This should work after getting a config
        config = factory.get_config()
        assert hasattr(config, "training")
        assert isinstance(config.training, TrainingConfig)

        # Test invalid attribute
        with pytest.raises(AttributeError):
            _ = factory.nonexistent_attribute


def test_get_default_metric_configs() -> None:
    """Test get_default_metric_configs function."""

    configs = get_default_metric_configs()

    assert isinstance(configs, dict)
    assert len(configs) > 0

    # Test some expected metrics
    expected_metrics = ["accuracy", "f1", "precision", "recall", "r2", "mean_squared_error"]
    for metric in expected_metrics:
        assert metric in configs
        assert isinstance(configs[metric], MetricConfig)

    # Test metric configurations
    assert configs["accuracy"].direction == "maximize"
    assert configs["accuracy"].better_score == gt

    assert configs["mean_squared_error"].direction == "minimize"
    assert configs["mean_squared_error"].better_score == lt


def test_config_validation_errors() -> None:
    """Test configuration validation errors."""

    # Test DataConfig validation
    with pytest.raises(ValueError):
        DataConfig(categorical_fill_strategy="invalid")  # type: ignore[arg-type]

    # Test TrainingConfig validation
    with pytest.raises(ValueError):
        TrainingConfig(optimize_metric="nonexistent_metric")

    # Test LoggingConfig validation
    with pytest.raises(ValueError):
        LoggingConfig(level="invalid")  # type: ignore[arg-type]

    # Test MetricConfig validation
    with pytest.raises(ValueError):
        MetricConfig(initial_score="not_a_number")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        MetricConfig(better_score="not_callable")  # type: ignore[arg-type]


@pytest.fixture
def sample_configs() -> Dict[str, Any]:
    """Fixture providing sample configurations for testing."""

    return {
        "data": DataConfig(categorical_fill_strategy="missing"),
        "training": TrainingConfig(n_trials=50),
        "logging": LoggingConfig(name="test_logger"),
        "metric": MetricConfig(direction="minimize"),
    }


def test_complete_config_workflow(sample_configs: Any) -> None:
    """Test complete configuration workflow."""

    # Create config with custom components
    config = AIToolkitConfig(
        environment="production",
        data=sample_configs["data"],
        training=sample_configs["training"],
        logging=sample_configs["logging"],
        metric=sample_configs["metric"],
    )

    # Save to file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        config.save_yaml(temp_path)

        # Load from file
        loaded_config = AIToolkitConfig.from_yaml(temp_path)

        # Verify all settings
        assert loaded_config.environment == "production"
        assert loaded_config.data.categorical_fill_strategy == "missing"
        assert loaded_config.training.n_trials == 50
        assert loaded_config.logging.name == "test_logger"
        assert loaded_config.metric.direction == "minimize"

        # Test merge functionality
        other_config = AIToolkitConfig(
            data=DataConfig(numerical_fill_strategy="zero"),
            training=TrainingConfig(random_state=42),
        )

        merged = loaded_config.merge_with(other_config)
        assert merged.data.categorical_fill_strategy == "missing"  # Kept
        assert merged.data.numerical_fill_strategy == "zero"  # Added
        assert merged.training.n_trials == 50  # Kept
        assert merged.training.random_state == 42  # Updated

    finally:
        Path(temp_path).unlink()
