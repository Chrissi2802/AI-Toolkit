import json
import logging
import os
from datetime import datetime
from operator import gt, lt
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

from ai_toolkit._meta import get_version


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    categorical_fill_strategy: Literal["mode", "missing"] = Field(
        default="mode",
        description=(
            "Strategy to fill missing categorical values. Options: "
            "'mode' (fill with mode), "
            "'missing' (fill with 'MISSING')"
        ),
    )

    numerical_fill_strategy: Literal["median", "mean", "zero"] = Field(
        default="median",
        description=(
            "Strategy to fill missing numerical values. Options: "
            "'median' (fill with median), "
            "'mean' (fill with mean), "
            "'zero' (fill with zero)"
        ),
    )

    categorical_preprocessing_strategy: Literal["OneHotEncoder", "LabelEncoder", "all"] = Field(
        default="OneHotEncoder",
        description=(
            "Strategy to preprocess categorical columns. Options: "
            "'OneHotEncoder' (apply one-hot encoding), "
            "'LabelEncoder' (apply label encoding), "
            "'all' (apply both label and one-hot encoding)"
        ),
    )
    numerical_preprocessing_strategy: Literal["StandardScaler", "RobustScaler", "MinMaxScaler"] = (
        Field(
            default="StandardScaler",
            description=(
                "Strategy to preprocess numerical columns. Options: "
                "'StandardScaler' (apply standard scaling), "
                "'RobustScaler' (apply robust scaling), "
                "'MinMaxScaler' (apply min-max scaling)"
            ),
        )
    )

    array_length_strategy: Literal["zero", "mean", "median", "last", "truncate", "global_mean"] = (
        Field(
            default="zero",
            description=(
                "Strategy to handle arrays of different lengths. Options: "
                "'zero' (pad with zeros), "
                "'mean' (pad with array mean), "
                "'median' (pad with array median), "
                "'last' (repeat last value), "
                "'truncate' (cut to shortest length), "
                "'global_mean' (pad with mean of all arrays)"
            ),
        )
    )

    statistical_feature_set: Literal["minimal", "efficient", "comprehensive", "custom"] = Field(
        default="minimal",
        description=(
            "Set of statistical features to extract. Options: "
            "'minimal' (basic features, approx. 10 features), "
            "'efficient' (efficient set, approx. 777 features), "
            "'comprehensive' (comprehensive set, approx. 783 features), "
            "'custom' (custom set, approx. 35 features)"
        ),
    )

    feature_selection: bool = Field(
        default=False,
        description=(
            "Whether to apply feature selection. Options: "
            "'True' (apply feature selection), "
            "'False' (do not apply feature selection)"
        ),
    )

    class Config:
        env_prefix = "DATA_"
        case_sensitive = False
        arbitrary_types_allowed = True


class MetricConfig(BaseModel):
    """Configuration for optimization metrics."""

    direction: Literal["maximize", "minimize"] = Field(
        default="maximize",
        description=(
            "Direction of optuna optimization. Options: "
            "'maximize' (maximize the objective function), "
            "'minimize' (minimize the objective function)"
        ),
    )

    initial_score: float = Field(
        default=float("-inf"),
        description=(
            "Initial score for optuna optimization. Options: "
            "'float('-inf')' (negative infinity, for maximization), "
            "'float('inf')' (positive infinity, for minimization)"
        ),
    )

    better_score: Callable[[float, float], bool] = Field(
        default=gt,
        description=(
            "Function to compare new and old scores. Options: "
            "'gt' (greater than), "
            "'lt' (less than)"
        ),
    )

    @field_validator("initial_score")
    @classmethod
    def validate_initial_score(cls, initial_score: float) -> float:
        """Validate the initial score.

        Args:
            initial_score (float): The initial score to validate.

        Returns:
            float: The validated initial score.
        """

        if initial_score is not None and not isinstance(initial_score, (int, float)):
            raise ValueError("Initial score must be a number.")

        return initial_score

    @field_validator("better_score")
    @classmethod
    def validate_better_score(
        cls, better_score: Callable[[float, float], bool]
    ) -> Callable[[float, float], bool]:
        """Validate the better_score function.

        Args:
            better_score (Callable[[float, float], bool]): The better_score function to validate.

        Returns:
            Callable[[float, float], bool]: The validated better_score function.
        """

        if better_score is not None and not callable(better_score):
            raise ValueError("Better score must be a callable function.")

        return better_score

    class Config:
        env_prefix = "METRIC_"
        case_sensitive = False
        arbitrary_types_allowed = True


def get_default_metric_configs() -> Dict[str, MetricConfig]:
    """Get default metric configurations.

    Returns:
        Dict[str, MetricConfig]: Dictionary of metric configurations.
    """

    dict_metric_configs = {
        # Classification metrics
        "accuracy": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "balanced_accuracy": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "precision": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "recall": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "f1": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "matthews_correlation_coefficient": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "quadratic_weighted_kappa": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "jaccard": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "hamming_loss": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        # "d2_log_loss": MetricConfig(
        #     direction="maximize",
        #     initial_score=float("-inf"),
        #     better_score=gt,
        # ),
        "zero_one_loss": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "log_loss": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "roc_auc": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "brier_score": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        # Regression metrics
        "explained_variance": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "max_error": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "mean_absolute_error": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "mean_squared_error": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "root_mean_squared_error": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "median_absolute_error": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "r2": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "mean_absolute_percentage_error": MetricConfig(
            direction="minimize",
            initial_score=float("inf"),
            better_score=lt,
        ),
        "d2_absolute_error": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "d2_pinball": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        "d2_tweedie": MetricConfig(
            direction="maximize",
            initial_score=float("-inf"),
            better_score=gt,
        ),
        # "mean_squared_log_error": MetricConfig(
        #     direction="minimize",
        #     initial_score=float("inf"),
        #     better_score=lt,
        # ),
        # "root_mean_squared_log_error": MetricConfig(
        #     direction="minimize",
        #     initial_score=float("inf"),
        #     better_score=lt,
        # ),
        # "mean_poisson_deviance": MetricConfig(
        #     direction="minimize",
        #     initial_score=float("inf"),
        #     better_score=lt,
        # ),
        # "mean_gamma_deviance": MetricConfig(
        #     direction="minimize",
        #     initial_score=float("inf"),
        #     better_score=lt,
        # ),
    }

    return dict_metric_configs


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    n_splits: int = Field(default=5, ge=2, le=10, description="Number of cross-validation splits.")

    random_state: int = Field(default=28, ge=0, description="Random state for reproducibility.")

    n_trials: int = Field(default=2, ge=1, le=1000, description="Number of optimization trials.")

    experiment_name: str = Field(
        default="ai_toolkit_classification_experiment",
        description="Name of the MLflow experiment to log results.",
    )

    optimize_metric: str = Field(
        default="f1",
        description="Metric to optimize during hyperparameter optimization.",
    )

    use_smote: bool = Field(
        default=True,
        description="Whether to use SMOTE for imbalanced datasets. "
        "Only for classification. Options: "
        "'True' (use SMOTE), "
        "'False' (do not use SMOTE)",
    )

    smote_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of minority to majority class after SMOTE.",
    )

    # metric_configs: MetricConfig = Field(
    #        default=default_metric_configs[optimize_metric],
    #        description="Configuration for the chosen optimization metric.",
    # )

    @field_validator("optimize_metric")
    @classmethod
    def validate_optimize_metric(cls, optimize_metric: str) -> str:
        """Validate the optimize_metric.

        Args:
            optimize_metric (str): The optimize_metric to validate.

        Returns:
            str: The validated optimize_metric.
        """

        default_metric_configs = get_default_metric_configs()

        if optimize_metric not in default_metric_configs:
            raise ValueError(
                f"optimize_metric '{optimize_metric}' not supported. "
                f"Supported metrics: {list(default_metric_configs.keys())}"
            )

        return optimize_metric

    def get_metric_configs(self) -> MetricConfig:
        """Get the metric configuration for the chosen optimization metric.

        Returns:
            MetricConfig: The metric configuration for the chosen optimization metric.
        """

        default_metric_configs = get_default_metric_configs()

        if self.optimize_metric not in default_metric_configs:
            raise ValueError(
                f"optimize_metric '{self.optimize_metric}' not supported. "
                f"Supported metrics: {list(default_metric_configs.keys())}"
            )

        return default_metric_configs[self.optimize_metric]

    class Config:
        env_prefix = "TRAINING_"
        case_sensitive = False
        arbitrary_types_allowed = True


class LoggingConfig(BaseModel):
    """Configurations for the logger."""

    name: str = Field(default="logger", description="The name of the logger.")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="The format of the log message.",
    )
    level: int = Field(
        default=logging.INFO,
        description=(
            "The logging level. Options: "
            "'DEBUG' (detailed information, typically of interest only when diagnosing problems), "
            "'INFO' (confirmation that things are working as expected), "
            "'WARNING' (an indication that something unexpected happened, "
            "or indicative of some problem in the near future), "
            "'ERROR' (due to a more serious problem, the software has not been able "
            "to perform some function), "
            "'CRITICAL' (a very serious error, indicating that the program itself may be "
            "unable to continue running)"
        ),
    )
    dir: Union[str, Path] = Field(
        default=Path("logs"), description="The directory to save the log file."
    )

    @field_validator("dir", mode="before")
    @classmethod
    def validate_dir(cls, v: Union[str, Path]) -> Path:
        return Path(v)

    file: str = Field(default=".log", description="The name of the log file.")
    enable_console: bool = Field(
        default=True,
        description=(
            "Whether to log to the console. Options: "
            "'true' (enable console logging), "
            "'false' (disable console logging)"
        ),
    )
    enable_file: bool = Field(
        default=True,
        description=(
            "Whether to log to the file. Options: "
            "'true' (enable file logging), "
            "'false' (disable file logging)"
        ),
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, level: int) -> int:
        """Validate the logging level.

        Args:
            level (int): The logging level to validate.

        Returns:
            int: The validated logging level.
        """

        if not isinstance(level, int) or level < 0 or level > 50:
            raise ValueError("Logging level must be an integer between 0 and 50.")

        return level

    class Config:
        env_prefix = "LOGGING_"
        case_sensitive = False
        arbitrary_types_allowed = True


class AIToolkitConfig(BaseSettings):
    """Main configuration for AI Toolkit."""

    version: str = Field(
        default_factory=get_version,
        description="Version of the AI Toolkit configuration.",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="Config creation timestamp."
    )
    project_root: Path = Field(
        default_factory=Path.cwd,
        description="Root directory of the project.",
    )

    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration settings.")
    training: TrainingConfig = Field(
        default_factory=TrainingConfig, description="Training configuration settings."
    )
    metric: MetricConfig = Field(
        default_factory=MetricConfig,
        description="Configuration for optimization metrics.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration settings.",
    )

    environment: Literal["development", "production"] = Field(
        default="development",
        description=(
            "Environment in which the toolkit is running. Options: "
            "'development' (for development and testing), "
            "'production' (for production use)"
        ),
    )

    class Config:
        env_prefix = "AI_TOOLKIT_"
        env_nested_delimiter = "__"
        case_sensitive = False
        arbitrary_types_allowed = True

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AIToolkitConfig":
        """Load configuration from a YAML file.

        Args:
            path (Union[str, Path]): Path to the YAML configuration file.

        Returns:
            AIToolkitConfig: An instance of AIToolkitConfig populated
                with the data from the YAML file.
        """

        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        data = cls._deserialize_special_types(data)

        return cls(**data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "AIToolkitConfig":
        """Load configuration from a JSON file.

        Args:
            path (Union[str, Path]): Path to the JSON configuration file.

        Returns:
            AIToolkitConfig: An instance of AIToolkitConfig populated
                with the data from the JSON file.
        """

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        data = cls._deserialize_special_types(data)

        return cls(**data)

    @classmethod
    def from_env(cls) -> "AIToolkitConfig":
        """Load configuration from environment variables.

        Returns:
            AIToolkitConfig: An instance of AIToolkitConfig populated
                with the data from environment variables.
        """

        return cls()

    def _serialize_special_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize special types to JSON-compatible formats.

        Args:
            data (Dict[str, Any]): The input dictionary.

        Returns:
            Dict[str, Any]: The dictionary with serialized values.
        """

        result: Dict[str, Any] = {}

        for key, value in data.items():
            # Recursively serialize nested dictionaries
            if isinstance(value, dict):
                result[key] = self._serialize_special_types(value)
            # Paths
            elif isinstance(value, Path):
                result[key] = value.as_posix()
            # Operators
            elif value is gt:
                result[key] = "gt"
            elif value is lt:
                result[key] = "lt"
            # Datetime
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value

        return result

    @classmethod
    def _deserialize_special_types(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize special types from a dictionary.

        Args:
            data (Dict[str, Any]): The input dictionary.

        Returns:
            Dict[str, Any]: The dictionary with deserialized values.
        """

        result: Dict[str, Any] = {}

        for key, value in data.items():
            # Recursively deserialize nested dictionaries
            if isinstance(value, dict):
                result[key] = cls._deserialize_special_types(value)
            # Paths
            elif key == "dir" and isinstance(value, str):
                result[key] = Path(value)
            elif key == "project_root" and isinstance(value, str):
                result[key] = Path(value)
            # Operators
            elif key == "better_score" and isinstance(value, str) and value == "gt":
                result[key] = gt
            elif key == "better_score" and isinstance(value, str) and value == "lt":
                result[key] = lt
            else:
                result[key] = value

        return result

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save the configuration to a YAML file.

        Args:
            path (Union[str, Path]): Path to save the YAML configuration file.
        """

        data = self.model_dump()
        data = self._serialize_special_types(data)

        with open(path, "w", encoding="utf-8") as file:
            yaml.dump(
                data,
                file,
                default_flow_style=False,
                allow_unicode=True,
            )

    def save_json(self, path: Union[str, Path]) -> None:
        """Save the configuration to a JSON file.

        Args:
            path (Union[str, Path]): Path to save the JSON configuration file.
        """

        data = self.model_dump()
        data = self._serialize_special_types(data)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

    def merge_with(self, other: "AIToolkitConfig") -> "AIToolkitConfig":
        """Merge with another configuration.

        Args:
            other (AIToolkitConfig): The other configuration to merge with.

        Returns:
            AIToolkitConfig: The merged configuration.
        """

        self_dict = self.model_dump()
        other_dict = other.model_dump()
        default_dict = AIToolkitConfig().model_dump()

        def deep_merge(base: dict, override: dict, default: dict) -> dict:
            """Recursively merge two dictionaries.

            Args:
                base (dict): The base dictionary.
                override (dict): The dictionary with override values.
                default (dict): The default values to fall back on.

            Returns:
                dict: The merged dictionary.
            """

            result = base.copy()

            for key, override_value in override.items():
                # Key not present in base, add it
                if key not in result:
                    result[key] = override_value
                # Both dictionaries, merged recursively
                elif isinstance(result[key], dict) and isinstance(override_value, dict):
                    default_sub = default.get(key, {}) if isinstance(default.get(key), dict) else {}
                    result[key] = deep_merge(result[key], override_value, default_sub)
                # Override value is different from default, replace it
                elif key in default and override_value != default[key]:
                    result[key] = override_value

            return result

        merged = deep_merge(self_dict, other_dict, default_dict)

        return AIToolkitConfig(**merged)


class ConfigFactory:
    """Factory class to create and manage the AI Toolkit configuration."""

    _instance: Optional[AIToolkitConfig] = None

    @classmethod
    def get_config(cls, reload: bool = False) -> AIToolkitConfig:
        """Singleton pattern for global configuration.

        Args:
            reload (bool): If True, reload the configuration even if it already exists.
                Default is False.

        Returns:
            AIToolkitConfig: The global configuration instance.
        """

        if cls._instance is None or reload:
            cls._instance = cls.create_config()

        return cls._instance

    @classmethod
    def create_config(
        cls,
        environment: Literal["development", "production"] = "development",
        config_file: Optional[Union[str, Path]] = None,
    ) -> AIToolkitConfig:
        """Create a configuration instance with various sources.

        Args:
            environment (Optional[str], optional): The environment for the configuration
                (e.g., "development", "production"). Defaults to "development".
            config_file (Optional[Union[str, Path]], optional):
                Path to a configuration file. Defaults to None.

        Returns:
            AIToolkitConfig: The created configuration instance.
        """

        # 1. Base configuration
        config = AIToolkitConfig(environment=environment)

        # 2. Environment-specific config
        cwd = Path.cwd()
        env_config_path_yaml = cwd / f"config_{environment}.yaml"
        env_config_path_json = cwd / f"config_{environment}.json"
        if env_config_path_yaml.exists():
            env_config = cls._load_config_file(env_config_path_yaml)
            config = config.merge_with(env_config)
        elif env_config_path_json.exists():
            env_config = cls._load_config_file(env_config_path_json)
            config = config.merge_with(env_config)

        # 3. User-specified config file
        if config_file:
            user_config = cls._load_config_file(config_file)
            config = config.merge_with(user_config)

        # 4. Environment variables
        env_config = AIToolkitConfig.from_env()
        if any(conf.startswith("AI_TOOLKIT_") for conf in os.environ):
            config = config.merge_with(env_config)

        return config

    @classmethod
    def _load_config_file(cls, path: Union[str, Path]) -> AIToolkitConfig:
        """Load configuration from a file.

        Args:
            path (Union[str, Path]): Path to the configuration file.

        Returns:
            AIToolkitConfig: The loaded configuration.
        """

        path = Path(path)

        if path.suffix.lower() == ".yaml" or path.suffix.lower() == ".yml":
            return AIToolkitConfig.from_yaml(path)
        elif path.suffix.lower() == ".json":
            return AIToolkitConfig.from_json(path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    @classmethod
    def save_config_file(cls, path: Union[str, Path]) -> None:
        """Save the current configuration to a file.

        Args:
            path (Union[str, Path]): Path to save the configuration file.
        """

        if isinstance(path, str):
            path = Path(path)

        if cls._instance is None:
            raise RuntimeError("No configuration loaded. Call get_config() first.")

        if path.suffix.lower() == ".yaml" or path.suffix.lower() == ".yml":
            cls._instance.save_yaml(path)
        elif path.suffix.lower() == ".json":
            cls._instance.save_json(path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    def __getattr__(self, name: str) -> Any:
        """Direct access to config attributes.

        Args:
            name (str): The name of the attribute to access.

        Returns:
            Any: The value of the requested attribute.
        """

        config = self.get_config()
        if hasattr(config, name):
            return getattr(config, name)

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    pass
