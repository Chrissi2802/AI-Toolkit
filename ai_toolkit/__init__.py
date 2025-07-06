"""
Comprehensive toolkit for AI.
"""

import pyfiglet


__version__ = "0.1.0"
__author__ = "Chrissi"

# base
from ai_toolkit.base.data import BaseDataset, DatasetConfig, extract_statistical_features_from_array
from ai_toolkit.base.models import BaseMlEnsembleModel, BaseMlModel
from ai_toolkit.base.training import (
    BaseMlTrainer,
    MetricConfig,
    MlTrainerConfig,
    get_default_metric_configs,
)

# models
from ai_toolkit.models.classification import (
    DecisionTreeModel,
    EnsembleStackingClassifierModel,
    EnsembleVotingClassifierModel,
    KNNModel,
    LightGBMModel,
    LogisticRegressionModel,
    MobileNetV3SmallModel,
    NaiveBayesModel,
    RandomForestModel,
    SVCModel,
    XGBoostModel,
    get_all_classification_models,
)
from ai_toolkit.models.regression import (
    BayesianRidgeRegressionModel,
    CatBoostRegressorModel,
    EnsembleStackingRegressorModel,
    EnsembleVotingRegressorModel,
    KNNRegressorModel,
    LightGBMRegressorModel,
    RidgeRegressionModel,
    SVRModel,
    XGBoostRegressorModel,
    get_all_regression_models,
)

# training
from ai_toolkit.training.classification import (
    ClassificationModelTrainer,
    lazypredict_classification,
)
from ai_toolkit.training.regression import (
    RegressionModelTrainer,
    lazypredict_regression,
)

# utils
from ai_toolkit.utils.evaluation import (
    ClassificationMetrics,
    CrossValidationMetrics,
    RegressionMetrics,
)
from ai_toolkit.utils.logging import (
    Logger,
    LoggerConfig,
    get_logger,
)
from ai_toolkit.utils.visualization import (
    ClassificationPlots,
    ModelAnalysisPlots,
    RegressionPlots,
)


def get_version() -> str:
    """Get the current version of AI-Toolkit.

    Returns:
        str: The current version of AI-Toolkit.
    """

    return __version__


def get_author() -> str:
    """Get the author of AI-Toolkit.

    Returns:
        str: The author of AI-Toolkit.
    """

    return __author__


def display_banner() -> None:
    """Display the banner for AI-Toolkit."""

    print(pyfiglet.figlet_format("AI - TOOLKIT", font="standard"))
    print(f"Version: {get_version()}")
    print(f"Author:  {get_author()}")
