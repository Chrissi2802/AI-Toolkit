"""
Comprehensive toolkit for AI.
"""

__version__ = "0.1"
__author__ = "Chrissi"

# base
from ai_toolkit.base.model import BaseMlModel
from ai_toolkit.base.ensemble import BaseMlEnsembleModel

# models
from ai_toolkit.models.classification import (
    LogisticRegressionModel,
    SVCModel,
    KNNModel,
    NaiveBayesModel,
    DecisionTreeModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    get_all_classification_models,
)

from ai_toolkit.models.regression import (
    RidgeRegressionModel,
    BayesianRidgeRegressionModel,
    SVRModel,
    KNNRegressorModel,
    XGBoostRegressorModel,
    LightGBMRegressorModel,
    CatBoostRegressorModel,
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
    RegressionMetrics,
    CrossValidationMetrics,
)

from ai_toolkit.utils.visualization import (
    ClassificationPlots,
    RegressionPlots,
    ModelAnalysisPlots,
)


def get_version():
    """Get the current version of AI-Toolkit."""

    return __version__
