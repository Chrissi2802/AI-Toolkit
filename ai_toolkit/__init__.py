"""
Comprehensive toolkit for AI.
"""

# meta
from ai_toolkit._meta import __author__, __version__, display_banner, get_author, get_version

# base
from ai_toolkit.base.config import (
    AIToolkitConfig,
    ConfigFactory,
    DataConfig,
    LoggingConfig,
    MetricConfig,
    TrainingConfig,
    get_default_metric_configs,
)
from ai_toolkit.base.data import (
    BaseDataset,
    extract_statistical_features_from_array,
    extract_statistical_features_from_array_tsfresh,
)
from ai_toolkit.base.models import BaseMlEnsembleModel, BaseMlModel
from ai_toolkit.base.training import BaseMlTrainer

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
    get_logger,
)
from ai_toolkit.utils.visualization import (
    ClassificationPlots,
    ModelAnalysisPlots,
    RegressionPlots,
)
