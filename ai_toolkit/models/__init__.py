"""
Model implementations for AI Toolkit.
"""

from ai_toolkit.models.classification import (
    DecisionTreeModel,
    EnsembleStackingClassifierModel,
    EnsembleVotingClassifierModel,
    KNNModel,
    LightGBMModel,
    LogisticRegressionModel,
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


__all__ = [
    # Classification models
    "LogisticRegressionModel",
    "SVCModel",
    "KNNModel",
    "NaiveBayesModel",
    "DecisionTreeModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "get_all_classification_models",
    "EnsembleVotingClassifierModel",
    "EnsembleStackingClassifierModel",
    # Regression models
    "RidgeRegressionModel",
    "BayesianRidgeRegressionModel",
    "KNNRegressorModel",
    "SVRModel",
    "XGBoostRegressorModel",
    "LightGBMRegressorModel",
    "CatBoostRegressorModel",
    "get_all_regression_models",
    "EnsembleVotingRegressorModel",
    "EnsembleStackingRegressorModel",
]
