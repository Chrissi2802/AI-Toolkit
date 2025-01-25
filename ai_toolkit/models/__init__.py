"""
Model implementations for AI Toolkit.
"""

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
    # Regression models
    "RidgeRegressionModel",
    "BayesianRidgeRegressionModel",
    "KNNRegressorModel",
    "SVRModel",
    "XGBoostRegressorModel",
    "LightGBMRegressorModel",
    "CatBoostRegressorModel",
    "get_all_regression_models",
]
