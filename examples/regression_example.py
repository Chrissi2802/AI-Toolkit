import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import sys

sys.path.append(".")

from ai_toolkit import (
    RidgeRegressionModel,
    get_all_regression_models,
    RegressionModelTrainer,
    lazypredict_regression,
)

# from ai_toolkit.models.regression import RidgeRegressionModel, get_all_regression_models
# from ai_toolkit.training.regression import (
#     RegressionModelTrainer,
#     lazypredict_regression,
# )


# Example data
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
X_reg, y_reg = load_diabetes(return_X_y=True, as_frame=True)

scaler = StandardScaler()
X_reg = pd.DataFrame(data=scaler.fit_transform(X_reg), columns=X_reg.columns)

N_SPLITS = 5
RANDOM_STATE = 28
N_TRAILS = 2  # 100


# Regression
# Train one example model
base_model = RidgeRegressionModel()

# Create a regression model trainer
trainer = RegressionModelTrainer(
    base_model=base_model,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    experiment_name="ml_regression",
    optimize_metric="root_mean_squared_error",
)

# Train and optimize the model
best_model, mean_metrics = trainer.train_and_optimize(
    X=X_reg,
    y=y_reg,
    n_trials=N_TRAILS,
)

# y_pred = trainer.predict(X_test)

# Train all regression models
base_models = get_all_regression_models()

for base_model in base_models.values():

    # Create a regression model trainer
    trainer = RegressionModelTrainer(
        base_model=base_model,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        experiment_name="ml_regression",
        optimize_metric="root_mean_squared_error",
    )

    # Train and optimize the model
    best_model, mean_metrics = trainer.train_and_optimize(
        X=X_reg,
        y=y_reg,
        n_trials=N_TRAILS,
    )

df_mean_results = lazypredict_regression(
    X=X_reg,
    y=y_reg,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
)

print(df_mean_results)


if __name__ == "__main__":
    pass
