import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import sys

sys.path.append(".")

from ai_toolkit import (
    LogisticRegressionModel,
    get_all_classification_models,
    ClassificationModelTrainer,
    lazypredict_classification,
)

# from ai_toolkit.models.classification import (
#     LogisticRegressionModel,
#     get_all_classification_models,
# )
# from ai_toolkit.training.classification import (
#     ClassificationModelTrainer,
#     lazypredict_classification,
# )


# Example data
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
X_clf, y_clf = load_breast_cancer(return_X_y=True, as_frame=True)

scaler = StandardScaler()
X_clf = pd.DataFrame(data=scaler.fit_transform(X_clf), columns=X_clf.columns)

N_SPLITS = 5
RANDOM_STATE = 28
N_TRAILS = 2  # 100


# Classification
# Train one example model
base_model = LogisticRegressionModel()

# Create a classification model trainer
trainer = ClassificationModelTrainer(
    base_model=base_model,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
    experiment_name="ml_classification",
    optimize_metric="f1",
    use_smote=True,
    smote_ratio=1.0,
)

# Train and optimize the model
best_model, mean_metrics = trainer.train_and_optimize(
    X=X_clf,
    y=y_clf,
    n_trials=N_TRAILS,
)

# y_pred = trainer.predict(X_test)

# Train all classification models
base_models = get_all_classification_models()

for base_model in base_models.values():

    # Create a classification model trainer
    trainer = ClassificationModelTrainer(
        base_model=base_model,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        experiment_name="ml_classification",
        optimize_metric="f1",
        use_smote=True,
        smote_ratio=1.0,
    )

    # Train and optimize the model
    best_model, mean_metrics = trainer.train_and_optimize(
        X=X_clf,
        y=y_clf,
        n_trials=N_TRAILS,
    )

df_mean_results = lazypredict_classification(
    X=X_clf,
    y=y_clf,
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
)

print(df_mean_results)


if __name__ == "__main__":
    pass
