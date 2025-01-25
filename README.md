# AI-Toolkit
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![Version](https://img.shields.io/badge/version-0.1-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Code Style](https://img.shields.io/badge/flake8-checked-blue.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A comprehensive Python toolkit for machine learning workflows with a focus on model development, training and evaluation.

## 📑 Table of contents
- [Features](#-features)
- [Development & Tests](#-development--tests)
- [Quickstart](#-quickstart)
- [Documentation](#-detailed-documentation)
- [Licence](#-licence)
- [Contact](#-contact)

## 🚀 Features
**Model variety**
- Classification (Logistic Regression, SVC, KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM)
- Regression (Ridge, Bayesian Ridge, SVR, KNN, XGBoost, LightGBM, CatBoost)
- Ensemble models

**Automated training**
- Hyperparameter optimization with [Optuna](https://optuna.org/)
- Cross-validation
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) for imbalanced data
- [MLflow](https://mlflow.org/) integration for experiment tracking
  - `mlflow ui --port 5000` → [http://localhost:5000](http://localhost:5000)

**Extensive evaluation**
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, etc.)
- Regression metrics (RMSE, MAE, R², etc.)
- Cross-validation metrics

**Visualisation**
- ROC curve
- Confusion matrix
- Residual plots
- Feature Importance
- Prediction scatter plots

## 🔬 Development & tests
### Code Style
We use:
- [Visual Studio Code](https://code.visualstudio.com/) for development
- [flake8](https://flake8.pycqa.org/en/latest/) for linting
- [pytest](https://docs.pytest.org/en/stable/) for testing

### Dependencies
[requirements.txt](requirements.txt)

Create new requirements.txt:
```bash
# Basic dependencies
pipreqs . --force
```

### 📦 Setup
```bash
# Clone repository
git clone https://github.com/Chrissi2802/AI-Toolkit.git
cd ai-toolkit

# Create a virtual environment
python -m v_env v_env
source v_env/bin/activate  # Linux/Mac
v_env\Scripts\activate     # Windows

# Install the ai toolkit
pip install -e .

# Install the ai toolkit with developer tools
pip install -e ".[dev]"
```

```bash
# Install only the required dependencies
pip install -r requirements.txt

# Install only the required dependencies with developer tools
pip install -r requirements-dev.txt
```

### 🧪 Tests
```bash
# Linting
flake8 .

# Tests
pytest

# Specific tests
pytest tests/test_base/
pytest tests/test_models/
pytest tests/test_training/
pytest tests/test_utils/
```

## 🚀 Quick start
[examples.ipynb](examples/examples.ipynb)

### Classification
[classification_example.py](examples/classification_example.py)

### Regression
[regression_example.py](examples/regression_example.py)

## 📚 Detailed documentation
### Models
#### Classification
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Nearest Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost Classifier](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [LightGBM Classifier](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

#### Regression
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Bayesian Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
- [Support Vector Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [K-Nearest Neighbors Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [XGBoost Regressor](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [LightGBM Regressor](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [CatBoost Regressor](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor)

### Training
- ClassificationModelTrainer
- lazypredict_classification
- RegressionModelTrainer
- lazy_predict_regression

### Evaluation
- ClassificationMetrics
- RegressionMetrics
- CrossValidationMetrics

### Visualisation
- ClassificationPlots
- RegressionPlots
- ModelAnalysisPlots

### Structure
```bash
AI-Toolkit/
├── .github/
│   └── workflows/
│       └── ...
├── ai_toolkit/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── ensemble.py
│   │   └── model.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   └── regression.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   └── regression.py
│   └── utils/
│        ├── __init__.py
│        ├── evaluation.py
│        └── visualization.py
├── examples/
│   ├── classification_example.py
│   ├── examples.ipynb
│   └── regression_example.py
├── tests/
│   └── test_base/
│       └── ...
│   └── test_models/
│       └── ...
│   └── test_training/
│       └── ...
│   └── test_utils/
│       └── ...
├── .env
├── .flake8
├── .gitignore
├── LICENSE
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements-dev.txt
├── requirements.txt
└── template.env
```

## 📝 Licence
This project is licensed under the Apache 2.0 licence - see [LICENSE](LICENSE) file for details.

## 📫 Contact
Chrissi - [GitHub](https://github.com/Chrissi2802)

### 🤝 Contributing
Contributions are very welcome! 😊

Make sure that:
- The linting is okay (`flake8`)
- The tests are successful (`pytest`)
