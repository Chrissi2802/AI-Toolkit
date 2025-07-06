# AI-Toolkit
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?logo=semver)
![License](https://img.shields.io/badge/license-Apache%202.0-D22128.svg?logo=apache)
![Formatting](https://img.shields.io/badge/black-checked-blue.svg)
![Sort Imports](https://img.shields.io/badge/isort-checked-blue.svg)
![Linting](https://img.shields.io/badge/flake8-checked-blue.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg?logo=pytest)
![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg?logo=sphinx)

<img src="./docs/ai_toolkit_logo.png" width="320" height="180" alt="Logo">

A comprehensive Python toolkit for machine learning workflows with a focus on model development, training and evaluation.

## 📑 Table of contents
- [Features](#-features)
- [Development & Tests](#-development--tests)
- [Quick start](#-quick-start)
- [Detailed documentation](#-detailed-documentation)
- [License](#-license)
- [Contact](#-contact)

## 🚀 Features
**Model variety**
- Classification (Logistic Regression, SVC, KNN, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM)
- Regression (Ridge, Bayesian Ridge, SVR, KNN, XGBoost, LightGBM, CatBoost)
- Ensemble models (Voting, Stacking)

**Automated training**
- Hyperparameter optimization with [Optuna](https://optuna.org/)
- K-fold cross-validation 
  - [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
  - [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) for imbalanced data
- [MLflow](https://mlflow.org/) integration for experiment tracking
  - `mlflow ui --port 5000` → [http://localhost:5000](http://localhost:5000)
- [Lazy Predict](https://github.com/shankarpandala/lazypredict) integration for rapid model evaluation and comparison

**Extensive evaluation**
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, etc.)
- Regression metrics (RMSE, MAE, R², etc.)
- Cross-validation metrics

**Visualisation**
- ROC curve
- Confusion matrix
- Residual plots
- Prediction scatter plots
- Feature Importance
- SHAP values

## 🔬 Development & tests
### Code Style
We use:
- [Visual Studio Code](https://code.visualstudio.com/) for development
- [black](https://black.readthedocs.io/en/stable/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/en/latest/) for linting
- [pytest](https://docs.pytest.org/en/stable/) for testing

### Dependencies
[requirements.txt](requirements.txt)

Create new requirements.txt:
```bash
# Basic dependencies
pipreqs . --force

# Manually add jupyter
jupyter>=1.0.0

# Manual downgrading of the numpy version
numpy>=1.16.0
```

### 📦 Setup
```bash
# Clone repository
git clone https://github.com/Chrissi2802/AI-Toolkit.git
cd ai-toolkit

# Create a virtual environment
python -m venv v_env_ai_toolkit
source v_env_ai_toolkit/bin/activate  # Linux / Mac
v_env_ai_toolkit\Scripts\activate     # Windows

# Create a virtual environment with conda
conda create --name v_env_ai_toolkit python=3.12
conda activate v_env_ai_toolkit

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
# Formatting
black .

# Import sorting
isort .

# Linting
flake8 .

# Tests
pytest

# Specific tests
pytest tests/test_base/
pytest tests/test_models/
pytest tests/test_training/ # takes some time
pytest tests/test_utils/
```

## 🚀 Quick start

### Classification
[classification examples](examples/classification_examples.ipynb)

[classification tensorflow examples](examples/classification_tf_examples.ipynb)

### Regression
[regression examples](examples/regression_examples.ipynb)

## 📚 Detailed documentation
### Base `ai_toolkit.base`
#### Data
- DatasetConfig
- BaseDataset
- extract_statistical_features_from_array

#### Models
- BaseMlModel
- BaseMlEnsembleModel

#### Training
- MetricConfig
- get_default_metric_configs
- MlTrainerConfig
- BaseMlTrainer

### Models `ai_toolkit.models`
#### Classification
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Nearest Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Naive Bayes Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost Classifier](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [LightGBM Classifier](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [MobileNetV3Small Classifier](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small)
- get_all_classification_models

Ensembles
- [Voting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [Stacking Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

#### Regression
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Bayesian Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
- [Support Vector Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [K-Nearest Neighbors Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [XGBoost Regressor](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [LightGBM Regressor](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [CatBoost Regressor](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor)
- get_all_regression_models

Ensembles
- [Voting Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
- [Stacking Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)

### Training `ai_toolkit.training`
#### Classification
- ClassificationModelTrainer
- lazypredict_classification

#### Regression
- RegressionModelTrainer
- lazy_predict_regression

### Utils `ai_toolkit.utils`
#### Evaluation
- ClassificationMetrics
- RegressionMetrics
- CrossValidationMetrics

#### Logging
- LoggerConfig
- Logger
- get_logger

#### Visualisation
- ClassificationPlots
- RegressionPlots
- ModelAnalysisPlots

### 📁 Project Structure
```bash
AI-Toolkit/
├── .git/
│   └── ...
├── .github/
│   └── workflows/
│       └── ...
├── ai_toolkit/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── models.py
│   │   └── training.py
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
│        ├── logging.py
│        └── visualization.py
├── docs/
│   ├── build/
│   │   └── ...
│   ├── source/
│   │   └── ...
│   ├── ai_toolkit_logo.png
│   ├── classes.svg
│   ├── make.bat
│   ├── Makefile
│   └── packages.svg
├── examples/
│   ├── classification_example.ipynb
│   └── regression_example.ipynb
├── logs/
│   └── ...
├── mlruns/
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_init.py
│   │── test_base/
│   │   └── ...
│   │── test_models/
│   │   └── ...
│   │── test_training/
│   │   └── ...
│   └── test_utils/
│       └── ...
├── .env
├── .flake8
├── .gitignore
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-dev.txt
├── requirements.txt
└── template.env
```

### Architecture
The [docs](docs) contain diagrams that show the class structure and package dependencies of the project.

```bash
# Create class diagram
pyreverse -o svg -d .\docs .\ai_toolkit\
```

### Sphinx Documentation 
The documentation is generated using Sphinx. To build the documentation, follow these steps:
```bash
# Navigate to the docs folder
cd docs

# If you want to create a clean new documentation
sphinx-quickstart

# Setup the documentation structure
sphinx-apidoc -o source/ ../ai_toolkit/

# Clean the documentation
.\make.bat clean

# Build the documentation
.\make.bat html
```

You can view the generated documentation by opening the `docs/build/html/index.html` file in your web browser.

## 📝 License
This project is licensed under the Apache 2.0 licence - see [LICENSE](LICENSE) file for details.

## 📫 Contact
Chrissi - [GitHub](https://github.com/Chrissi2802)

### 🤝 Contributing
Contributions are very welcome! 😊
Please check out the [Contributing guidelines](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) for more information.
