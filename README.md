# AI-Toolkit
![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?logo=semver)
![License](https://img.shields.io/badge/license-Apache%202.0-D22128.svg?logo=apache)
[![CI](https://github.com/Chrissi2802/AI-Toolkit/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Chrissi2802/AI-Toolkit/actions/workflows/python-app.yml)
![Formatting](https://img.shields.io/badge/black-checked-blue.svg)
![Sort Imports](https://img.shields.io/badge/isort-checked-blue.svg)
![Linting](https://img.shields.io/badge/flake8-checked-blue.svg)
![Type Checking](https://img.shields.io/badge/mypy-checked-blue.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg?logo=pre-commit)](https://github.com/Chrissi2802/AI-Toolkit/blob/main/.pre-commit-config.yaml)
![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg?logo=sphinx)

<img src="./docs/ai_toolkit_logo.png" width="320" height="180" alt="Logo">

A comprehensive Python toolkit for machine learning workflows with a focus on model development, training and evaluation.

## 📑 Table of contents
- [Features](#-features)
- [Development & tests](#-development--tests)
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
Data
- Histograms
- Correlation matrix
- Predictive Power Score (PPS)
- Maximal Information Coefficient (MIC)
- Target distribution
- Feature importance correlation
- Outlier boxplot

Models
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
- [mypy](https://mypy.readthedocs.io/en/stable/) for static type checking
- [pytest](https://docs.pytest.org/en/stable/) for testing

### Development tools
This project was developed with support from modern AI coding assistants:
- [Anthropic Claude](https://claude.ai/)
- [Google Antigravity](https://antigravity.google/)

I believe in transparent development practices and acknowledge the role of AI tools in modern software engineering, especially fitting for an AI toolkit! 🚀

The final code quality, architecture decisions, and project direction remain human-driven and thoroughly tested.

### Dependencies
Create new [requirements.txt](requirements.txt):
```bash
# Basic dependencies
pipreqs . --force --mode gt

# Manually add jupyter
jupyter>=1.0.0

# Manual downgrading of the numpy version
numpy>=1.16.0

# Delete duplicate PyYAML
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
conda create --name v_env_ai_toolkit python=3.10
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

# Type checking
mypy

# Tests
pytest

# Specific tests
pytest tests/test_base/
pytest tests/test_models/
pytest tests/test_training/ # takes some time
pytest tests/test_utils/
```

### Code quality checks
Run pre-commit validation which executes linting, testing, docs generation, and project analysis.
```bash
# Run all checks
.\pre-commit.ps1

# Skip tests
.\pre-commit.ps1 -SkipTests

# Skip documentation generation
.\pre-commit.ps1 -SkipDocs

# Skip both tests and documentation, only linting and project analysis
.\pre-commit.ps1 -SkipTests -SkipDocs
```

### Git hooks
This project uses [pre-commit](https://pre-commit.com/) to automatically run formatting and linting checks before every commit.

```bash
# Run manually against all files
pre-commit run --all-files
```

The following hooks run on every `git commit`:
- **trailing-whitespace** — removes trailing whitespace
- **end-of-file-fixer** — ensures files end with a newline
- **check-yaml / check-toml** — validates config file syntax
- **check-merge-conflict** — prevents committing unresolved merge conflicts
- **debug-statements** — catches forgotten `pdb`/`breakpoint()` calls
- **black** — code formatting
- **isort** — import sorting
- **flake8** — linting
- **mypy** — static type checking

## 🚀 Quick start

### Classification
[classification examples](examples/classification_examples.ipynb)

[classification tensorflow examples](examples/classification_tf_examples.ipynb)

### Regression
[regression examples](examples/regression_examples.ipynb)

## 📚 Detailed documentation
### AI Toolkit `ai_toolkit`
- get_version
- get_author
- display_banner

### Base `ai_toolkit.base`
#### Config
- DataConfig
- MetricConfig
- TrainingConfig
- LoggingConfig
- AIToolkitConfig
- ConfigFactory
- get_default_metric_configs

#### Data
- BaseDataset
- extract_statistical_features_from_array
- extract_statistical_features_from_array_tsfresh

#### Models
- BaseMlModel
- BaseMlEnsembleModel

#### Training
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
├── .idea/
│   └── ...
├── ai_toolkit/
│   ├── __init__.py
│   ├── _meta.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── config_development.yaml
│   │   ├── config_production.yaml
│   │   ├── config.py
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
│   ├── logs
│   │   └── ...
│   ├── source/
│   │   └── ...
│   ├── ai_toolkit_logo.png
│   ├── classes.svg
│   ├── make.bat
│   ├── Makefile
│   └── packages.svg
├── examples/
│   ├── classification_examples.ipynb
│   ├── classification_tf_examples.ipynb
│   ├── regression_examples.ipynb
│   ├── logs/
│   │   └── ...
│   └── mlruns/
│       └── ...
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
├── .env.example
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── pre-commit.ps1
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── requirements.txt
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

# Navigate back to the root folder
cd ..
```

You can view the generated documentation by opening the `docs/build/html/index.html` file in your web browser.

### Project2md
Use [project2md](https://pypi.org/project/project2md/) to create a comprehensive markdown summary of the project:
```bash
project2md process --output=.idea/summary.md
```

## 📝 License
This project is licensed under the Apache 2.0 licence - see [LICENSE](LICENSE) file for details.

## 📫 Contact
Chrissi - [GitHub](https://github.com/Chrissi2802)

### 🤝 Contributing
Contributions are very welcome! 😊
Please check out the [Contributing guidelines](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md) for more information.
