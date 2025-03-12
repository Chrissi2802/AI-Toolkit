import numpy as np
import pandas as pd
import pytest

from ai_toolkit.base.data import BaseDataset, DatasetConfig


class DummyDataset(BaseDataset):
    """Dummy dataset implementation for testing."""

    def load_data(self) -> None:
        """Load dummy data."""

        # Create dummy data with mixed types and missing values
        self.X = pd.DataFrame(
            {
                "num1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "num2": [10, 20, 30, np.nan, 50],
                "cat1": ["A", "B", np.nan, "B", "C"],
                "cat2": ["X", np.nan, "Z", "X", "Y"],
            }
        )

        self.y = pd.Series([0, 1, 0, 1, 0], name="target")

        # Test data
        self.X_test = pd.DataFrame(
            {
                "num1": [1.5, np.nan, 4.5],
                "num2": [15, 25, np.nan],
                "cat1": ["A", np.nan, "C"],
                "cat2": [np.nan, "Y", "Z"],
            }
        )


def test_dataset_config_defaults():
    """Test DatasetConfig default values."""

    config = DatasetConfig()

    assert config.CATEGORICAL_FILL_STRATEGY == "mode"
    assert config.NUMERICAL_FILL_STRATEGY == "median"
    assert config.CATEGORICAL_PREPROCESSING_STRATEGY == "OneHotEncoder"
    assert config.NUMERICAL_PREPROCESSING_STRATEGY == "StandardScaler"


@pytest.mark.parametrize(
    "config_params",
    [
        {"CATEGORICAL_FILL_STRATEGY": "missing"},
        {"NUMERICAL_FILL_STRATEGY": "mean"},
        {"CATEGORICAL_PREPROCESSING_STRATEGY": "LabelEncoder"},
        {"NUMERICAL_PREPROCESSING_STRATEGY": "MinMaxScaler"},
        {
            "CATEGORICAL_FILL_STRATEGY": "missing",
            "NUMERICAL_FILL_STRATEGY": "mean",
            "CATEGORICAL_PREPROCESSING_STRATEGY": "all",
            "NUMERICAL_PREPROCESSING_STRATEGY": "RobustScaler",
        },
    ],
)
def test_dataset_config_custom(config_params):
    """Test DatasetConfig with custom parameters.

    Args:
        config_params: Custom configuration parameters.
    """

    config = DatasetConfig(**config_params)

    for param, value in config_params.items():
        assert getattr(config, param) == value


def test_base_dataset_initialization():
    """Test BaseDataset initialization."""

    config = DatasetConfig()
    dataset = DummyDataset(config)

    assert dataset.config == config
    assert dataset.X is None
    assert dataset.X_test is None
    assert dataset.y is None


def test_data_loading():
    """Test data loading functionality."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()

    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.X_test, pd.DataFrame)
    assert isinstance(dataset.y, pd.Series)
    assert len(dataset.X) == 5
    assert len(dataset.X_test) == 3
    assert len(dataset.y) == 5


def test_column_type_detection():
    """Test automatic column type detection."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()
    dataset._detect_column_types()

    assert set(dataset.categorical_columns) == {"cat1", "cat2"}
    assert set(dataset.numerical_columns) == {"num1", "num2"}


@pytest.mark.parametrize("fill_strategy", ["mode", "missing"])
def test_categorical_missing_value_handling(fill_strategy):
    """Test handling of missing values in categorical columns."""

    config = DatasetConfig(CATEGORICAL_FILL_STRATEGY=fill_strategy)
    dataset = DummyDataset(config)
    dataset.load_data()
    dataset._detect_column_types()
    dataset._handle_missing_categorical_values()

    # Check that no missing values remain
    assert not dataset.X[dataset.categorical_columns].isnull().any().any()
    assert not dataset.X_test[dataset.categorical_columns].isnull().any().any()

    # Check missing indicators
    assert "cat1_is_missing" in dataset.X.columns
    assert "cat2_is_missing" in dataset.X.columns

    if fill_strategy == "missing":
        assert "MISSING" in dataset.X["cat1"].values
        assert "MISSING" in dataset.X["cat2"].values


@pytest.mark.parametrize("fill_strategy", ["median", "mean", "zero"])
def test_numerical_missing_value_handling(fill_strategy):
    """Test handling of missing values in numerical columns."""

    config = DatasetConfig(NUMERICAL_FILL_STRATEGY=fill_strategy)
    dataset = DummyDataset(config)
    dataset.load_data()
    dataset._detect_column_types()
    dataset._handle_missing_numerical_values()

    # Check that no missing values remain
    assert not dataset.X[dataset.numerical_columns].isnull().any().any()
    assert not dataset.X_test[dataset.numerical_columns].isnull().any().any()

    # Check missing indicators
    assert "num1_is_missing" in dataset.X.columns
    assert "num2_is_missing" in dataset.X.columns

    if fill_strategy == "zero":
        assert 0.0 in dataset.X["num1"].values
        assert 0.0 in dataset.X["num2"].values


def test_label_encoding():
    """Test label encoding of categorical variables."""

    config = DatasetConfig(CATEGORICAL_PREPROCESSING_STRATEGY="LabelEncoder")
    dataset = DummyDataset(config)
    dataset.load_data()
    dataset._detect_column_types()
    dataset._handle_missing_categorical_values()
    dataset._preprocess_categorical()

    # Check that categorical columns are encoded as numbers
    for col in dataset.categorical_columns:
        assert np.issubdtype(dataset.X[col].dtype, np.number)
        assert np.issubdtype(dataset.X_test[col].dtype, np.number)

    # Check that encoders are stored
    assert hasattr(dataset, "label_encoders")
    assert set(dataset.label_encoders.keys()) == set(dataset.categorical_columns)


def test_one_hot_encoding():
    """Test one-hot encoding of categorical variables."""

    config = DatasetConfig(CATEGORICAL_PREPROCESSING_STRATEGY="OneHotEncoder")
    dataset = DummyDataset(config)
    dataset.load_data()
    dataset._detect_column_types()
    dataset._handle_missing_categorical_values()
    dataset._preprocess_categorical()

    # Original categorical columns should be removed
    assert not any(col in dataset.X.columns for col in ["cat1", "cat2"])
    assert not any(col in dataset.X_test.columns for col in ["cat1", "cat2"])

    # Check for dummy columns
    assert any(col.startswith("cat1_") for col in dataset.X.columns)
    assert any(col.startswith("cat2_") for col in dataset.X.columns)

    # Check that dummy columns are binary
    dummy_cols = [
        col for col in dataset.X.columns if col.startswith(("cat1_", "cat2_"))
    ]
    for col in dummy_cols:
        assert set(dataset.X[col].unique()).issubset({0, 1})
        assert set(dataset.X_test[col].unique()).issubset({0, 1})


def test_all_encoding():
    """Test combined label and one-hot encoding."""

    config = DatasetConfig(CATEGORICAL_PREPROCESSING_STRATEGY="all")
    dataset = DummyDataset(config)
    dataset.load_data()
    dataset._detect_column_types()
    dataset._handle_missing_categorical_values()
    dataset._preprocess_categorical()

    # Check that original categorical columns are encoded
    for col in ["cat1", "cat2"]:
        assert np.issubdtype(dataset.X[col].dtype, np.number)
        assert np.issubdtype(dataset.X_test[col].dtype, np.number)

    # Check for dummy columns
    assert any(col.startswith("cat1_") for col in dataset.X.columns)
    assert any(col.startswith("cat2_") for col in dataset.X.columns)


@pytest.mark.parametrize(
    "scaler_type", ["StandardScaler", "RobustScaler", "MinMaxScaler"]
)
def test_numerical_preprocessing(scaler_type):
    """Test numerical feature scaling."""

    config = DatasetConfig(NUMERICAL_PREPROCESSING_STRATEGY=scaler_type)
    dataset = DummyDataset(config)
    dataset.load_data()

    dataset._check_data()
    dataset._detect_column_types()
    dataset._handle_missing_categorical_values()
    dataset._handle_missing_numerical_values()
    dataset._preprocess_categorical()
    dataset._preprocess_numerical()

    if scaler_type == "StandardScaler":
        for col in dataset.numerical_columns:
            assert abs(dataset.X[col].mean()) < 1e-10
    elif scaler_type == "RobustScaler":
        for col in dataset.numerical_columns:
            assert abs(dataset.X[col].median()) < 1e-10
    elif scaler_type == "MinMaxScaler":
        for col in dataset.numerical_columns:
            assert dataset.X[col].min() >= -1e-10
            assert dataset.X[col].max() <= 1 + 1e-10


def test_complete_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()
    dataset.preprocess()

    # Get processed data
    X, y, X_test = dataset.get_data()

    # Check basic properties
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X) == len(y)
    assert X.shape[1] == X_test.shape[1]

    # Check that no missing values remain
    assert not X.isnull().any().any()
    assert not X_test.isnull().any().any()

    # Check that all columns are numeric
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
    assert all(np.issubdtype(dtype, np.number) for dtype in X_test.dtypes)


def test_get_feature_types():
    """Test getting feature types."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()
    dataset._detect_column_types()

    cat_cols, num_cols = dataset.get_feature_types()
    assert set(cat_cols) == {"cat1", "cat2"}
    assert set(num_cols) == {"num1", "num2"}


def test_error_handling():
    """Test error handling in dataset processing."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()

    # Test data checks
    dataset.X_test = None
    with pytest.raises(ValueError, match="X_test data not loaded."):
        dataset._check_data()

    dataset.y = None
    with pytest.raises(ValueError, match="y data not loaded."):
        dataset._check_data()

    dataset.X = None
    with pytest.raises(ValueError, match="X data not loaded."):
        dataset._check_data()

    dataset = DummyDataset(DatasetConfig())

    # Test preprocessing without loading
    with pytest.raises(ValueError, match="X data not loaded"):
        dataset.preprocess()

    # Test with invalid fill strategy
    with pytest.raises(ValueError):
        config = DatasetConfig(CATEGORICAL_FILL_STRATEGY="invalid")
        dataset = DummyDataset(config)
        dataset.load_data()
        dataset.preprocess()

    with pytest.raises(ValueError):
        config = DatasetConfig(NUMERICAL_FILL_STRATEGY="invalid")
        dataset = DummyDataset(config)
        dataset.load_data()
        dataset.preprocess()

    # Test with invalid preprocessing strategy
    with pytest.raises(ValueError):
        config = DatasetConfig(CATEGORICAL_PREPROCESSING_STRATEGY="invalid")
        dataset = DummyDataset(config)
        dataset.load_data()
        dataset.preprocess()

    with pytest.raises(ValueError):
        config = DatasetConfig(NUMERICAL_PREPROCESSING_STRATEGY="invalid")
        dataset = DummyDataset(config)
        dataset.load_data()
        dataset.preprocess()


def test_dataset_load_and_preprocess():
    """Test data loading and preprocessing pipeline."""

    config = DatasetConfig()
    dataset = DummyDataset(config)

    # Load data
    dataset.load_data()
    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.y, pd.Series)
    assert isinstance(dataset.X_test, pd.DataFrame)

    # Preprocess
    dataset.preprocess()

    # Check that no missing values remain
    assert not dataset.X.isnull().any().any()
    assert not dataset.X_test.isnull().any().any()

    # Check that numerical columns are float
    for col in dataset.numerical_columns:
        assert np.issubdtype(dataset.X[col].dtype, np.number)
        assert np.issubdtype(dataset.X_test[col].dtype, np.number)


def test_get_data():
    """Test getting processed data."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()
    dataset.preprocess()

    X, y, X_test = dataset.get_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X) == len(y)
    assert X.shape[1] == X_test.shape[1]


def test_missing_indicator_creation():
    """Test creation of missing value indicators."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()
    dataset._detect_column_types()

    # Process categorical columns
    dataset._handle_missing_categorical_values()
    for col in dataset.categorical_columns:
        if dataset.X[col].isnull().any():
            assert f"{col}_is_missing" in dataset.X.columns
            assert f"{col}_is_missing" in dataset.X_test.columns

    # Process numerical columns
    dataset._handle_missing_numerical_values()
    for col in dataset.numerical_columns:
        if dataset.X[col].isnull().any():
            assert f"{col}_is_missing" in dataset.X.columns
            assert f"{col}_is_missing" in dataset.X_test.columns
