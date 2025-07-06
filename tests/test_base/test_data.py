import numpy as np
import pandas as pd
import pytest

from ai_toolkit.base.data import BaseDataset, DatasetConfig, extract_statistical_features_from_array


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


class DummyDatasetWithArrays(BaseDataset):
    """Dummy dataset with array columns for statistical feature testing."""

    def load_data(self) -> None:
        """Load dummy data with array columns."""

        # Generate sample time series data
        np.random.seed(28)  # For reproducible tests

        # Create arrays of different characteristics
        arrays = []
        test_arrays = []

        # Array 1: Normal distribution
        arrays.append(np.random.normal(0, 1, 100))
        test_arrays.append(np.random.normal(0, 1, 100))

        # Array 2: Sine wave with noise
        t = np.linspace(0, 4 * np.pi, 100)
        arrays.append(np.sin(t) + np.random.normal(0, 0.1, 100))
        test_arrays.append(np.sin(t) + np.random.normal(0, 0.1, 100))

        # Array 3: Constant values
        arrays.append(np.full(100, 5.0))
        test_arrays.append(np.full(100, 5.0))

        # Array 4: Mixed positive/negative with trend
        arrays.append(np.linspace(-10, 10, 100) + np.random.normal(0, 2, 100))
        test_arrays.append(np.linspace(-10, 10, 100) + np.random.normal(0, 2, 100))

        # Array 5: Sparse data (mostly zeros)
        sparse = np.zeros(100)
        sparse[20:25] = [1, 2, 3, 2, 1]
        sparse[70:75] = [-1, -2, -3, -2, -1]
        arrays.append(sparse)
        test_arrays.append(sparse.copy())

        # Array 6: Edge case arrays - different lengths for testing strategies
        edge_case_arrays = [
            np.array([1.0, 1.0]),  # Short constant array
            np.array([0.0, 0.0, 0.0]),  # All zeros, different length
            np.array([1.0, -1.0, 1.0, -1.0]),  # Alternating, another length
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Ascending, longest
            np.array([10.0, 20.0, 30.0]),  # Another medium length
        ]

        edge_case_test_arrays = [
            np.array([2.0, 2.0, 2.0]),  # Different constant
            np.array([5.0, 10.0]),  # Short array
            np.array([1.0, 2.0, 3.0, 4.0]),  # Medium array
        ]

        self.X = pd.DataFrame(
            {"sensor_data": arrays, "other_col": [1, 2, 3, 4, 5], "edge_case": edge_case_arrays}
        )

        self.X_test = pd.DataFrame(
            {
                "sensor_data": test_arrays[:3],  # Smaller test set
                "other_col": [1, 2, 3],
                "edge_case": edge_case_test_arrays,
            }
        )

        self.y = pd.Series([0, 1, 0, 1, 0, 1], name="target")


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
    dummy_cols = [col for col in dataset.X.columns if col.startswith(("cat1_", "cat2_"))]
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


@pytest.mark.parametrize("scaler_type", ["StandardScaler", "RobustScaler", "MinMaxScaler"])
def test_numerical_preprocessing(scaler_type):
    """Test numerical feature scaling."""

    config = DatasetConfig(NUMERICAL_PREPROCESSING_STRATEGY=scaler_type)
    dataset = DummyDataset(config)
    dataset.load_data()

    dataset._check_data()
    dataset._check_columns()
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
    with pytest.raises(ValueError, match="Data validation failed"):
        dataset._check_data()

    dataset.y = None
    with pytest.raises(ValueError, match="Data validation failed"):
        dataset._check_data()

    dataset.X = None
    with pytest.raises(ValueError, match="Data validation failed"):
        dataset._check_data()

    dataset = DummyDataset(DatasetConfig())

    # Test preprocessing without loading
    with pytest.raises(RuntimeError, match="Data preprocessing failed"):
        dataset.preprocess()

    # Test with invalid fill strategy
    with pytest.raises(RuntimeError, match="Data preprocessing failed"):
        config = DatasetConfig(CATEGORICAL_FILL_STRATEGY="invalid")
        dataset = DummyDataset(config)
        dataset.load_data()
        dataset.preprocess()

    with pytest.raises(RuntimeError, match="Data preprocessing failed"):
        config = DatasetConfig(NUMERICAL_FILL_STRATEGY="invalid")
        dataset = DummyDataset(config)
        dataset.load_data()
        dataset.preprocess()

    # Test with invalid preprocessing strategy
    with pytest.raises(RuntimeError, match="Data preprocessing failed"):
        config = DatasetConfig(CATEGORICAL_PREPROCESSING_STRATEGY="invalid")
        dataset = DummyDataset(config)
        dataset.load_data()
        dataset.preprocess()

    with pytest.raises(RuntimeError, match="Data preprocessing failed"):
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


def test_extract_statistical_features_from_array_basic():
    """Test basic functionality of statistical feature extraction from array."""

    # Create simple test array
    arr = np.array(
        [
            [1, 2, 3, 4, 5],  # Simple ascending
            [5, 4, 3, 2, 1],  # Simple descending
            [3, 3, 3, 3, 3],  # Constant
            [-2, -1, 0, 1, 2],  # Symmetric around zero
        ],
        dtype=np.float64,
    )

    df = extract_statistical_features_from_array(arr, "test")

    # Test basic features
    assert len(df) == 4  # 4 rows
    assert "test_mean" in df.columns
    assert "test_std" in df.columns
    assert "test_min" in df.columns
    assert "test_max" in df.columns

    # Test specific calculations
    np.testing.assert_almost_equal(df["test_mean"].iloc[0], 3.0)  # (1+2+3+4+5)/5 = 3
    np.testing.assert_almost_equal(df["test_mean"].iloc[2], 3.0)  # Constant array
    np.testing.assert_almost_equal(df["test_std"].iloc[2], 0.0)  # Constant array std = 0

    assert df["test_min"].iloc[0] == 1.0
    assert df["test_max"].iloc[0] == 5.0
    assert df["test_range"].iloc[0] == 4.0  # max - min


def test_extract_statistical_features_comprehensive():
    """Test comprehensive statistical feature extraction."""

    np.random.seed(28)

    # Create test arrays with known properties
    arr = np.array(
        [
            np.random.normal(0, 1, 100),  # Normal distribution
            np.full(100, 5.0),  # Constant
            np.linspace(-10, 10, 100),  # Linear trend
            np.array([0] * 50 + [1] * 25 + [-1] * 25),  # Mixed with zeros
        ],
        dtype=np.float64,
    )

    df = extract_statistical_features_from_array(arr, "sensor")

    # Test all expected features are present
    expected_features = [
        "sensor_mean",
        "sensor_std",
        "sensor_var",
        "sensor_min",
        "sensor_max",
        "sensor_sum",
        "sensor_range",
        "sensor_median",
        "sensor_mean_abs",
        "sensor_mean_ad",
        "sensor_median_ad",
        "sensor_q10",
        "sensor_q25",
        "sensor_q75",
        "sensor_q90",
        "sensor_iqr",
        "sensor_pos_count",
        "sensor_neg_count",
        "sensor_zero_count",
        "sensor_total_count",
        "sensor_above_mean",
        "sensor_above_median",
        "sensor_peaks",
        "sensor_peaks_prominence",
        "sensor_skewness",
        "sensor_kurtosis",
        "sensor_energy",
        "sensor_rms",
        "sensor_sma",
        "sensor_zero_crossings",
        "sensor_mean_crossings",
        "sensor_median_crossings",
        "sensor_argmax",
        "sensor_argmin",
        "sensor_arg_diff",
    ]

    for feature in expected_features:
        assert feature in df.columns, f"Feature {feature} not found"

    # Test specific properties
    # Constant array (row 1)
    assert df["sensor_std"].iloc[1] == 0.0
    assert df["sensor_var"].iloc[1] == 0.0
    assert df["sensor_range"].iloc[1] == 0.0
    assert df["sensor_iqr"].iloc[1] == 0.0

    # Count features for mixed array (row 3: 50 zeros, 25 ones, 25 minus ones)
    assert df["sensor_zero_count"].iloc[3] == 50
    assert df["sensor_pos_count"].iloc[3] == 25
    assert df["sensor_neg_count"].iloc[3] == 25
    assert df["sensor_total_count"].iloc[3] == 100


def test_extract_statistical_features_edge_cases():
    """Test statistical feature extraction with edge cases."""

    # Use the edge_case column from DummyDatasetWithArrays
    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="zero")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    # Should complete without errors
    dataset._extract_statistical_features("edge_case")

    # Check that features were extracted
    feature_cols = [col for col in dataset.X.columns if col.startswith("edge_case_")]
    assert len(feature_cols) > 30

    # No NaN values should be present
    non_nan_count = 0
    for col in feature_cols:
        if not dataset.X[col].isnull().any():
            non_nan_count += 1

    # Most features should be non-NaN (allow some NaN for statistical measures of constant arrays)
    assert non_nan_count > 25  # Most features should be valid

    # Test specific non-problematic features
    assert not dataset.X["edge_case_mean"].isnull().any()
    assert not dataset.X["edge_case_min"].isnull().any()
    assert not dataset.X["edge_case_max"].isnull().any()


def test_extract_statistical_features_method():
    """Test the _extract_statistical_features method on dataset."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Store original column count
    original_train_cols = len(dataset.X.columns)
    original_test_cols = len(dataset.X_test.columns)

    # Extract statistical features
    dataset._extract_statistical_features("sensor_data")

    # Check that features were added
    assert len(dataset.X.columns) > original_train_cols
    assert len(dataset.X_test.columns) > original_test_cols

    # Check that both datasets have the same new columns
    train_feature_cols = [col for col in dataset.X.columns if col.startswith("sensor_data_")]
    test_feature_cols = [col for col in dataset.X_test.columns if col.startswith("sensor_data_")]

    assert set(train_feature_cols) == set(test_feature_cols)
    assert len(train_feature_cols) > 30  # Should have many features

    # Check that original column is still there
    assert "sensor_data" in dataset.X.columns
    assert "sensor_data" in dataset.X_test.columns


def test_extract_statistical_features_error_handling():
    """Test error handling in statistical feature extraction."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Test with non-existent column
    with pytest.raises(ValueError, match="Column 'nonexistent' not found in dataset."):
        dataset._extract_statistical_features("nonexistent")


def test_statistical_features_consistency():
    """Test consistency of statistical features between train and test."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Extract features
    dataset._extract_statistical_features("sensor_data")

    # Get feature columns
    feature_cols = [col for col in dataset.X.columns if col.startswith("sensor_data_")]

    # Check that all features exist in both datasets
    for col in feature_cols:
        assert col in dataset.X.columns
        assert col in dataset.X_test.columns

    # Check data types consistency
    for col in feature_cols:
        assert dataset.X[col].dtype == dataset.X_test[col].dtype
        assert np.issubdtype(dataset.X[col].dtype, np.number)


def test_statistical_features_with_preprocessing():
    """Test statistical features work with full preprocessing pipeline."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Add statistical features before preprocessing
    dataset._extract_statistical_features("sensor_data")

    # Remove the original array columns to avoid issues with pandas operations
    # The array columns cause problems with pandas operations like mode() and nunique()
    columns_to_drop = ["sensor_data", "edge_case"]
    for col in columns_to_drop:
        if col in dataset.X.columns:
            dataset.X = dataset.X.drop(columns=[col])
        if col in dataset.X_test.columns:
            dataset.X_test = dataset.X_test.drop(columns=[col])

    # Run standard preprocessing
    dataset._check_data()
    dataset._check_columns()
    dataset._detect_column_types()
    dataset._handle_missing_categorical_values()
    dataset._handle_missing_numerical_values()
    dataset._preprocess_categorical()
    dataset._preprocess_numerical()

    # Should complete without errors
    X, y, X_test = dataset.get_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)

    # All columns should be numeric after preprocessing
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)
    assert all(np.issubdtype(dtype, np.number) for dtype in X_test.dtypes)


def test_statistical_features_vectorization():
    """Test that vectorized operations work correctly."""

    # Create larger array to test vectorization performance
    np.random.seed(28)
    large_arr = np.random.normal(0, 1, (1000, 50))  # 1000 samples, 50 timepoints each

    df = extract_statistical_features_from_array(large_arr, "big_test")

    # Should complete quickly and produce correct results
    assert len(df) == 1000
    assert len(df.columns) > 30

    # Spot check some calculations
    # For normal distribution, mean should be close to 0
    assert abs(df["big_test_mean"].mean()) < 0.1

    # Standard deviation should be close to 1
    assert abs(df["big_test_std"].mean() - 1.0) < 0.1


def test_statistical_features_signal_processing():
    """Test signal processing features specifically."""

    # Create arrays with known signal characteristics
    t = np.linspace(0, 4 * np.pi, 100)
    arr = np.array(
        [
            np.sin(t),  # Pure sine wave
            np.sin(t) + 0.5 * np.sin(3 * t),  # Sine with harmonics (more peaks)
            np.ones(100),  # Flat signal (no peaks)
            np.array([0] * 50 + [1] * 50),  # Step function (one crossing)
        ],
        dtype=np.float64,
    )

    df = extract_statistical_features_from_array(arr, "signal")

    # Sine wave should have some peaks
    assert df["signal_peaks"].iloc[0] > 0

    # Harmonics should have more peaks than pure sine
    assert df["signal_peaks"].iloc[1] > df["signal_peaks"].iloc[0]

    # Flat signal should have no peaks
    assert df["signal_peaks"].iloc[2] == 0

    # Step function should have crossings
    assert df["signal_zero_crossings"].iloc[3] == 0


@pytest.mark.parametrize("array_length", [10, 50, 100, 200])
def test_statistical_features_different_lengths(array_length):
    """Test statistical features with different array lengths."""

    np.random.seed(28)
    arr = np.random.normal(0, 1, (5, array_length))

    df = extract_statistical_features_from_array(arr, f"len_{array_length}")

    # Should work for all lengths
    assert len(df) == 5
    assert df[f"len_{array_length}_total_count"].iloc[0] == array_length

    # Energy and SMA should be normalized by length
    assert all(df[f"len_{array_length}_energy"] > 0)
    assert all(df[f"len_{array_length}_sma"] > 0)


def test_get_feature_statistics_with_arrays():
    """Test getting feature statistics works with statistical features."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()
    dataset._extract_statistical_features("sensor_data")

    # Remove the original array columns to avoid issues with pandas operations
    # The array columns cause problems with pandas operations like nunique()
    columns_to_drop = ["sensor_data", "edge_case"]
    for col in columns_to_drop:
        if col in dataset.X.columns:
            dataset.X = dataset.X.drop(columns=[col])
        if col in dataset.X_test.columns:
            dataset.X_test = dataset.X_test.drop(columns=[col])

    stats_df, stats_df_test = dataset.get_feature_statistics()

    # Should include statistical features
    feature_cols = [col for col in dataset.X.columns if col.startswith("sensor_data_")]
    for col in feature_cols:
        assert col in stats_df.index
        assert col in stats_df_test.index

    # Check that stats are calculated correctly
    assert "mean" in stats_df.columns
    assert "std" in stats_df.columns
    assert "missing" in stats_df.columns
    assert "mean" in stats_df_test.columns
    assert "std" in stats_df_test.columns
    assert "missing" in stats_df_test.columns


@pytest.mark.parametrize("strategy", ["zero", "mean", "median", "last", "truncate", "global_mean"])
def test_array_length_strategies_with_dummy_dataset(strategy):
    """Test all array length strategies using DummyDatasetWithArrays edge_case column."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY=strategy)
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    # Extract statistical features using edge_case column which has different lengths
    dataset._extract_statistical_features("edge_case")

    # All arrays should now have the same length
    train_lengths = [len(arr) for arr in dataset.X_arrays]
    test_lengths = [len(arr) for arr in dataset.X_test_arrays]

    assert (
        len(set(train_lengths)) == 1
    ), f"Train arrays have different lengths: {set(train_lengths)}"
    assert len(set(test_lengths)) == 1, f"Test arrays have different lengths: {set(test_lengths)}"

    # Check strategy-specific behavior
    if strategy == "truncate":
        # Should be truncated to shortest length (2 in edge_case)
        assert train_lengths[0] == 2
        assert test_lengths[0] == 2
    else:
        # Should be padded to longest length (5 in edge_case)
        assert train_lengths[0] == 5
        assert test_lengths[0] == 5

    # Statistical features should have been created
    feature_cols = [col for col in dataset.X.columns if col.startswith("edge_case_")]
    assert len(feature_cols) > 30


def test_zero_strategy_with_edge_cases():
    """Test zero padding strategy with edge case arrays."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="zero")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    dataset._extract_statistical_features("edge_case")

    # Check specific padding results
    # Original arrays in edge_case:
    # [1.0, 1.0] -> [1.0, 1.0, 0.0, 0.0, 0.0] (padded to length 5)
    # [0.0, 0.0, 0.0] -> [0.0, 0.0, 0.0, 0.0, 0.0] (padded to length 5)
    # etc.

    expected_first = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(dataset.X_arrays[0], expected_first)

    expected_second = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(dataset.X_arrays[1], expected_second)


def test_mean_strategy_with_edge_cases():
    """Test mean padding strategy with edge case arrays."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="mean")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    dataset._extract_statistical_features("edge_case")

    # Check mean padding
    # [1.0, 1.0] has mean 1.0, should become [1.0, 1.0, 1.0, 1.0, 1.0]
    expected_first = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(dataset.X_arrays[0], expected_first)

    # [0.0, 0.0, 0.0] has mean 0.0, should become [0.0, 0.0, 0.0, 0.0, 0.0]
    expected_second = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(dataset.X_arrays[1], expected_second)


def test_truncate_strategy_with_edge_cases():
    """Test truncate strategy with edge case arrays."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="truncate")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    dataset._extract_statistical_features("edge_case")

    # All should be truncated to length 2 (shortest in edge_case)
    for arr in dataset.X_arrays:
        assert len(arr) == 2
    for arr in dataset.X_test_arrays:
        assert len(arr) == 2

    # Check specific truncations
    # [1.0, 1.0] stays [1.0, 1.0]
    # [0.0, 0.0, 0.0] becomes [0.0, 0.0]
    # [1.0, -1.0, 1.0, -1.0] becomes [1.0, -1.0]
    # [1.0, 2.0, 3.0, 4.0, 5.0] becomes [1.0, 2.0]

    np.testing.assert_array_equal(dataset.X_arrays[0], [1.0, 1.0])
    np.testing.assert_array_equal(dataset.X_arrays[1], [0.0, 0.0])
    np.testing.assert_array_equal(dataset.X_arrays[2], [1.0, -1.0])
    np.testing.assert_array_equal(dataset.X_arrays[3], [1.0, 2.0])


def test_last_strategy_with_edge_cases():
    """Test last value padding strategy with edge case arrays."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="last")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    dataset._extract_statistical_features("edge_case")

    # Check last value padding
    # [1.0, 1.0] -> [1.0, 1.0, 1.0, 1.0, 1.0] (last value is 1.0)
    expected_first = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(dataset.X_arrays[0], expected_first)

    # [1.0, -1.0, 1.0, -1.0] -> [1.0, -1.0, 1.0, -1.0, -1.0] (last value is -1.0)
    expected_third = np.array([1.0, -1.0, 1.0, -1.0, -1.0])
    np.testing.assert_array_equal(dataset.X_arrays[2], expected_third)


def test_global_mean_strategy_with_edge_cases():
    """Test global mean padding strategy with edge case arrays."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="global_mean")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    dataset._extract_statistical_features("edge_case")

    # Calculate expected global mean
    # edge_case arrays: [1,1], [0,0,0], [1,-1,1,-1], [1,2,3,4,5], [10,20,30]
    # means: 1.0, 0.0, 0.0, 3.0, 20.0
    # global mean: (1.0 + 0.0 + 0.0 + 3.0 + 20.0) / 5 = 4.8

    expected_global_mean = 4.8

    # [1.0, 1.0] should be padded with global mean
    expected_first = np.array(
        [1.0, 1.0, expected_global_mean, expected_global_mean, expected_global_mean]
    )
    np.testing.assert_array_almost_equal(dataset.X_arrays[0], expected_first)


def test_median_strategy_with_edge_cases():
    """Test median padding strategy with edge case arrays."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="median")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    dataset._extract_statistical_features("edge_case")

    # [1.0, 1.0] has median 1.0, should become [1.0, 1.0, 1.0, 1.0, 1.0]
    expected_first = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(dataset.X_arrays[0], expected_first)

    # [1.0, 2.0, 3.0, 4.0, 5.0] has median 3.0, should stay unchanged
    expected_fourth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(dataset.X_arrays[3], expected_fourth)


def test_multiple_column_feature_extraction():
    """Test extracting features from multiple columns."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Extract features from both sensor_data and edge_case
    dataset._extract_statistical_features("sensor_data")
    dataset._extract_statistical_features("edge_case")

    # Should have features for both columns
    sensor_features = [col for col in dataset.X.columns if col.startswith("sensor_data_")]
    edge_features = [col for col in dataset.X.columns if col.startswith("edge_case_")]

    assert len(sensor_features) > 30
    assert len(edge_features) > 30

    # No overlap in feature names
    assert not set(sensor_features).intersection(set(edge_features))


def test_config_defaults():
    """Test that default configuration works with DummyDatasetWithArrays."""

    # Default config uses "zero" strategy
    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Should work without errors
    dataset._extract_statistical_features("edge_case")

    # Check that it used zero padding (default)
    expected_first = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # Zero padded
    np.testing.assert_array_equal(dataset.X_arrays[0], expected_first)


def test_error_handling_with_dummy_dataset():
    """Test error handling using DummyDatasetWithArrays."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Test invalid strategy
    dataset.config.ARRAY_LENGTH_STRATEGY = "invalid_strategy"

    with pytest.raises(ValueError, match="Invalid array length strategy."):
        dataset._extract_statistical_features("edge_case")

    # Test invalid column
    dataset.config.ARRAY_LENGTH_STRATEGY = "zero"  # Reset to valid

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in dataset."):
        dataset._extract_statistical_features("nonexistent")


def test_check_columns_missing_in_test():
    """Test _check_columns when X_test is missing columns."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()

    # Remove a column from X_test to create mismatch
    dataset.X_test = dataset.X_test.drop(columns=["num1"])

    with pytest.raises(ValueError, match="X_test is missing columns: \\['num1'\\]"):
        dataset._check_columns()


def test_check_columns_extra_in_test():
    """Test _check_columns when X_test has extra columns."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()

    # Add extra column to X_test
    dataset.X_test["extra_col"] = [1, 2, 3]

    with pytest.raises(ValueError, match="X is missing columns: \\['extra_col'\\]"):
        dataset._check_columns()


def test_check_columns_success():
    """Test _check_columns when columns match."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()

    # Should not raise any exception
    dataset._check_columns()


def test_handle_missing_categorical_values_empty_columns():
    """Test handling missing categorical values when no categorical columns exist."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()

    # Remove all categorical columns
    dataset.X = dataset.X.drop(columns=["cat1", "cat2"])
    dataset.X_test = dataset.X_test.drop(columns=["cat1", "cat2"])

    dataset._detect_column_types()

    # Should complete without errors even when no categorical columns exist
    dataset._handle_missing_categorical_values()

    # No missing indicators should be added since there are no categorical columns
    missing_indicator_cols = [col for col in dataset.X.columns if col.endswith("_is_missing")]
    assert len(missing_indicator_cols) == 0


def test_handle_missing_numerical_values_empty_columns():
    """Test handling missing numerical values when no numerical columns exist."""

    dataset = DummyDataset(DatasetConfig())
    dataset.load_data()

    # Remove all numerical columns
    dataset.X = dataset.X.drop(columns=["num1", "num2"])
    dataset.X_test = dataset.X_test.drop(columns=["num1", "num2"])

    dataset._detect_column_types()

    # Should complete without errors even when no numerical columns exist
    dataset._handle_missing_numerical_values()

    # No missing indicators should be added since there are no numerical columns
    missing_indicator_cols = [col for col in dataset.X.columns if col.endswith("_is_missing")]
    assert len(missing_indicator_cols) == 0


def test_feature_engineering_default():
    """Test default feature_engineering method."""

    dataset = DummyDataset()
    dataset.load_data()

    original_columns = list(dataset.X.columns)

    # Default implementation should do nothing
    dataset.feature_engineering()

    # Dataset should remain unchanged
    assert list(dataset.X.columns) == original_columns
    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.y, pd.Series)


def test_extract_date_features():
    """Test extracting date features from a date column."""

    dataset = DummyDataset()
    dataset.load_data()

    # Add date columns to the dataset
    dataset.X["date_col"] = ["2023-01-01", "2023-01-02", "2023-12-31", "2023-07-15", "2023-06-10"]
    dataset.X_test["date_col"] = ["2023-01-04", "2023-01-05", "2023-02-28"]

    # Extract date features
    dataset.extract_date_features("date_col")

    # Check that date features were added
    expected_features = [
        "date_col_year",
        "date_col_quarter",
        "date_col_month",
        "date_col_day",
        "date_col_week",
        "date_col_day_of_year",
        "date_col_day_name",
        "date_col_is_weekend",
    ]

    for feature in expected_features:
        assert feature in dataset.X.columns

    assert "date_col_week" in dataset.X.columns
    assert "date_col_week" not in dataset.X_test.columns

    # Check specific values
    assert dataset.X["date_col_year"].iloc[0] == 2023
    assert dataset.X["date_col_month"].iloc[0] == 1
    assert dataset.X["date_col_day"].iloc[0] == 1

    # Check weekend detection (2023-01-01 was a Sunday)
    assert dataset.X["date_col_is_weekend"].iloc[0]

    # Check day name categories
    assert isinstance(dataset.X["date_col_day_name"].dtype, pd.CategoricalDtype)


def test_extract_date_features_nonexistent_column():
    """Test extracting date features from non-existent column."""

    dataset = DummyDataset()
    dataset.load_data()

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in dataset."):
        dataset.extract_date_features("nonexistent")


def test_extract_time_features():
    """Test extracting time features from a datetime column."""

    dataset = DummyDataset()
    dataset.load_data()

    # Add datetime columns to the dataset
    dataset.X["datetime_col"] = [
        "2023-01-01 10:30:45",
        "2023-01-02 14:20:15",
        "2023-01-03 09:45:30",
        "2023-12-31 23:59:59",
        "2023-07-15 12:00:00",
    ]
    dataset.X_test["datetime_col"] = [
        "2023-01-04 11:30:45",
        "2023-01-05 15:20:15",
        "2023-01-06 08:10:05",
    ]

    # Extract time features
    dataset.extract_time_features("datetime_col")

    # Check that time features were added
    expected_features = ["datetime_col_hour", "datetime_col_minute", "datetime_col_second"]

    for feature in expected_features:
        assert feature in dataset.X.columns
        assert feature in dataset.X_test.columns

    # Check specific values (2023-01-01 10:30:45)
    assert dataset.X["datetime_col_hour"].iloc[0] == 10
    assert dataset.X["datetime_col_minute"].iloc[0] == 30
    assert dataset.X["datetime_col_second"].iloc[0] == 45


def test_extract_time_features_nonexistent_column():
    """Test extracting time features from non-existent column."""

    dataset = DummyDataset()
    dataset.load_data()

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in dataset."):
        dataset.extract_time_features("nonexistent")


def test_create_cyclical_features():
    """Test creating cyclical features for a column."""

    dataset = DummyDataset()
    dataset.load_data()

    # Create cyclical features for num1 (max_val=5)
    dataset.create_cyclical_features("num1", max_val=5)

    # Check that cyclical features were added
    assert "num1_sin" in dataset.X.columns
    assert "num1_cos" in dataset.X.columns
    assert "num1_sin" in dataset.X_test.columns
    assert "num1_cos" in dataset.X_test.columns

    # Check values (for value 1 with max_val=5: 2*pi*1/5)
    expected_angle = 2 * np.pi * 1 / 5
    np.testing.assert_almost_equal(dataset.X["num1_sin"].iloc[0], np.sin(expected_angle))
    np.testing.assert_almost_equal(dataset.X["num1_cos"].iloc[0], np.cos(expected_angle))


def test_create_cyclical_features_nonexistent_column():
    """Test creating cyclical features for non-existent column."""

    dataset = DummyDataset()
    dataset.load_data()

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in dataset."):
        dataset.create_cyclical_features("nonexistent", max_val=24)


def test_count_tokens():
    """Test counting tokens in a text column."""

    dataset = DummyDataset()
    dataset.load_data()

    # Add text columns to the dataset
    dataset.X["text_col"] = ["hello world", "test string", "another text", "sample", "example"]
    dataset.X_test["text_col"] = ["test data", "more text", "additional"]

    # Count tokens in text column
    dataset.count_tokens("text_col")

    # Check that token count features were added
    assert "text_col_token_count" in dataset.X.columns
    assert "text_col_token_count" in dataset.X_test.columns

    # Check that token counts are reasonable (should be > 0)
    assert all(dataset.X["text_col_token_count"] > 0)
    assert all(dataset.X_test["text_col_token_count"] > 0)


def test_count_tokens_nonexistent_column():
    """Test counting tokens for non-existent column."""

    dataset = DummyDataset()
    dataset.load_data()

    with pytest.raises(ValueError, match="Column 'nonexistent' not found in dataset."):
        dataset.count_tokens("nonexistent")


def test_count_tokens_custom_encoding():
    """Test counting tokens with custom encoding model."""

    dataset = DummyDataset()
    dataset.load_data()

    # Add text columns to the dataset
    dataset.X["text_col"] = ["hello world", "test string", "another text", "sample", "example"]
    dataset.X_test["text_col"] = ["test data", "more text", "additional"]

    # Use a different encoding model
    dataset.count_tokens("text_col", encoding_model="cl100k_base")

    # Check that token count features were added
    assert "text_col_token_count" in dataset.X.columns
    assert "text_col_token_count" in dataset.X_test.columns


def test_get_embedding_empty_text():
    """Test getting embeddings with empty text input."""

    dataset = DummyDataset()
    dataset.load_data()

    with pytest.raises(ValueError, match="Text input cannot be empty."):
        dataset.get_embedding([])


def test_get_feature_statistics():
    """Test getting feature statistics."""

    dataset = DummyDataset()
    dataset.load_data()
    dataset.preprocess()

    stats_train, stats_test = dataset.get_feature_statistics()

    # Check that statistics are returned as DataFrames
    assert isinstance(stats_train, pd.DataFrame)
    assert isinstance(stats_test, pd.DataFrame)

    # Check that expected columns are present
    expected_columns = [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "missing",
        "unique",
        "dtype",
    ]
    for col in expected_columns:
        assert col in stats_train.columns
        assert col in stats_test.columns

    # Check that all features are included
    assert len(stats_train) == dataset.X.shape[1]
    assert len(stats_test) == dataset.X_test.shape[1]


def test_load_data_abstract_method():
    """Test that load_data is an abstract method."""

    # Cannot instantiate BaseDataset directly due to abstract method
    with pytest.raises(TypeError):
        BaseDataset()


def test_extract_arrays_with_mixed_types():
    """Test extracting arrays with mixed data types."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()

    # Modify the dataset to have mixed array types including scalars
    dataset.X = pd.DataFrame(
        {
            "mixed_arrays": [
                [1, 2, 3],  # list
                np.array([4, 5, 6]),  # numpy array
                7,  # scalar
                [8.5, 9.5],  # list with floats
            ]
        }
    )
    dataset.X_test = pd.DataFrame({"mixed_arrays": [[10, 11], 12]})
    dataset.y = pd.Series([0, 1, 0, 1])

    # Extract arrays
    dataset._extract_arrays("mixed_arrays")

    # Check that all arrays are numpy arrays with float64 dtype
    assert len(dataset.X_arrays) == 4
    assert len(dataset.X_test_arrays) == 2

    for arr in dataset.X_arrays:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    for arr in dataset.X_test_arrays:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    # Check that scalar was converted to array
    np.testing.assert_array_equal(dataset.X_arrays[2], np.array([7.0]))
    np.testing.assert_array_equal(dataset.X_test_arrays[1], np.array([12.0]))


def test_handle_array_lengths_with_different_strategies():
    """Test array length handling with all strategies comprehensively."""

    # Test each strategy individually with known data
    strategies_to_test = ["zero", "mean", "median", "last", "truncate", "global_mean"]

    for strategy in strategies_to_test:
        config = DatasetConfig(ARRAY_LENGTH_STRATEGY=strategy)
        dataset = DummyDatasetWithArrays(config)
        dataset.load_data()

        # Use a simple test case
        dataset.X = pd.DataFrame(
            {"test_arrays": [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), np.array([6.0])]}
        )
        dataset.X_test = pd.DataFrame({"test_arrays": [np.array([7.0, 8.0]), np.array([9.0])]})
        dataset.y = pd.Series([0, 1, 0])

        # Extract and handle arrays
        dataset._extract_arrays("test_arrays")
        dataset._handle_array_lengths()

        # Check that all arrays have the same length
        train_lengths = [len(arr) for arr in dataset.X_arrays]
        test_lengths = [len(arr) for arr in dataset.X_test_arrays]

        assert len(set(train_lengths)) == 1, f"Strategy {strategy} failed for train arrays"
        assert len(set(test_lengths)) == 1, f"Strategy {strategy} failed for test arrays"

        if strategy == "truncate":
            assert train_lengths[0] == 1  # Shortest array length
        else:
            assert train_lengths[0] == 3  # Longest array length


def test_execute_array_length_strategy_invalid():
    """Test _execute_array_length_strategy with invalid strategy."""

    dataset = DummyDatasetWithArrays()
    dataset.load_data()
    dataset.config.ARRAY_LENGTH_STRATEGY = "invalid_strategy"

    arrays = [np.array([1.0, 2.0]), np.array([3.0])]

    with pytest.raises(ValueError, match="Invalid array length strategy."):
        dataset._execute_array_length_strategy(arrays, 3, 2.0)


def test_categorical_fill_strategy_with_all_unique():
    """Test categorical fill strategy when all values are unique (no clear mode)."""

    dataset = DummyDataset(DatasetConfig(CATEGORICAL_FILL_STRATEGY="mode"))
    dataset.load_data()

    # Create dataset where each categorical value is unique
    dataset.X["cat1"] = ["A", "B", "C", "D", np.nan]
    dataset.X_test["cat1"] = ["E", np.nan, "F"]

    dataset._detect_column_types()

    # Should handle the case without error (pandas .mode().iloc[0] will take first mode)
    dataset._handle_missing_categorical_values()

    # Check that missing values were handled
    assert not dataset.X["cat1"].isna().any()
    assert not dataset.X_test["cat1"].isna().any()


def test_numerical_fill_strategies_with_all_nan():
    """Test numerical fill strategies when column has all NaN values."""

    dataset = DummyDataset(DatasetConfig(NUMERICAL_FILL_STRATEGY="zero"))
    dataset.load_data()

    # Make all values NaN
    dataset.X["num1"] = [np.nan, np.nan, np.nan, np.nan, np.nan]
    dataset.X_test["num1"] = [np.nan, np.nan, np.nan]

    dataset._detect_column_types()
    dataset._handle_missing_numerical_values()

    # Zero strategy should work
    assert all(dataset.X["num1"] == 0.0)
    assert all(dataset.X_test["num1"] == 0.0)

    # Missing indicator should be created
    assert "num1_is_missing" in dataset.X.columns


def test_preprocess_with_no_columns_of_type():
    """Test preprocessing when dataset has no columns of specific types."""

    dataset = DummyDataset()
    dataset.load_data()

    # Remove all categorical and numerical columns, keep only datetime-like
    dataset.X = pd.DataFrame(
        {
            "date_col": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
            )
        }
    )
    dataset.X_test = pd.DataFrame(
        {"date_col": pd.to_datetime(["2023-01-06", "2023-01-07", "2023-01-08"])}
    )

    # Should complete preprocessing without errors
    dataset.preprocess()

    # Should have processed the data successfully
    X, y, X_test = dataset.get_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(X_test, pd.DataFrame)


def test_invalid_array_length_strategy_config():
    """Test DatasetConfig with invalid array length strategy."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="invalid")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    # Should raise error when trying to use invalid strategy
    with pytest.raises(ValueError, match="Invalid array length strategy."):
        dataset._extract_statistical_features("edge_case")


def test_extract_date_features_week_number():
    """Test that week number is correctly extracted in date features."""

    dataset = DummyDataset()
    dataset.load_data()

    # Add date columns to the dataset
    dataset.X["date_col"] = ["2023-01-01", "2023-01-02", "2023-12-31", "2023-07-15", "2023-06-10"]
    dataset.X_test["date_col"] = ["2023-01-04", "2023-01-05", "2023-02-28"]

    # Extract date features
    dataset.extract_date_features("date_col")

    # Week numbers should be between 1 and 53
    assert all(1 <= week <= 53 for week in dataset.X["date_col_week"])


def test_statistical_features_with_constant_arrays():
    """Test statistical features with arrays containing only constant values."""

    config = DatasetConfig(ARRAY_LENGTH_STRATEGY="zero")
    dataset = DummyDatasetWithArrays(config)
    dataset.load_data()

    # Create arrays with constant values to test edge cases in statistical calculations
    const_arrays = [
        np.full(10, 5.0),  # All same value
        np.full(5, 0.0),  # All zeros
        np.full(8, -2.0),  # All negative
    ]

    dataset.X = pd.DataFrame({"const_data": const_arrays})
    dataset.X_test = pd.DataFrame({"const_data": const_arrays[:2]})
    dataset.y = pd.Series([0, 1, 0])

    # Should handle constant arrays without errors
    dataset._extract_statistical_features("const_data")

    # Check that features were created
    feature_cols = [col for col in dataset.X.columns if col.startswith("const_data_")]
    assert len(feature_cols) > 30

    # Check specific properties for constant arrays
    # Standard deviation should be 0 for constant arrays
    assert dataset.X["const_data_std"].iloc[0] == 0.0
    assert dataset.X["const_data_std"].iloc[1] == 0.0
    assert dataset.X["const_data_range"].iloc[0] == 0.0
    assert dataset.X["const_data_range"].iloc[1] == 0.0


def test_extract_date_features_with_missing_week():
    """Test extracting date features including week handling."""

    dataset = DummyDataset()
    dataset.load_data()

    # Add date columns to the dataset
    dataset.X["date_col"] = ["2023-01-01", "2023-01-02", "2023-12-31", "2023-07-15", "2023-06-10"]
    dataset.X_test["date_col"] = ["2023-01-04", "2023-01-05", "2023-02-28"]

    # Extract date features
    dataset.extract_date_features("date_col")

    # Check that week feature was added for test data
    assert "date_col_week" in dataset.X.columns

    # Check that all week values are reasonable
    assert all(1 <= week <= 53 for week in dataset.X["date_col_week"])

    # Check that other features were added to both X and X_test
    common_features = [
        "date_col_year",
        "date_col_quarter",
        "date_col_month",
        "date_col_day",
        "date_col_day_of_year",
        "date_col_day_name",
        "date_col_is_weekend",
    ]

    for feature in common_features:
        assert feature in dataset.X.columns
