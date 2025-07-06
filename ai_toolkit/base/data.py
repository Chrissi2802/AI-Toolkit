import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from scipy import signal, stats
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from ai_toolkit.utils.logging import get_logger


load_dotenv()


@dataclass
class DatasetConfig:
    """Configuration for data loading and preprocessing."""

    CATEGORICAL_FILL_STRATEGY: str = field(
        default="mode",
        metadata={
            "description": "Strategy to fill missing categorical values. Options: "
            "'mode' (fill with mode), "
            "'missing' (fill with 'MISSING')"
        },
    )
    NUMERICAL_FILL_STRATEGY: str = field(
        default="median",
        metadata={
            "description": "Strategy to fill missing numerical values. Options: "
            "'median' (fill with median), "
            "'mean' (fill with mean), "
            "'zero' (fill with zero)"
        },
    )
    CATEGORICAL_PREPROCESSING_STRATEGY: str = field(
        default="OneHotEncoder",
        metadata={
            "description": "Strategy to preprocess categorical columns. Options: "
            "'LabelEncoder' (apply label encoding), "
            "'OneHotEncoder' (apply one-hot encoding), "
            "'all' (apply both label and one-hot encoding)"
        },
    )
    NUMERICAL_PREPROCESSING_STRATEGY: str = field(
        default="StandardScaler",
        metadata={
            "description": "Strategy to preprocess numerical columns. Options: "
            "'StandardScaler' (apply standard scaling), "
            "'RobustScaler' (apply robust scaling), "
            "'MinMaxScaler' (apply min-max scaling)"
        },
    )
    ARRAY_LENGTH_STRATEGY: str = field(
        default="zero",
        metadata={
            "description": "Strategy to handle arrays of different lengths. Options: "
            "'zero' (pad with zeros), "
            "'mean' (pad with array mean), "
            "'median' (pad with array median), "
            "'last' (repeat last value), "
            "'truncate' (cut to shortest length), "
            "'global_mean' (pad with mean of all arrays)"
        },
    )


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    def __init__(self, config: DatasetConfig = DatasetConfig()) -> None:
        """Initialize the base dataset.

        Args:
            config (DatasetConfig, optional):
                Configuration for data loading and preprocessing. Defaults to DatasetConfig.
        """

        self.config = config
        self.X = None
        self.X_test = None
        self.y = None

        # Initialize logger
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing dataset", config=self.config.__dict__)

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset."""
        pass

    def preprocess(self) -> None:
        """Preprocess the dataset."""

        try:
            self._check_data()
            self._check_columns()
            self._detect_column_types()
            self._handle_missing_categorical_values()
            self._handle_missing_numerical_values()
            self._preprocess_categorical()
            self._preprocess_numerical()

            self.logger.info(
                "Data preprocessing completed",
                n_samples=self.X.shape[0],
                n_features=self.X.shape[1],
                categorical_columns=list(self.categorical_columns),
                numerical_columns=list(self.numerical_columns),
                n_classes=self.y.nunique(),
            )

        except Exception as e:
            self.logger.error("Data preprocessing failed", error=e)
            raise RuntimeError("Data preprocessing failed") from e

    def _check_data(self) -> None:
        """Check if the data is loaded."""

        self.logger.debug("Validating data")

        try:
            if self.X is None:
                raise ValueError("X data not loaded.")
            elif self.y is None:
                raise ValueError("y data not loaded.")
            elif self.X_test is None:
                raise ValueError("X_test data not loaded.")

            self.logger.info(
                "Data validation successful",
                X_shape=self.X.shape,
                y_shape=self.y.shape,
                X_test_shape=self.X_test.shape,
            )

        except Exception as e:
            self.logger.error("Data validation failed", error=e)
            raise ValueError("Data validation failed") from e

    def _check_columns(self) -> None:
        """Check if the columns in X and X_test match."""

        self.logger.debug("Checking dataset columns")

        X_cols = self.X.columns
        X_test_cols = self.X_test.columns

        # Check if all columns in X are present in X_test
        missing_in_test = X_cols[~X_cols.isin(X_test_cols)]
        if not missing_in_test.empty:
            raise ValueError(f"X_test is missing columns: {missing_in_test.tolist()}")

        # Check if all columns in X_test are present in X
        missing_in_train = X_test_cols[~X_test_cols.isin(X_cols)]
        if not missing_in_train.empty:
            raise ValueError(f"X is missing columns: {missing_in_train.tolist()}")

    def _detect_column_types(self) -> None:
        """Detect column types in the dataset."""

        self.logger.debug("Detecting column types")

        # Categorical columns
        self.categorical_columns = self.X.select_dtypes(include=["object", "category"]).columns

        # Numerical columns
        self.numerical_columns = self.X.select_dtypes(include=["int64", "float64"]).columns

    def _handle_missing_categorical_values(self) -> None:
        """Handle missing categorical values in the dataset."""

        self.logger.debug("Handling missing categorical values")

        # Check if categorical columns exist
        if self.categorical_columns.empty:
            return

        # Add missing indicator
        for col in self.categorical_columns:
            if self.X[col].isna().any():
                self.X[f"{col}_is_missing"] = self.X[col].isna().astype(int)
                self.X_test[f"{col}_is_missing"] = self.X_test[col].isna().astype(int)

        if self.config.CATEGORICAL_FILL_STRATEGY == "mode":
            self.X[self.categorical_columns] = self.X[self.categorical_columns].fillna(
                self.X[self.categorical_columns].mode().iloc[0]
            )
            self.X_test[self.categorical_columns] = self.X_test[self.categorical_columns].fillna(
                self.X_test[self.categorical_columns].mode().iloc[0]
            )
        elif self.config.CATEGORICAL_FILL_STRATEGY == "missing":
            self.X[self.categorical_columns] = self.X[self.categorical_columns].fillna("MISSING")
            self.X_test[self.categorical_columns] = self.X_test[self.categorical_columns].fillna(
                "MISSING"
            )
        else:
            raise ValueError("Invalid categorical fill strategy.")

    def _handle_missing_numerical_values(self) -> None:
        """Handle missing numerical values in the dataset."""

        self.logger.debug("Handling missing numerical values")

        # Check if numerical columns exist
        if self.numerical_columns.empty:
            return

        # Add missing indicator
        for col in self.numerical_columns:
            if self.X[col].isna().any():
                self.X[f"{col}_is_missing"] = self.X[col].isna().astype(int)
                self.X_test[f"{col}_is_missing"] = self.X_test[col].isna().astype(int)

        if self.config.NUMERICAL_FILL_STRATEGY == "median":
            self.X[self.numerical_columns] = self.X[self.numerical_columns].fillna(
                self.X[self.numerical_columns].median()
            )
            self.X_test[self.numerical_columns] = self.X_test[self.numerical_columns].fillna(
                self.X_test[self.numerical_columns].median()
            )
        elif self.config.NUMERICAL_FILL_STRATEGY == "mean":
            self.X[self.numerical_columns] = self.X[self.numerical_columns].fillna(
                self.X[self.numerical_columns].mean()
            )
            self.X_test[self.numerical_columns] = self.X_test[self.numerical_columns].fillna(
                self.X_test[self.numerical_columns].mean()
            )
        elif self.config.NUMERICAL_FILL_STRATEGY == "zero":
            self.X[self.numerical_columns] = self.X[self.numerical_columns].fillna(0.0)
            self.X_test[self.numerical_columns] = self.X_test[self.numerical_columns].fillna(0.0)
        else:
            raise ValueError("Invalid numerical fill strategy.")

    def _label_encode(self) -> None:
        """Label encode categorical columns in the dataset."""

        self.logger.debug("Label encoding of categorical features")

        self.label_encoders = {}

        # Encode each categorical column
        for col in self.categorical_columns:
            unique_values = pd.concat([self.X[col], self.X_test[col]]).unique()

            encoder = LabelEncoder()
            encoder.fit(unique_values)
            self.X[col] = encoder.transform(self.X[col])
            self.X_test[col] = encoder.transform(self.X_test[col])
            self.label_encoders[col] = encoder

    def _one_hot_encode(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """One hot encode categorical columns in the dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple of one hot encoded features and test data.
        """

        self.logger.debug("One hot encoding of categorical features")

        # Combine training and test data
        combined_data = pd.concat(
            [self.X[self.categorical_columns], self.X_test[self.categorical_columns]]
        )

        # Create dummy variables
        dummies = pd.get_dummies(combined_data, prefix=self.categorical_columns)

        # Split training and test data
        n_train = self.X.shape[0]
        df_encoded = dummies[:n_train]
        df_encoded_test = dummies[n_train:]

        return df_encoded, df_encoded_test

    def _preprocess_categorical(self) -> None:
        """Preprocess categorical columns in the dataset."""

        self.logger.debug("Preprocessing categorical features")

        # Check if categorical columns exist
        if self.categorical_columns.empty:
            return

        if self.config.CATEGORICAL_PREPROCESSING_STRATEGY == "LabelEncoder":
            self._label_encode()

        elif self.config.CATEGORICAL_PREPROCESSING_STRATEGY == "OneHotEncoder":
            df_encoded, df_encoded_test = self._one_hot_encode()

            self.X = pd.concat([self.X.drop(columns=self.categorical_columns), df_encoded], axis=1)
            self.X_test = pd.concat(
                [self.X_test.drop(columns=self.categorical_columns), df_encoded_test],
                axis=1,
            )
        elif self.config.CATEGORICAL_PREPROCESSING_STRATEGY == "all":
            df_encoded, df_encoded_test = self._one_hot_encode()
            self._label_encode()

            self.X = pd.concat([self.X, df_encoded], axis=1)
            self.X_test = pd.concat([self.X_test, df_encoded_test], axis=1)
        else:
            raise ValueError("Invalid categorical preprocessing strategy.")

    def _preprocess_numerical(self) -> None:
        """Preprocess numerical columns in the dataset."""

        self.logger.debug("Preprocessing numerical features")

        # Standardize numerical columns
        if self.config.NUMERICAL_PREPROCESSING_STRATEGY == "StandardScaler":
            scaler = StandardScaler()
        elif self.config.NUMERICAL_PREPROCESSING_STRATEGY == "RobustScaler":
            scaler = RobustScaler()
        elif self.config.NUMERICAL_PREPROCESSING_STRATEGY == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid numerical preprocessing strategy.")

        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X.columns)

    def get_feature_types(self) -> Tuple[List[str], List[str]]:
        """Get feature types.

        Returns:
            Tuple[List[str], List[str]]: Tuple of categorical and numerical columns.
        """

        return self.categorical_columns, self.numerical_columns

    def get_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Get the dataset.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: Tuple of features, target and test data.
        """

        return self.X, self.y, self.X_test

    def get_feature_statistics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get statistical information about features.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple of training and test feature statistics.
        """

        df_stats = self.X.describe().T
        df_stats["missing"] = self.X.isna().sum()
        df_stats["unique"] = self.X.nunique()
        df_stats["dtype"] = self.X.dtypes

        df_stats_test = self.X_test.describe().T
        df_stats_test["missing"] = self.X_test.isna().sum()
        df_stats_test["unique"] = self.X_test.nunique()
        df_stats_test["dtype"] = self.X_test.dtypes

        return df_stats, df_stats_test

    def feature_engineering(self) -> None:
        """Perform feature engineering on the dataset."""
        pass

    def extract_date_features(self, date_column: str) -> None:
        """Extract date features from a date column.

        Args:
            date_column (str): The name of the date column to extract features from.
        """

        self.logger.debug("Extracting date features")

        if date_column not in self.X.columns:
            raise ValueError(f"Column '{date_column}' not found in dataset.")

        self.X[date_column] = pd.to_datetime(self.X[date_column], errors="coerce")
        self.X_test[date_column] = pd.to_datetime(self.X_test[date_column], errors="coerce")

        # Extract date features
        self.X[f"{date_column}_year"] = self.X[date_column].dt.year
        self.X[f"{date_column}_quarter"] = self.X[date_column].dt.quarter
        self.X[f"{date_column}_month"] = self.X[date_column].dt.month
        self.X[f"{date_column}_day"] = self.X[date_column].dt.day
        self.X[f"{date_column}_week"] = self.X[date_column].dt.isocalendar().week
        self.X[f"{date_column}_day_of_year"] = self.X[date_column].dt.dayofyear
        self.X[f"{date_column}_day_name"] = self.X[date_column].dt.day_name()
        self.X[f"{date_column}_is_weekend"] = self.X[date_column].dt.dayofweek >= 5

        self.X_test[f"{date_column}_year"] = self.X_test[date_column].dt.year
        self.X_test[f"{date_column}_quarter"] = self.X_test[date_column].dt.quarter
        self.X_test[f"{date_column}_month"] = self.X_test[date_column].dt.month
        self.X_test[f"{date_column}_day"] = self.X_test[date_column].dt.day
        self.X_test[f"{date_column}_day_of_year"] = self.X_test[date_column].dt.dayofyear
        self.X_test[f"{date_column}_day_name"] = self.X_test[date_column].dt.day_name()
        self.X_test[f"{date_column}_is_weekend"] = self.X_test[date_column].dt.dayofweek >= 5

        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        self.X[f"{date_column}_day_name"] = pd.Categorical(
            self.X[f"{date_column}_day_name"], categories=order, ordered=True
        )
        self.X_test[f"{date_column}_day_name"] = pd.Categorical(
            self.X_test[f"{date_column}_day_name"], categories=order, ordered=True
        )

    def extract_time_features(self, time_column: str) -> None:
        """Extract time features from a time column.
        Args:
            time_column (str): The name of the time column to extract features from.
        """

        self.logger.debug("Extracting time features")

        if time_column not in self.X.columns:
            raise ValueError(f"Column '{time_column}' not found in dataset.")

        self.X[time_column] = pd.to_datetime(self.X[time_column], errors="coerce")
        self.X_test[time_column] = pd.to_datetime(self.X_test[time_column], errors="coerce")

        # Extract time features
        self.X[f"{time_column}_hour"] = self.X[time_column].dt.hour
        self.X[f"{time_column}_minute"] = self.X[time_column].dt.minute
        self.X[f"{time_column}_second"] = self.X[time_column].dt.second

        self.X_test[f"{time_column}_hour"] = self.X_test[time_column].dt.hour
        self.X_test[f"{time_column}_minute"] = self.X_test[time_column].dt.minute
        self.X_test[f"{time_column}_second"] = self.X_test[time_column].dt.second

    def create_cyclical_features(self, column: str, max_val: int) -> None:
        """Creates sine and cosine features for cyclical data.

        Args:
            column (str): Name of the column to transform.
            max_val (int): Maximum value of the cycle (e.g., 24 for hours, 7 for days of week)
        """

        self.logger.debug("Creating cyclical features")

        if column not in self.X.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        self.X[f"{column}_sin"] = np.sin(2 * np.pi * self.X[column] / max_val)
        self.X[f"{column}_cos"] = np.cos(2 * np.pi * self.X[column] / max_val)

        self.X_test[f"{column}_sin"] = np.sin(2 * np.pi * self.X_test[column] / max_val)
        self.X_test[f"{column}_cos"] = np.cos(2 * np.pi * self.X_test[column] / max_val)

    def count_tokens(self, text_column: str, encoding_model: str = "o200k_base") -> None:
        """Count tokens in a text column.

        Args:
            text_column (str): The name of the text column to count tokens in.
            encoding_model (str, optional):
                The encoding model to use. Defaults to "o200k_base" (gpt-4o).

            Find the right encoding model
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """

        self.logger.debug("Counting tokens")

        if text_column not in self.X.columns:
            raise ValueError(f"Column '{text_column}' not found in dataset.")

        encoding = tiktoken.get_encoding(encoding_model)

        self.X[f"{text_column}_token_count"] = self.X[text_column].apply(
            lambda x: len(encoding.encode(x))
        )
        self.X_test[f"{text_column}_token_count"] = self.X_test[text_column].apply(
            lambda x: len(encoding.encode(x))
        )

    def get_embedding(self, text: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
        """Get embeddings for a list of text strings using OpenAI API.

        Args:
            text (List[str]): The text strings to embed.
            model (str, optional): The model to use for embedding.
                Defaults to "text-embedding-3-large".

        Returns:
            np.ndarray: The embeddings as a NumPy array.
        """

        self.logger.debug("Getting embeddings")

        if not text:
            raise ValueError("Text input cannot be empty.")

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

        openai_embedding = client.embeddings.create(
            input=text,
            model=model,
        )

        embedding = np.array([embedding.embedding for embedding in openai_embedding.data])

        return embedding

    def _extract_arrays(self, column: str) -> None:
        """Extract arrays from a column in the dataset.

        Args:
            column (str): The name of the column to extract arrays from.
        """

        self.logger.debug("Extracting arrays from column", column=column)

        # Extract arrays from the training data
        self.X_arrays = []
        for item in self.X[column]:
            if isinstance(item, (list, np.ndarray)):
                self.X_arrays.append(np.array(item, dtype=np.float64))
            else:
                self.X_arrays.append(np.array([float(item)], dtype=np.float64))

        # Extract arrays from test data
        self.X_test_arrays = []
        for item in self.X_test[column]:
            if isinstance(item, (list, np.ndarray)):
                self.X_test_arrays.append(np.array(item, dtype=np.float64))
            else:
                self.X_test_arrays.append(np.array([float(item)], dtype=np.float64))

    def _handle_array_lengths(self) -> None:
        """Handle arrays of different lengths in the dataset."""

        self.logger.debug("Handling arrays of different lengths")

        # Get the minimum length of all arrays and calculate global mean
        lengths = [len(arr) for arr in (*self.X_arrays, *self.X_test_arrays)]
        global_mean = np.mean([np.mean(arr) for arr in self.X_arrays])
        global_mean_test = np.mean([np.mean(arr) for arr in self.X_test_arrays])

        if self.config.ARRAY_LENGTH_STRATEGY == "truncate":
            target_length = min(lengths)
        else:
            target_length = max(lengths)

        self.X_arrays = self._execute_array_length_strategy(
            self.X_arrays, target_length, global_mean
        )
        self.X_test_arrays = self._execute_array_length_strategy(
            self.X_test_arrays, target_length, global_mean_test
        )

    def _execute_array_length_strategy(
        self, arrays: np.ndarray, target_length: int, global_mean: float
    ) -> np.ndarray:
        """Execute the array length strategy to ensure all arrays have the same length.

        Args:
            arrays (np.ndarray): The array to handle.
            target_length (int): The target length for padding/truncating.
            global_mean (float): The global mean value for padding.

        Returns:
            arrays (np.ndarray): The processed array with uniform length.
        """

        # Execute the array length strategy
        for i, arr in enumerate(arrays):
            pad_length = target_length - len(arr)

            if self.config.ARRAY_LENGTH_STRATEGY == "zero":
                # Pad with zeros
                arrays[i] = np.pad(arr, (0, pad_length), mode="constant")

            elif self.config.ARRAY_LENGTH_STRATEGY == "mean":
                # Pad with mean
                arrays[i] = np.pad(
                    arr, (0, pad_length), mode="constant", constant_values=np.mean(arr)
                )

            elif self.config.ARRAY_LENGTH_STRATEGY == "median":
                # Pad with median
                arrays[i] = np.pad(
                    arr, (0, pad_length), mode="constant", constant_values=np.median(arr)
                )

            elif self.config.ARRAY_LENGTH_STRATEGY == "last":
                # Pad with last value
                arrays[i] = np.pad(arr, (0, pad_length), mode="constant", constant_values=arr[-1])

            elif self.config.ARRAY_LENGTH_STRATEGY == "truncate":
                # Truncate to minimum length
                arrays[i] = arr[:target_length]

            elif self.config.ARRAY_LENGTH_STRATEGY == "global_mean":
                # Pad with global mean
                arrays[i] = np.pad(
                    arr, (0, pad_length), mode="constant", constant_values=global_mean
                )

            else:
                raise ValueError("Invalid array length strategy.")

        return arrays

    def _extract_statistical_features(self, column: str) -> None:
        """Extract statistical features from a column in the dataset.

        Args:
            column (str): The name of the column to extract features from.
        """

        self.logger.debug("Extracting statistical features")

        if column not in self.X.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

        self._extract_arrays(column)
        self._handle_array_lengths()

        X_array = np.array(self.X_arrays, dtype=np.float64)
        X_test_array = np.array(self.X_test_arrays, dtype=np.float64)

        df_statistical = extract_statistical_features_from_array(X_array, column)
        df_statistical_test = extract_statistical_features_from_array(X_test_array, column)

        self.X = pd.concat([self.X, df_statistical], axis=1)
        self.X_test = pd.concat([self.X_test, df_statistical_test], axis=1)


def extract_statistical_features_from_array(arr: np.ndarray, column: str) -> pd.DataFrame:
    """Extract statistical features from a NumPy array.

    Args:
        arr (np.ndarray): The array from which to extract features.
        column (str): The name of the column to extract features from.

    Returns:
        pd.DataFrame: DataFrame with the extracted statistical features.
    """

    arr = np.array(arr, dtype=np.float64)
    df = pd.DataFrame()

    # Extract statistical features
    df[f"{column}_mean"] = np.mean(arr, axis=1)
    df[f"{column}_std"] = np.std(arr, axis=1)
    df[f"{column}_var"] = np.var(arr, axis=1)
    df[f"{column}_min"] = np.min(arr, axis=1)
    df[f"{column}_max"] = np.max(arr, axis=1)
    df[f"{column}_sum"] = np.sum(arr, axis=1)
    df[f"{column}_range"] = df[f"{column}_max"] - df[f"{column}_min"]
    df[f"{column}_median"] = np.median(arr, axis=1)
    df[f"{column}_mean_abs"] = np.mean(np.abs(arr), axis=1)
    df[f"{column}_mean_ad"] = np.mean(np.abs(arr - np.mean(arr, axis=1, keepdims=True)), axis=1)
    df[f"{column}_median_ad"] = np.median(
        np.abs(arr - np.median(arr, axis=1, keepdims=True)), axis=1
    )
    df[f"{column}_q10"] = np.percentile(arr, 10, axis=1)
    df[f"{column}_q25"] = np.percentile(arr, 25, axis=1)
    df[f"{column}_q75"] = np.percentile(arr, 75, axis=1)
    df[f"{column}_q90"] = np.percentile(arr, 90, axis=1)
    df[f"{column}_iqr"] = df[f"{column}_q75"] - df[f"{column}_q25"]
    df[f"{column}_pos_count"] = np.sum(arr > 0.0, axis=1)
    df[f"{column}_neg_count"] = np.sum(arr < 0.0, axis=1)
    df[f"{column}_zero_count"] = np.sum(arr == 0.0, axis=1)
    df[f"{column}_total_count"] = arr.shape[1]
    df[f"{column}_above_mean"] = np.sum(arr > np.mean(arr, axis=1, keepdims=True), axis=1)
    df[f"{column}_above_median"] = np.sum(arr > np.median(arr, axis=1, keepdims=True), axis=1)
    df[f"{column}_peaks"] = np.array([signal.find_peaks(np.float64(a))[0].size for a in arr])
    df[f"{column}_peaks_prominence"] = np.array(
        [signal.find_peaks(np.float64(a), prominence=np.std(np.float64(a)))[0].size for a in arr]
    )
    df[f"{column}_skewness"] = stats.skew(arr, axis=1, nan_policy="omit")
    df[f"{column}_kurtosis"] = stats.kurtosis(arr, axis=1, nan_policy="omit")
    df[f"{column}_energy"] = np.sum(arr**2, axis=1) / arr.shape[1]
    df[f"{column}_rms"] = np.sqrt(np.mean(arr**2, axis=1))
    df[f"{column}_sma"] = np.sum(np.abs(arr), axis=1) / arr.shape[1]
    df[f"{column}_zero_crossings"] = np.sum(np.diff(np.signbit(arr)), axis=1)
    df[f"{column}_mean_crossings"] = np.sum(
        np.diff(np.signbit(arr - np.mean(arr, axis=1, keepdims=True))), axis=1
    )
    df[f"{column}_median_crossings"] = np.sum(
        np.diff(np.signbit(arr - np.median(arr, axis=1, keepdims=True))), axis=1
    )
    df[f"{column}_argmax"] = np.argmax(arr, axis=1)
    df[f"{column}_argmin"] = np.argmin(arr, axis=1)
    df[f"{column}_arg_diff"] = df[f"{column}_argmax"] - df[f"{column}_argmin"]

    return df


if __name__ == "__main__":
    pass
