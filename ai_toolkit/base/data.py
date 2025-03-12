from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


@dataclass
class DatasetConfig:
    """Configuration for data loading and preprocessing."""

    CATEGORICAL_FILL_STRATEGY: str = field(
        default="mode",
        metadata={"description": "Strategy to fill missing categorical values."},
    )
    NUMERICAL_FILL_STRATEGY: str = field(
        default="median",
        metadata={"description": "Strategy to fill missing numerical values."},
    )
    CATEGORICAL_PREPROCESSING_STRATEGY: str = field(
        default="OneHotEncoder",
        metadata={"description": "Strategy to preprocess categorical columns."},
    )
    NUMERICAL_PREPROCESSING_STRATEGY: str = field(
        default="StandardScaler",
        metadata={"description": "Strategy to preprocess numerical columns."},
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

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset."""
        pass

    def preprocess(self) -> None:
        """Preprocess the dataset."""

        self._check_data()
        self._detect_column_types()
        self._handle_missing_categorical_values()
        self._handle_missing_numerical_values()
        self._preprocess_categorical()
        self._preprocess_numerical()

    def _check_data(self) -> None:
        """Check if the data is loaded."""

        if self.X is None:
            raise ValueError("X data not loaded.")
        elif self.y is None:
            raise ValueError("y data not loaded.")
        elif self.X_test is None:
            raise ValueError("X_test data not loaded.")

    def _detect_column_types(self) -> None:
        """Detect column types in the dataset."""

        # Categorical columns
        self.categorical_columns = self.X.select_dtypes(
            include=["object", "category"]
        ).columns

        # Numerical columns
        self.numerical_columns = self.X.select_dtypes(
            include=["int64", "float64"]
        ).columns

    def _handle_missing_categorical_values(self) -> None:
        """Handle missing categorical values in the dataset."""

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
            self.X_test[self.categorical_columns] = self.X_test[
                self.categorical_columns
            ].fillna(self.X_test[self.categorical_columns].mode().iloc[0])
        elif self.config.CATEGORICAL_FILL_STRATEGY == "missing":
            self.X[self.categorical_columns] = self.X[self.categorical_columns].fillna(
                "MISSING"
            )
            self.X_test[self.categorical_columns] = self.X_test[
                self.categorical_columns
            ].fillna("MISSING")
        else:
            raise ValueError("Invalid categorical fill strategy.")

    def _handle_missing_numerical_values(self) -> None:
        """Handle missing numerical values in the dataset."""

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
            self.X_test[self.numerical_columns] = self.X_test[
                self.numerical_columns
            ].fillna(self.X_test[self.numerical_columns].median())
        elif self.config.NUMERICAL_FILL_STRATEGY == "mean":
            self.X[self.numerical_columns] = self.X[self.numerical_columns].fillna(
                self.X[self.numerical_columns].mean()
            )
            self.X_test[self.numerical_columns] = self.X_test[
                self.numerical_columns
            ].fillna(self.X_test[self.numerical_columns].mean())
        elif self.config.NUMERICAL_FILL_STRATEGY == "zero":
            self.X[self.numerical_columns] = self.X[self.numerical_columns].fillna(0.0)
            self.X_test[self.numerical_columns] = self.X_test[
                self.numerical_columns
            ].fillna(0.0)
        else:
            raise ValueError("Invalid numerical fill strategy.")

    def _label_encode(self) -> None:
        """Label encode categorical columns in the dataset."""

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

        # Check if categorical columns exist
        if self.categorical_columns.empty:
            return

        if self.config.CATEGORICAL_PREPROCESSING_STRATEGY == "LabelEncoder":
            self._label_encode()

        elif self.config.CATEGORICAL_PREPROCESSING_STRATEGY == "OneHotEncoder":
            df_encoded, df_encoded_test = self._one_hot_encode()

            self.X = pd.concat(
                [self.X.drop(columns=self.categorical_columns), df_encoded], axis=1
            )
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
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test), columns=self.X.columns
        )

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

    def get_feature_statistics(self) -> pd.DataFrame:
        """Get statistical information about features.

        Returns:
            pd.DataFrame: DataFrame containing feature statistics.
        """

        df_stats = self.X.describe().T
        df_stats["missing"] = self.X.isna().sum()
        df_stats["unique"] = self.X.nunique()
        df_stats["dtype"] = self.X.dtypes

        return df_stats

    def feature_engineering(self) -> None:
        """Perform feature engineering on the dataset."""
        pass


if __name__ == "__main__":
    pass
