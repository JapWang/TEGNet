import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from typing import Optional
import warnings

class DataPreprocessor:
    """
    TEG-Net data preprocessing module.
    Loads static single-cell expression data, performs KNN imputation for dropout events,
    and applies MinMax scaling to [0,1] for downstream heteroskedasticity detection.
    """

    @classmethod
    def load_data(
        cls,
        file_path: str,
        n_features: Optional[int] = None,
        n_neighbors: int = 5,
        standardize: bool = True
    ) -> pd.DataFrame:

        # Auto-detect index column
        temp_df = pd.read_csv(file_path, nrows=1)
        first_col = str(temp_df.columns[0]).lower()
        if "unnamed" in first_col or "id" in first_col or "time" in first_col or "cell" in first_col:
            data = pd.read_csv(file_path, index_col=0)
        else:
            data = pd.read_csv(file_path)

        # Optional feature truncation for benchmarking
        if n_features:
            data = data.iloc[:, :n_features]

        # KNN imputation for single-cell dropout events
        if data.isna().any().any():
            warnings.warn("Missing values (dropout) detected. Running KNN imputation...")
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            data = pd.DataFrame(imputer.fit_transform(data), index=data.index, columns=data.columns)

        # Ensure non-negative expression values
        data = data.clip(lower=0.0)

        # MinMax scaling to [0,1] — preserves variance gradient for HSIC
        if standardize:
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(data.values)
            data = pd.DataFrame(scaled_values, index=data.index, columns=data.columns)

        return data
