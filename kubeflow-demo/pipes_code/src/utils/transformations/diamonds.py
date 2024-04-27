import json
import pandas as pd
import numpy as np
import torch
import os

from pipes_code.src.utils.transformations.base import BaseDataTransform
from pipes_code.src.configs import DiamondsDatasetConfig
from typing import cast


class DiamondsDataTransform(BaseDataTransform, DiamondsDatasetConfig):

    @classmethod
    def preprocessing(
        cls,
        data: pd.DataFrame,
        for_inference: bool = False
    ) -> tuple[pd.DataFrame, dict]:
        metadata = {}

        # Capture just the important features and target
        if for_inference:
            data = data[cls.TRAINING_FEATURES]
        else:
            data = data[cls.TRAINING_FEATURES + [cls.TARGET]]

        # Drop unknown categorical labels
        if not for_inference:
            data = cls.drop_unknown_labels(data)

        # Process variables for training
        for col_name in data.columns:
            if col_name in cls.POSITIVE_DATA:
                data = data.loc[data[col_name] > 0]
            
            if not for_inference:
                data = cls.drop_outliers(data, col_name)

            if col_name in cls.CATEGORICAL_FEATURES:      
                data = cls.apply_one_hot_encoder(data, col_name)
            elif col_name in cls.NUMERICAL_FEATURES:
                data[col_name] = cls.apply_std_scaler(data[col_name])
            elif col_name == cls.TARGET:
                data[col_name], m = cls.apply_target_scaler(data[col_name])
                metadata.update(m)

        return data, metadata


    @classmethod
    def postprocessing(
        cls, tensor: np.ndarray | torch.Tensor,
        metadata: dict | str | os.PathLike
    ) -> np.ndarray | torch.Tensor:
        if isinstance(metadata, (str, os.PathLike)):
            with open(metadata, 'r') as f:
                metadata = cast(dict, json.load(f))
            
        tensor *= metadata['target_std']
        tensor += metadata['target_mu']
        tensor = np.exp(tensor)
        return tensor
    

    @staticmethod
    def z_score(col):
        sigma = col.std()
        mean = col.median()
        return np.abs((col - mean) / sigma)


    @classmethod
    def drop_outliers(cls, data, col_name):
        if col_name in cls.ZSCORE_WITH_THRESHOLD_4:
            data = data[cls.z_score(data[col_name]) < 4]
        elif col_name in cls.ZSCORE_WITH_THRESHOLD_3:
            data = data[cls.z_score(data[col_name]) < 3]
        return data


    @classmethod
    def drop_unknown_labels(cls, data):
        return data[
            (data["clarity"].isin(cls.CLARITY_LABELS))
            & (data['color'].isin(cls.COLOR_LABELS))
            & (data['cut'].isin(cls.CUT_LABELS))
        ]


    @staticmethod
    def apply_one_hot_encoder(data, col_name):
        encoded_df = pd.get_dummies(data[col_name], prefix=col_name, dtype=int)
        return data.drop(columns=[col_name]).join(encoded_df)


    @staticmethod
    def apply_std_scaler(col):
        col -= col.mean()
        col /= col.std()
        return col


    @staticmethod
    def apply_target_scaler(col):
        col = np.log(col)
        mu = col.mean()
        std = col.std()
        col -= mu
        col /= std

        metadata = {
            'target_mu': mu,
            'target_std': std
        }
        return col, metadata
