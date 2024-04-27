from kfp import dsl
from kfp.dsl import Input, Dataset, Output
from pipes_code.src.configs import DiamondsDatasetConfig
from functools import partial


@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'numpy'])
def _preprocess_component(
    user_data: Input[Dataset],
    base_data: Input[Dataset],
    cfg: dict,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    val_dataset: Output[Dataset],
):
    import numpy as np
    import pandas as pd
    from itertools import accumulate

    print("[*] Preprocessing and Splitting data")

    def z_score(col):
        sigma = col.std()
        mean = col.median()
        return np.abs((col - mean) / sigma)


    def drop_outliers(data):
        threshold_4_cols = ['table', 'depth', 'carat']
        threshold_3_cols = ['x', 'y', 'z', 'price']
        for col_name in data.columns:
            if col_name in threshold_4_cols:
                data = data[z_score(data[col_name]) < 4]
            elif col_name in threshold_3_cols:
                data = data[z_score(data[col_name]) < 3]
        return data


    def drop_unknown_labels(data):
        return data[
            (data["clarity"].isin(cfg['CLARITY_LABELS']))
            & (data['color'].isin(cfg['COLOR_LABELS']))
            & (data['cut'].isin(cfg['CUT_LABELS']))
        ]


    def apply_one_hot_encoder(data, col_name):
        encoded_df = pd.get_dummies(data[col_name], prefix=col_name, dtype=int)
        return data.drop(columns=[col_name]).join(encoded_df)


    def apply_std_scaler(col):
        col -= col.mean()
        col /= col.std()
        return col


    def apply_target_scaler(col):
        col = np.log(col)
        col -= col.mean()
        col /= col.std()
        return col


    def random_split(data: pd.DataFrame, partitions: list[float]) -> list[pd.DataFrame]:
        shuffled_data = data.copy().sample(frac=1)
        m = shuffled_data.shape[0]
        samples = [int(i * m) for i in accumulate(partitions)]
        *datasets, residual = np.split(shuffled_data, samples)
        if residual.shape[0] < 10:
            datasets[-1] = pd.concat([datasets[-1], residual])
        return datasets

    
    def preprocess(data):
        # Capture just the important features and target
        data = data[cfg['TRAINING_FEATURES'] + [cfg['TARGET']]]

        # Price should be more than zero
        data = data.loc[data['price'] > 0]

        # Sizes should be more than zero
        data = data.loc[data['x'] > 0]

        # Drop outliers
        data = drop_outliers(data)

        # Drop unknown categorical labels
        data = drop_unknown_labels(data)

        # Process variables for training
        for col_name in data.columns:
            if col_name in cfg['CATEGORICAL_FEATURES']:
                data = apply_one_hot_encoder(data, col_name)
            elif col_name in cfg['NUMERICAL_FEATURES']:
                data[col_name] = apply_std_scaler(data[col_name])
            elif col_name == cfg['TARGET']:
                data[col_name] = apply_target_scaler(data[col_name])

        return data

    
    _user_data = pd.read_csv(user_data.path)
    _base_data = pd.read_csv(base_data.path)

    _user_data = preprocess(_user_data)
    _base_data = preprocess(_base_data)

    test_data, _user_data = random_split(_user_data, [0.5, 0.5])
    entire_train_data = pd.concat((_base_data, _user_data))
    train_data, val_data = random_split(entire_train_data, [0.9, 0.1])

    train_data.to_csv(train_dataset.path, index=False)
    test_data.to_csv(test_dataset.path, index=False)
    val_data.to_csv(val_dataset.path, index=False)


preprocess_component = partial(
    _preprocess_component,
    cfg=DiamondsDatasetConfig.asdict()
)
