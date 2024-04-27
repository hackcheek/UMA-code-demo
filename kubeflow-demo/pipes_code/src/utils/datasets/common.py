import pandas as pd
import numpy as np
from itertools import accumulate


def random_split(data: pd.DataFrame, partitions: list[float]) -> list[pd.DataFrame]:
    shuffled_data = data.copy().sample(frac=1)
    m = shuffled_data.shape[0]
    samples = [int(i * m) for i in accumulate(partitions)]
    *datasets, residual = np.split(shuffled_data, samples)
    if residual.shape[0] < 10:
        datasets[-1] = pd.concat([datasets[-1], residual])
    return datasets
