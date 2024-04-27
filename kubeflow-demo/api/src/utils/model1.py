import pandas as pd
import torch
from api.src.utils.model import _predict


V1_MODEL_PATH = 'modeling/best_model.pt'
V1_METADATA_PATH = 'modeling/preproc_metadata.json'


def predict(user_data: pd.DataFrame):
    model = torch.jit.load(V1_MODEL_PATH)
    return _predict(model, user_data, V1_METADATA_PATH)
