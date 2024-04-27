from kfp import dsl
from kfp.dsl import Input, Dataset, Model
from functools import partial

from pipes_code.src.configs import DiamondsDatasetConfig
from typing import NamedTuple


@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'torch', 'numpy'])
def _pytorch_model_evaluation_component(
    model: Input[Model],
    test_dataset: Input[Dataset],
    base_model: Input[Model],
    dataset_cfg: dict,
) -> NamedTuple(
  'results',
  [
    ('current_model_loss', float),
    ('base_model_loss', float)
  ]
):
    import torch
    import pandas as pd
    import numpy as np

    from torch.utils.data import DataLoader, Dataset

    print(f"[*] Evaluating model")


    class DiamondsPytorchDataset(Dataset):
        def __init__(self, data, cat_features, num_features, label):
            self.data = data
            self.cat_features = cat_features
            self.num_features = num_features
            self.label = label

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            x_num = torch.from_numpy(
                row[self.num_features].to_numpy().astype(np.float32)
            ).view(-1)

            x_cat = torch.from_numpy(
                row[self.cat_features].to_numpy().astype(np.int32)
            ).view(-1)

            target = torch.from_numpy(row[[self.label]].to_numpy().astype(np.float32))
            return x_cat, x_num, target


    def get_dataloader(data):
        cat_features = list(filter(
            lambda x: x not in dataset_cfg['NUMERICAL_FEATURES'] + [dataset_cfg['TARGET']],
            data.columns
        ))

        data = DiamondsPytorchDataset(
            data,
            cat_features,
            dataset_cfg['NUMERICAL_FEATURES'],
            dataset_cfg['TARGET']
        )

        return DataLoader(data, batch_size=1024, shuffle=True)


    def evaluate(model):
        model.eval()
        loss_fn = torch.nn.MSELoss()
        test_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, y in test_loader:
                pred = model(x_cat, x_num)
                loss = loss_fn(pred, y)
                test_loss += loss.item() * x_cat.size(0)

        test_loss = test_loss / len(_test_dataset)
        return test_loss


    _model = torch.jit.load(model.path)
    _test_dataset = pd.read_csv(test_dataset.path)
    _base_model = torch.jit.load(base_model.path)

    test_loader = get_dataloader(_test_dataset)

    current_model_loss = evaluate(_model)
    base_model_loss = evaluate(_base_model)
    
    model.metadata['test_loss'] = current_model_loss
    base_model.metadata['test_loss'] = base_model_loss

    print(f"[*] Current Model Loss: {current_model_loss}")
    print(f"[*] Baseline Model Loss: {base_model_loss}")
    
    return current_model_loss, base_model_loss
    

pytorch_model_evaluation_component = partial(
    _pytorch_model_evaluation_component,
    dataset_cfg=DiamondsDatasetConfig.asdict()
)
