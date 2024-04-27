import json
import torch
import pandas as pd

from torch.utils.data import DataLoader
from pipes_code.src.configs import DiamondsDatasetConfig
from pipes_code.src.local.components.evaluation.base import EvaluationComponent
from pipes_code.src.utils.datasets.diamonds import DiamondsPytorchDataset
from pipes_code.src.utils.metadata.artifacts.json import JsonArtifact
from pipes_code.src.utils.metadata.artifacts.model import ModelArtifact
from pipes_code.src.utils.metadata.executions.base import Execution
from pipes_code.src.utils.schemas.pipeline import EvaluationInput, EvaluationOutput


class DiamondsEvaluation(EvaluationComponent):
    """
    This evaluation object was built to evaluate only diamonds dataset.
    TODO: develop a generic evaluation object
    NOTE: Perhaps the better way is using ctx variable to manage depenedencies
    """
    @classmethod
    def do_eval(cls, input: EvaluationInput, ctx: Execution) -> EvaluationOutput:

        print(f"[*] Evaluating model")

        def get_dataloader(data):
            cat_features = list(filter(
                lambda x: x not in DiamondsDatasetConfig.NUMERICAL_FEATURES
                + [DiamondsDatasetConfig.TARGET],
                data.columns
            ))

            data = DiamondsPytorchDataset(
                data,
                cat_features,
                DiamondsDatasetConfig.NUMERICAL_FEATURES,
                DiamondsDatasetConfig.TARGET
            )

            return DataLoader(data, batch_size=1024, shuffle=True)


        def evaluate(model, test_loader):
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

        _model = torch.jit.load(input.model.uri)
        _test_dataset = pd.read_csv(input.test_data.uri)
        _base_model = torch.jit.load(input.base_model.uri)

        test_loader = get_dataloader(_test_dataset)

        current_model_loss = evaluate(_model, test_loader)
        base_model_loss = evaluate(_base_model, test_loader)
        
        input.model.metadata['test_loss'] = current_model_loss
        input.base_model.metadata['test_loss'] = base_model_loss

        print(f"[*] Current Model Loss: {current_model_loss}")
        print(f"[*] Baseline Model Loss: {base_model_loss}")
        
        results = {
            'current_model_loss': current_model_loss,
            'base_model_loss': base_model_loss
        }
        
        results_op = JsonArtifact('results')

        with open(results_op.uri, 'w') as f:
            json.dump(results, f)
        
        return EvaluationOutput(results_op)
