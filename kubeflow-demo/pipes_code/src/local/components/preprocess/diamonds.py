import pandas as pd

from pipes_code.src.local.components.preprocess.base import PreprocessComponent
from pipes_code.src.utils.datasets.common import random_split
from pipes_code.src.utils.metadata.artifacts.dataset import DatasetArtifact
from pipes_code.src.utils.metadata.executions.base import Execution
from pipes_code.src.utils.schemas.pipeline import PreprocessInput, PreprocessOutput
from pipes_code.src.utils.transformations.diamonds import DiamondsDataTransform


class DiamondsPreprocess(PreprocessComponent):

    @classmethod
    def do_preprocess(cls, input: PreprocessInput, ctx: Execution) -> PreprocessOutput:
        _user_data = pd.read_csv(input.user_data.uri)
        _user_data, m = DiamondsDataTransform.preprocessing(_user_data)
        ctx.update_metadata(m)

        _user_data = pd.read_csv(input.user_data.uri)
        _base_data = pd.read_csv(input.base_data.uri)

        _user_data, m = DiamondsDataTransform.preprocessing(_user_data)
        _base_data, m = DiamondsDataTransform.preprocessing(_base_data)
        ctx.update_metadata(m)

        test_data, _user_data = random_split(_user_data, [0.5, 0.5])
        entire_train_data = pd.concat((_base_data, _user_data))
        train_data, val_data = random_split(entire_train_data, [0.9, 0.1])

        train_dataset_op = DatasetArtifact('train_dataset')
        test_dataset_op = DatasetArtifact('test_dataset')
        val_dataset_op = DatasetArtifact('val_dataset')

        train_data.to_csv(train_dataset_op.uri, index=False)
        test_data.to_csv(test_dataset_op.uri, index=False)
        val_data.to_csv(val_dataset_op.uri, index=False)

        return PreprocessOutput(
            train_dataset=train_dataset_op,
            test_dataset=test_dataset_op,
            val_dataset=val_dataset_op,
        )
