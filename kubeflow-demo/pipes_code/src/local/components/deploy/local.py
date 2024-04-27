import shutil
import pandas as pd

from pipes_code.src.configs import ExamplePipelineConfig
from pipes_code.src.local.components.deploy.base import DeployComponent
from pipes_code.src.utils.metadata.artifacts.model import ModelArtifact
from pipes_code.src.utils.metadata.artifacts.dataset import DatasetArtifact
from pipes_code.src.utils.metadata.executions.base import Execution
from pipes_code.src.utils.schemas.pipeline import DeployInput, DeployOutput


class LocalDeploy(DeployComponent):
    cfg: ExamplePipelineConfig

    @classmethod
    def do_deploy(cls, input: DeployInput, ctx: Execution) -> DeployOutput:
        try:
            model_op = ModelArtifact('trained_model', uri=cls.cfg.output_model_path)
            data_op = DatasetArtifact('data', uri=cls.cfg.output_data_path)

            _user_data = pd.read_csv(input.user_data.uri)
            _base_data = pd.read_csv(input.base_data.uri)

            new_baseline_data = pd.concat((_base_data, _user_data)) 

            new_baseline_data.to_csv(data_op.uri)
            shutil.copy(input.model.uri, model_op.uri)
        except Exception as err:
            print("[!] Error has occurred trying to deploy:", err)
            ctx.update_metadata({'error_in_deploy': err}) 
            return DeployOutput(False)

        return DeployOutput(True)
