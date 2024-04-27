from tempfile import NamedTemporaryFile
import shutil

from pipes_code.src.local.components.ingest.base import IngestComponent
from pipes_code.src.utils.metadata.artifacts.dataset import DatasetArtifact
from pipes_code.src.utils.metadata.artifacts.model import ModelArtifact
from pipes_code.src.utils.metadata.executions.base import Execution
from pipes_code.src.utils.schemas.pipeline import DiamondsPipelineInput, IngestOutput


class LocalIngest(IngestComponent):
    
    @staticmethod
    def _mv_to_work_path(path: str) -> str:
        new_path = NamedTemporaryFile().name
        shutil.copy(path, new_path)
        return new_path

    @classmethod
    def do_ingest(cls, input: DiamondsPipelineInput, ctx: Execution) -> IngestOutput:
        _base_model = ModelArtifact(
            'base_model',
            cls._mv_to_work_path(input.base_model_path)
        )
        _base_data = DatasetArtifact(
            'base_data',
            cls._mv_to_work_path(input.base_data_path)
        )
        _user_data = DatasetArtifact(
            'user_data',
            cls._mv_to_work_path(input.user_data_path)
        )

        return IngestOutput(
            base_model=_base_model,
            base_data=_base_data,
            user_data=_user_data,
        ) 
