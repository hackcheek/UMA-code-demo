from dataclasses import dataclass

from pipes_code.src.utils.metadata.artifacts.dataset import DatasetArtifact
from pipes_code.src.utils.metadata.artifacts.json import JsonArtifact
from pipes_code.src.utils.metadata.artifacts.model import ModelArtifact


@dataclass
class PipelineInput:
    ...


@dataclass
class PipelineOutput:
    ...


@dataclass
class PipelineConfig:
    ...


@dataclass
class DiamondsPipelineInput(PipelineInput):
    base_data_path: str
    user_data_path: str
    base_model_path: str


@dataclass
class IngestOutput:
    base_model: ModelArtifact
    base_data: DatasetArtifact
    user_data: DatasetArtifact


@dataclass
class PreprocessInput:
    base_data: DatasetArtifact
    user_data: DatasetArtifact


@dataclass
class PreprocessOutput:
    train_dataset: DatasetArtifact
    test_dataset: DatasetArtifact
    val_dataset: DatasetArtifact


@dataclass
class TrainInput:
    train_dataset: DatasetArtifact
    val_dataset: DatasetArtifact


@dataclass
class TrainOutput:
    model: ModelArtifact


@dataclass
class EvaluationInput:
    model: ModelArtifact
    base_model: ModelArtifact
    test_data: DatasetArtifact


@dataclass
class EvaluationOutput:
    results: JsonArtifact


@dataclass
class DeployInput:
    user_data: DatasetArtifact
    base_data: DatasetArtifact
    model: ModelArtifact


@dataclass
class DeployOutput:
    deployed: bool
