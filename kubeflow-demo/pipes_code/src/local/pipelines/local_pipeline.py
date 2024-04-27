from pipes_code.src.configs import DiamondsDatasetConfig
from pipes_code.src.local.components.deploy.local import LocalDeploy
from pipes_code.src.local.components.evaluation.diamonds import DiamondsEvaluation
from pipes_code.src.local.components.ingest.local import LocalIngest
from pipes_code.src.local.components.preprocess.diamonds import DiamondsPreprocess
from pipes_code.src.local.components.train.diamonds import DiamondsTrain
from pipes_code.src.local.pipelines.base import BaseReTrainPipeline
from pipes_code.src.utils.schemas.pipeline import DiamondsPipelineInput


class DiamondsLocalPipeline(
    BaseReTrainPipeline,
    LocalIngest,
    DiamondsPreprocess,
    DiamondsTrain,
    DiamondsEvaluation,
    LocalDeploy
):
    def __init__(self, input: DiamondsPipelineInput, cfg = DiamondsDatasetConfig()):
        self.input = input
        self.cfg = cfg
        self.__class__.cfg = cfg
