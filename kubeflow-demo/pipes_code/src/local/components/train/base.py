from pipes_code.src.local.components.base import Component
from pipes_code.src.utils.metadata.executions.base import Execution
from pipes_code.src.utils.schemas.pipeline import TrainOutput, TrainInput
from typing import cast


class TrainComponent(Component):
    name = 'train'

    @classmethod
    def do_train(cls, input: TrainInput, ctx: Execution) -> TrainOutput:
        raise NotImplementedError

    @classmethod
    def train(cls, input: TrainInput) -> TrainOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_train(input, cast(Execution, ctx))
