from pipes_code.src.local.components.base import Component
from pipes_code.src.utils.metadata.executions.base import Execution
from pipes_code.src.utils.schemas.pipeline import EvaluationOutput, EvaluationInput
from typing import cast


class EvaluationComponent(Component):
    name = 'evaluation'

    @classmethod
    def do_eval(cls, input: EvaluationInput, ctx: Execution) -> EvaluationOutput:
        raise NotImplementedError

    @classmethod
    def eval(cls, input: EvaluationInput) -> EvaluationOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_eval(input, cast(Execution, ctx))
