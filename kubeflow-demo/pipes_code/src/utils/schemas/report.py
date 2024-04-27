from typing import NamedTuple


class PipelineReport(NamedTuple):
    current_model_loss: float
    base_model_loss: float 
