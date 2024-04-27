from typing import Any
from pipes_code.src.utils.schemas.pipeline import PipelineConfig

class Component:
    name: str
    metadata: dict[str, Any] = {}
    cfg: PipelineConfig
