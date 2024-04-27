from typing import Any
from pipes_code.src.utils.metadata.artifacts.base import Artifact
from tempfile import NamedTemporaryFile


class ModelArtifact(Artifact):
    def __init__(
        self,
        name: str,
        uri: str | None = None,
        metadata: dict[str, Any] = {}
    ):
        if uri is None:
            uri = NamedTemporaryFile().name 
        super().__init__(name=name, metadata=metadata, uri=uri)
