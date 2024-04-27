from typing import Any
from pipes_code.src.utils.metadata.store import store
from pipes_code.src.utils.metadata.base import MetadataType, MetadataDefinition, MetadataObject
from ml_metadata.proto import metadata_store_pb2


class ExecutionType(MetadataType, metadata_store_pb2.ExecutionType):
    def register(self) -> int:
        return store.put_execution_type(self)


class ExecutionDefinition(MetadataDefinition, metadata_store_pb2.Artifact):
    def register(self) -> int:
        return store.put_executions([self])[0]


class Execution(MetadataObject):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            uri=uri,
            metadata=metadata,
            type_object=ExecutionType,
            def_object=ExecutionDefinition
        )
