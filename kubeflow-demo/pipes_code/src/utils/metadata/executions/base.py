from typing import Any
from pipes_code.src.utils.schemas.state import State


# MLDB deesn't work in macos
# from __future__ import annotations
# 
# from pipes.src.utils.metadata.store import store
# from pipes.src.utils.metadata.base import MetadataType, MetadataDefinition, MetadataObject
# from ml_metadata.proto import metadata_store_pb2
# from contextlib import contextmanager
# 
# 
# class ExecutionType(MetadataType, metadata_store_pb2.ExecutionType):
#     def register(self) -> int:
#         return store.put_execution_type(self)
# 
# 
# class ExecutionDefinition(MetadataDefinition, metadata_store_pb2.Execution):
#     def register(self) -> int:
#         return store.put_executions([self])[0]
# 
# 
# class Execution(MetadataObject):
#     def __init__(self, name: str, metadata: dict[str, Any]):
#         super().__init__(
#             name=name,
#             metadata=metadata,
#             type_object=ExecutionType,
#             def_object=ExecutionDefinition
#         )
#         self.set_state(State.PENDING)
# 
#     def set_state(self, state: State):
#         self.state = state
#         self.metadata.update({'state': state})
#         self.update_metadata(self.metadata)
# 
#     def __enter__(self):
#         self.set_state(State.RUNNING)
#         try:
#             return self
#         except Exception as err:
#             self.set_state(State.STOPPED)
#             self.metadata.update({"error": err})
#             self.update_metadata(self.metadata)
# 
#     def __exit__(self, *args, **kwargs):
#         self.set_state(State.COMPLETED)


class Execution:
    def __init__(self, name: str, metadata: dict[str, Any]):
        self.name = name
        self.metadata = metadata
        self.set_state(State.PENDING)

    def set_state(self, state: State):
        self.state = state
        self.metadata.update({'state': state})

    def __enter__(self):
        self.set_state(State.RUNNING)
        try:
            return self
        except Exception as err:
            self.set_state(State.STOPPED)
            self.metadata.update({"error": err})

    def __exit__(self, *args, **kwargs):
        self.set_state(State.COMPLETED)

    def update_metadata(self, metadata: dict[str, Any]):
        self.metadata.update(metadata)
