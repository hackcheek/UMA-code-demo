import ml_metadata as mlmd
from ml_metadata import MetadataStore
from ml_metadata.proto import ArtifactType, metadata_store_pb2


class TaskArtifactType(ArtifactType):
    def __init__(self, store: MetadataStore):
        self.name = 'Task'
        self.store = store
        self.properties['id'] = metadata_store_pb2.INT
        self.properties['start_time'] = metadata_store_pb2.INT
        self.properties['block_number'] = metadata_store_pb2.INT
        self.properties['result_url'] = metadata_store_pb2.STRING
        self.properties['model_url'] = metadata_store_pb2.STRING
        self.properties['dataset_url'] = metadata_store_pb2.STRING


class DatasetArtifactType(ArtifactType):
    ...


class ModelArtifactType(ArtifactType):
    ...
