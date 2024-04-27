from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2


def get_metadata_store():
    """
    In memory by default
    """
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.fake_database.SetInParent()
    return metadata_store.MetadataStore(connection_config)


store = get_metadata_store()
