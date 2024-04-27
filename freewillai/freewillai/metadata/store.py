import os
from ml_metadata.metadata_store import MetadataStore
from ml_metadata.metadata_store.metadata_store import Optional
from ml_metadata.proto import metadata_store_pb2
from freewillai.globals import Global


SQLITE_FILENAME = os.path.join(Global.working_directory, 'metadata_db')


class DiskMetadataStore(MetadataStore):
    def __init__(self, db_path: Optional[str] = None):
        self.config = metadata_store_pb2.ConnectionConfig()
        self.config.sqlite.filename_uri = db_path or SQLITE_FILENAME
        self.config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
        super().__init__(self.config)


class MemoryMetadataStore(MetadataStore):
    def __init__(self):
        """
        WARNING: This store does not work with multiprocessing
        """
        connection_config = metadata_store_pb2.ConnectionConfig()
        connection_config.fake_database.SetInParent()
        super().__init__(connection_config)
