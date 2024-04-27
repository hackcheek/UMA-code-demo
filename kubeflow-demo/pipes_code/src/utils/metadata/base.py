from typing import Any
from ml_metadata.proto import metadata_store_pb2


class MetadataType:
    _types = {
        str: metadata_store_pb2.STRING,
        int: metadata_store_pb2.INT,
        float: metadata_store_pb2.DOUBLE,
    }

    def __init__(self, name: str, properties: dict[str, Any]):
        super().__init__()
        self.name = name
        self.properties = self.get_properties_schema(properties)

    def get_properties_schema(self, properties: dict[str, Any]):
        schema = {
            k:self._types.get(type(v), metadata_store_pb2.UNKNOWN)
            for k, v in properties.items()
        }
        return schema

    def register(self) -> int:
        raise NotImplementedError


class MetadataDefinition:
    _attribute_by_type = {
        str: 'string_value',
        int: 'int_value',
        float: 'double_value',
    }

    def __init__(self, name: str, metadata: dict[str, Any], uri: str, type_id: int):
        super().__init__()
        self.name = name
        self.uri = uri
        self.type_id = type_id

        self.metadata = metadata
        self.define_properties(metadata)

    def define_properties(self, properties: dict[str, Any]):
        for k, v in properties.items():
            attr = self._attribute_by_type[type(v)]
            setattr(self.properties[k], attr, v)
            
    def register(self) -> int:
        raise NotImplementedError

    def update_properties(self, properties: dict[str, Any]) -> int:
        self.define_properties(properties)
        return self.register()


class MetadataObject:
    def __init__(
        self,
        name: str,
        metadata: dict[str, Any],
        type_object: type[MetadataType],
        def_object: type[MetadataDefinition],
        **kwargs
    ):
        self.name = name
        self.metadata = metadata

        self.type = type_object(name, metadata)
        self.type_id = self.type.register()
        self._def = def_object(
            name=self.name,
            metadata=self.metadata,
            type_id=self.type_id,
            **kwargs,
        )
        self.id = self._def.register()

    def update_metadata(self, metadata: dict[str, Any]) -> int:
        return self._def.update_properties(metadata)
