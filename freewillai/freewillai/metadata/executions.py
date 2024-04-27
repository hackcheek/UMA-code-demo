import asyncio
from ml_metadata.metadata_store import MetadataStore
from ml_metadata.proto import metadata_store_pb2
from enum import Enum


class TaskStates(str, Enum):
    RUNNING = "RUNNING" 
    PENDING = "PENDING" 
    COMPLETED = "COMPLETED" 


connection_config = metadata_store_pb2.ConnectionConfig() # type: ignore
connection_config.fake_database.SetInParent()
store = MetadataStore(connection_config)


connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.sqlite.filename_uri = '/tmp/test_db'
connection_config.sqlite.connection_mode = 3 # READWRITE_OPENCREATE
store = MetadataStore(connection_config)


run_task_type = metadata_store_pb2.ExecutionType() # type: ignore
run_task_type.name = "RunTask"
run_task_type.properties["state"] = metadata_store_pb2.STRING # type: ignore
run_task_type_id = store.put_execution_type(run_task_type)

run_task_execution = metadata_store_pb2.Execution() # type: ignore
run_task_execution.type_id = run_task_type_id
run_task_execution.properties["state"].string_value = TaskStates.PENDING
[run_task_id] = store.put_executions([run_task_execution])


async def task():
    [run_task_execution] = store.get_executions_by_id([run_task_id])
    run_task_execution.id = run_task_id
    run_task_execution.properties['state'].string_value = TaskStates.RUNNING
    store.put_executions([run_task_execution])
    await asyncio.sleep(5)
    run_task_execution.properties['state'].string_value = TaskStates.COMPLETED
    store.put_executions([run_task_execution])
    return "result"
