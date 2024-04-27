from kfp import dsl
from kfp.dsl import Dataset, Output, Model
from pipes_code.src.configs import MinioConfig
from functools import partial


@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'minio'])
def _ingest_csv_component(
    csv_loc: str,
    baseline_model_loc: str,
    baseline_data_loc: str,
    port: int,
    access_key: str,
    secret_key: str,
    user_data: Output[Dataset],
    base_data: Output[Dataset],
    base_model: Output[Model]
):
    """
    Note that, in a real world environment, this kind of component ingest data to a database and would return table name.
    However for this case I'll ingest the data to a temp csv file and return the path.
    """
    import pandas as pd
    from minio import Minio
    from io import BytesIO

    client = Minio(
        f'host.k3d.internal:{port}',
        access_key,
        secret_key,
        secure=False
    )

    print("[*] Ingesting data")

    response = client.get_object(*csv_loc.split('/', 1))
    data = pd.read_csv(BytesIO(response.data))
    data.to_csv(user_data.path, index=False)

    response = client.get_object(*baseline_data_loc.split('/', 1))
    _base_data = pd.read_csv(BytesIO(response.data))
    _base_data.to_csv(base_data.path, index=False)

    response = client.get_object(*baseline_model_loc.split('/', 1))
    client.fget_object(*baseline_model_loc.split('/', 1), base_model.path)


ingest_csv_component = partial(
    _ingest_csv_component,
    port=MinioConfig.PORT,
    access_key=MinioConfig.ACCESS_KEY,
    secret_key=MinioConfig.SECRET_KEY
)
