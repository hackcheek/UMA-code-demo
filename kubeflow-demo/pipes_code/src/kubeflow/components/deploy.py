from kfp import dsl
from kfp.dsl import Dataset, Model, Input
from pipes_code.src.configs import MinioConfig
from functools import partial


@dsl.component(base_image='python:3.10', packages_to_install=['minio', 'pandas'])
def _deploy_component(
    trained_model: Input[Model],
    user_data: Input[Dataset],
    base_data: Input[Dataset],
    baseline_model_loc: str,
    baseline_data_loc: str,
    minio_port: int,
    minio_access_key: str,
    minio_secret_key: str
):
    import pandas as pd

    from minio import Minio
    from tempfile import NamedTemporaryFile


    print("[*] Deploying new model and new data")

    client = Minio(
        f'host.k3d.internal:{minio_port}',
        minio_access_key,
        minio_secret_key,
        secure=False
    )

    def put_on_minio(local_path, s3_location):
        result = client.fput_object(
            *s3_location.split('/', 1), local_path,
        )
        print(f"[*] Created {result.object_name} object; etag: {result.etag}, version-id: {result.version_id}")

    _user_data = pd.read_csv(user_data.path)
    _base_data = pd.read_csv(base_data.path)

    new_baseline_data = pd.concat((_base_data, _user_data)) 

    new_baseline_data_path = NamedTemporaryFile().name

    new_baseline_data.to_csv(new_baseline_data_path, index=False)

    put_on_minio(trained_model.path, baseline_model_loc)
    put_on_minio(new_baseline_data_path, baseline_data_loc)


deploy_component = partial(
    _deploy_component,
    minio_port=MinioConfig.PORT,
    minio_access_key=MinioConfig.ACCESS_KEY,
    minio_secret_key=MinioConfig.SECRET_KEY
)
