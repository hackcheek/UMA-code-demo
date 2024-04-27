from kfp import Client, local, dsl
from kfp.compiler import Compiler
from typing import NamedTuple

from pipes_code.src.kubeflow.components.ingest import ingest_csv_component
from pipes_code.src.kubeflow.components.preprocess import preprocess_component
from pipes_code.src.kubeflow.components.train import pytorch_model_train_component
from pipes_code.src.kubeflow.components.evaluation import pytorch_model_evaluation_component
from pipes_code.src.kubeflow.components.deploy import deploy_component
from pipes_code.src.utils.schemas.report import PipelineReport
from pipes_code.src.configs import MinioConfig


local.init(runner=local.DockerRunner())


@dsl.pipeline(
    name='Diamonds pipeline',
    description='The goal is run this pipeline every time that database is updated and \
    train the model with the new data',
)
def ch2_pipeline(
    csv_loc: str,
    baseline_model_loc: str,
    baseline_data_loc: str,
) -> PipelineReport:
    ingest_op = ingest_csv_component(
        csv_loc=csv_loc,
        baseline_model_loc=baseline_model_loc,
        baseline_data_loc=baseline_data_loc,
    )

    user_data = ingest_op.outputs.get('user_data')
    base_data = ingest_op.outputs.get('base_data')
    base_model = ingest_op.outputs.get('base_model')

    preprocess_op = preprocess_component(
        user_data=user_data,
        base_data=base_data
    ).after(ingest_op)

    train_dataset = preprocess_op.outputs.get('train_dataset')
    test_dataset = preprocess_op.outputs.get('test_dataset')
    val_dataset = preprocess_op.outputs.get('val_dataset')

    train_op = pytorch_model_train_component(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=1,   # 50
    ).after(preprocess_op)

    trained_model = train_op.outputs.get('trained_model')

    evaluation_op = pytorch_model_evaluation_component(
        model=trained_model,
        test_dataset=test_dataset,
        base_model=base_model,
    ).after(train_op)

    current_model_loss = evaluation_op.outputs.get('current_model_loss')
    base_model_loss = evaluation_op.outputs.get('base_model_loss')

    with dsl.If(current_model_loss < base_model_loss, 'Deploy if model is better'):
        deploy_component(
            trained_model=trained_model,
            user_data=user_data,
            base_data=base_data,
            baseline_model_loc=baseline_model_loc,
            baseline_data_loc=baseline_data_loc,
        )

    return PipelineReport(current_model_loss, base_model_loss)


if __name__ == "__main__":
    import subprocess
    import sys

    mode = sys.argv[1]

    # Copy csv file in minio bucket
    csv_base_path = 'datasets/diamonds/diamonds.csv'
    csv_path = 'datasets/diamonds/new_records.csv'
    baseline_model_path = 'modeling/base_model.pt'

    csv_base_loc = 'mlpipeline/local_data/baseline_data.csv'
    csv_loc = 'mlpipeline/local_data/user_data.csv'
    baseline_model_loc = 'mlpipeline/local_data/baseline_model.pt'

    csv_base_s3_uri = f's3://{csv_base_loc}'
    csv_s3_uri = f's3://{csv_loc}'
    baseline_model_s3_uri = f's3://{baseline_model_loc}'

    # Upload baseline model to bucket
    subprocess.run([
        'aws', '--endpoint-url', f'http://{MinioConfig.HOST}:{MinioConfig.PORT}',
        's3', 'cp', baseline_model_path, baseline_model_s3_uri
    ])

    # Upload baseline csv to bucket
    subprocess.run([
        'aws', '--endpoint-url', f'http://{MinioConfig.HOST}:{MinioConfig.PORT}',
        's3', 'cp', csv_base_path, csv_base_s3_uri
    ])

    # Upload new csv to bucket
    subprocess.run([
        'aws', '--endpoint-url', f'http://{MinioConfig.HOST}:{MinioConfig.PORT}',
        's3', 'cp', csv_path, csv_s3_uri
    ])

    if mode == 'run':
        print('[*] Pipeline Runing')
        client = Client()
        client.create_run_from_pipeline_func(
            ch2_pipeline,
            arguments={
                'csv_loc': csv_loc,
                'baseline_model_loc': baseline_model_loc,
                'baseline_data_loc': csv_base_loc
            },
            enable_caching=False,
            run_name='ch2_pipeline'
        )
    if mode == 'save':
        pipeline_path = 'pipes/pipeline.yaml'
        Compiler().compile(pipeline_func=ch2_pipeline, package_path=pipeline_path)
        print(f"[*] Pipeline save on {pipeline_path}")
