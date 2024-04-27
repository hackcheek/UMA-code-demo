from pipes_code.src.utils.schemas.pipeline import PipelineConfig
from dataclasses import dataclass


class Config:
    @classmethod
    def asdict(cls) -> dict:
        return {
            i:j 
            for i, j in vars(cls).items()
            if not i.startswith('__') 
                and not i.endswith('__') 
                and not isinstance(j, classmethod)
        }


class DiamondsDatasetConfig(Config):
    BASELINE_DATASET_PATH: str = "pipes_code/src/datasets/actual_dataset.csv"
    TARGET: str = "price"
    NUMERICAL_FEATURES: list[str] = ["x", "carat"]
    CATEGORICAL_FEATURES: list[str] = ["color", "clarity", "cut"]
    TRAINING_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    COLOR_LABELS: list[str] = ['H', 'I', 'F', 'G', 'E', 'D', 'J']
    CUT_LABELS: list[str] = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
    CLARITY_LABELS: list[str] = ['SI2', 'SI1', 'VS2', 'IF', 'VVS2', 'VS1', 'I1', 'VVS1']
    POSITIVE_DATA: list[str] = ['price', 'x']
    ZSCORE_WITH_THRESHOLD_4 = ['table', 'depth', 'carat']
    ZSCORE_WITH_THRESHOLD_3 = ['x', 'y', 'z', 'price']


class MinioConfig(Config):
    HOST = 'localhost'
    PORT = 9000
    ACCESS_KEY = 'minio'
    SECRET_KEY = 'minio123'


@dataclass
class ExamplePipelineConfig(PipelineConfig):
    num_epochs: int = 5
    output_model_path: str = 'pipes_code/bucket/deployed_model.pt'
    output_data_path: str = 'pipes_code/bucket/data.csv'
