from pipes_code.src.configs import ExamplePipelineConfig
from pipes_code.src.local.pipelines import DiamondsLocalPipeline
from pipes_code.src.utils.schemas.pipeline import DiamondsPipelineInput


if __name__ == "__main__":
    input = DiamondsPipelineInput(
        base_data_path = 'datasets/diamonds/diamonds.csv',
        user_data_path = 'datasets/diamonds/new_records.csv',
        base_model_path = 'modeling/best_model.pt',
    )
    DiamondsLocalPipeline(input, ExamplePipelineConfig(num_epochs=5)).run()
