import pandas as pd
import torch
import os
import json
import tempfile

from pipes_code.src.configs import ExamplePipelineConfig, DiamondsDatasetConfig
from pipes_code.src.local.components.train.base import TrainComponent
from pipes_code.src.utils.datasets.diamonds import DiamondsPytorchDataset
from pipes_code.src.utils.metadata.artifacts.model import ModelArtifact
from pipes_code.src.utils.metadata.executions.base import Execution
from pipes_code.src.utils.models.pytorch import RegressionModel
from pipes_code.src.utils.schemas.pipeline import TrainInput, TrainOutput

from torch.utils.data import DataLoader


checkpoints_directory = tempfile.TemporaryDirectory().name


def training_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    train_dataset_length,
    val_dataset_length,
    checkpoints_directory,
    num_epochs,
):
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_cat, x_num, y in train_loader:
            optimizer.zero_grad()
            pred = model(x_cat, x_num)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_cat.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, y in val_loader:
                pred = model(x_cat, x_num)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * x_cat.size(0)

        train_loss = train_loss / train_dataset_length
        val_loss = val_loss / val_dataset_length

        save_dir = os.path.join(checkpoints_directory, f'epoch_{epoch + 1}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        weights_path = os.path.join(save_dir, 'weights.pth')
        torch.save(model.state_dict(), weights_path)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1

        metadata = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
        }

        with open(os.path.join(save_dir, 'metadata.json'), 'w') as mfile:
            json.dump(metadata, mfile)

        print(f'Epoch {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

    return best_epoch


class DiamondsTrain(TrainComponent):
    cfg: ExamplePipelineConfig

    @classmethod
    def do_train(cls, input: TrainInput, ctx: Execution) -> TrainOutput:
        _train_dataset = pd.read_csv(input.train_dataset.uri) 
        _val_dataset = pd.read_csv(input.val_dataset.uri) 

        cat_features = list(filter(
            lambda x: x not in DiamondsDatasetConfig.NUMERICAL_FEATURES
            + [DiamondsDatasetConfig.TARGET],
            _train_dataset.columns
        ))

        _train_dataset = DiamondsPytorchDataset(
            _train_dataset,
            cat_features,
            DiamondsDatasetConfig.NUMERICAL_FEATURES,
            DiamondsDatasetConfig.TARGET
        )

        _val_dataset = DiamondsPytorchDataset(
            _val_dataset,
            cat_features,
            DiamondsDatasetConfig.NUMERICAL_FEATURES,
            DiamondsDatasetConfig.TARGET
        )

        train_loader = DataLoader(_train_dataset, batch_size=1024, shuffle=True)
        val_loader = DataLoader(_val_dataset, batch_size=1024, shuffle=True)

        embedding_dim = 3
        model = RegressionModel(
            len(cat_features),
            embedding_dim,
            len(DiamondsDatasetConfig.NUMERICAL_FEATURES)
        )
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("[*] Starting training loop")

        best_epoch = training_loop(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            len(_train_dataset),
            len(_val_dataset),
            checkpoints_directory,
            cls.cfg.num_epochs,
        )

        best_weights_path = os.path.join(checkpoints_directory, f'epoch_{best_epoch}', 'weights.pth')
        model.load_state_dict(torch.load(best_weights_path))
        
        trained_model_op = ModelArtifact('trained_model')
         
        torch.jit.script(model).save(trained_model_op.uri)

        return TrainOutput(trained_model_op)
