from kfp import dsl
from functools import partial

from pipes_code.src.configs import DiamondsDatasetConfig


@dsl.component(base_image='python:3.10', packages_to_install=['pandas', 'torch', 'numpy'])
def _pytorch_model_train_component(
    train_dataset: dsl.Input[dsl.Dataset],
    val_dataset: dsl.Input[dsl.Dataset],
    num_epochs: int,
    dataset_cfg: dict,
    trained_model: dsl.Output[dsl.Model]
):
    import pandas as pd
    import numpy as np
    import torch
    import os
    import json
    import tempfile

    from torch.utils.data import DataLoader, Dataset


    checkpoints_directory = tempfile.TemporaryDirectory().name


    class RegressionModel(torch.nn.Module):
        def __init__(self, cat_features_amount, embedding_dim, num_features_amount):
            super(RegressionModel, self).__init__()
            self.embedding = torch.nn.Embedding(
                num_embeddings=cat_features_amount,
                embedding_dim=embedding_dim
            )

            self.input_size = (embedding_dim * cat_features_amount) + num_features_amount
            self.layer1 = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, 512),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(512),
                torch.nn.Dropout(0.5)
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(256),
                torch.nn.Dropout(0.3)
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(128),
                torch.nn.Dropout(0.3)
            )
            self.layer4 = torch.nn.Sequential(
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(64),
                torch.nn.Dropout(0.3)
            )
            self.output = torch.nn.Linear(64, 1)

        def forward(self, x_cat, x_num):
            x_cat = self.embedding(x_cat)
            x_cat = x_cat.view(x_cat.size(0), -1)
            x = torch.cat([x_cat, x_num], dim=1)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.output(x)
            return x


    class DiamondsPytorchDataset(Dataset):
        def __init__(self, data, cat_features, num_features, label):
            self.data = data
            self.cat_features = cat_features
            self.num_features = num_features
            self.label = label

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            x_num = torch.from_numpy(
                row[self.num_features].to_numpy().astype(np.float32)
            ).view(-1)

            x_cat = torch.from_numpy(
                row[self.cat_features].to_numpy().astype(np.int32)
            ).view(-1)

            target = torch.from_numpy(row[[self.label]].to_numpy().astype(np.float32))
            return x_cat, x_num, target


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


    _train_dataset = pd.read_csv(train_dataset.path) 
    _val_dataset = pd.read_csv(val_dataset.path) 

    cat_features = list(filter(
        lambda x: x not in dataset_cfg['NUMERICAL_FEATURES'] + [dataset_cfg['TARGET']],
        _train_dataset.columns
    ))

    _train_dataset = DiamondsPytorchDataset(
        _train_dataset,
        cat_features,
        dataset_cfg['NUMERICAL_FEATURES'],
        dataset_cfg['TARGET']
    )

    _val_dataset = DiamondsPytorchDataset(
        _val_dataset,
        cat_features,
        dataset_cfg['NUMERICAL_FEATURES'],
        dataset_cfg['TARGET']
    )

    train_loader = DataLoader(_train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(_val_dataset, batch_size=1024, shuffle=True)

    embedding_dim = 3
    model = RegressionModel(
        len(cat_features),
        embedding_dim,
        len(dataset_cfg['NUMERICAL_FEATURES'])
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
        num_epochs,
    )

    best_weights_path = os.path.join(checkpoints_directory, f'epoch_{best_epoch}', 'weights.pth')
    model.load_state_dict(torch.load(best_weights_path))
    torch.jit.script(model).save(trained_model.path)



pytorch_model_train_component = partial(
    _pytorch_model_train_component,
    dataset_cfg=DiamondsDatasetConfig.asdict()
)
