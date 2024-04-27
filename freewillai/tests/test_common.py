import torch
import pickle
import keras
import numpy as np
import tensorflow as tf
from torch.nn import functional as F
from freewillai.common import FWAIDataset, FWAIModel
from transformers import AutoModelForCausalLM


class TestModelWrapper:
    def test_torch(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
                self.fc2 = torch.nn.Linear(120, 84)
                self.fc3 = torch.nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = Net()
        state_dict = torch.load('bucket/test/models/cifar_net.pth')
        model.load_state_dict(state_dict)
        assert 'torch' == FWAIModel.by_model(model).name()

    def test_keras(self):
        model_path = 'bucket/test/models/keras_model_dnn/'
        model = keras.models.load_model(model_path)
        assert 'keras' == FWAIModel.by_model(model).name()

    def test_sklearn(self):
        model_path = 'bucket/test/models/sklearn_model.pkl'
        model =  pickle.load(open(model_path, "rb"))
        assert 'sklearn' == FWAIModel.by_model(model).name()

    def test_huggingface(self):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        assert 'huggingface' == FWAIModel.by_model(model).name()


class TestDatasetWrapper:
    def test_json(self):
        path = "bucket/test/datasets/test.json"
        assert FWAIDataset.by_dataset(path).format() == 'json'

    def test_csv(self):
        path = "bucket/test/datasets/dummy_data_set_sklearn.csv"
        assert FWAIDataset.by_dataset(path).format() == 'csv'

    def test_image(self):
        path = "bucket/test/datasets/cat.png"
        assert FWAIDataset.by_dataset(path).format() == 'image'

    def test_numpy(self):
        path = "bucket/test/datasets/cat_img_pytorch.npy"
        array = np.load(path)
        assert FWAIDataset.by_dataset(array).format() == 'numpy'

    def test_torch(self):
        path = "bucket/test/datasets/cat_img_pytorch.npy"
        array = np.load(path)
        tensor = torch.from_numpy(array)
        assert FWAIDataset.by_dataset(tensor).format() == 'torch'

    def test_tensorflow(self):
        path = "bucket/test/datasets/cat_img_pytorch.npy"
        array = np.load(path)
        tensor = tf.convert_to_tensor(array)
        assert FWAIDataset.by_dataset(tensor).format() == 'tensorflow'

    def test_text(self):
        text = "testing..."
        assert FWAIDataset.by_dataset(text).format() == "text"
