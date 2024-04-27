import pickle
import os
import numpy as np
from freewillai.common import NonTuringModel, FWAIModel, FWAIDataset


def sklearn_model_dataset():
    filename = 'demos/python/sklearn/model_prediction_houses_sklearn2.pkl'
    model = pickle.load(open(filename, 'rb'))
    data = {
        "bedrooms": 3, 
        "bathrooms": 1,
        "sqft_living": 200,
        "sqft_lot": 3362,
        "floors": 5,
        "waterfront": 1,   # 0 or 1
        "view": 0,         # 0 or 1
        "condition": 1,    # 5 is best, 1 is worst
        "sqft_above": 1050,
        "sqft_basement": 90,
        "yr_built": 2023
    }

    data = np.array([list(data.values())])
    return model, data


class NonTuringModelFromPath(NonTuringModel):
    def __init__(self, path):
        self.path = path
        self.lib = 'onnx'
    

def inference_setup():
    onnx_path = '/tmp/model.onnx'
    sk_model, sk_dataset = sklearn_model_dataset()
    fw_dataset = FWAIDataset(sk_dataset) 
    FWAIModel(sk_model, fw_dataset.format()).save(onnx_path)
    onnx_model = NonTuringModelFromPath(onnx_path)
    print('{HERERERERRERRE}')
    # os.remove(onnx_path) 
    return onnx_model, sk_dataset


class TestOnnxInference:
    def test_onnx_on_cpu(self):
        onnx_model, dataset = inference_setup()
        result = onnx_model.inference(dataset)
        print(result)

    def test_onnx_on_gpu(self):
        os.environ['GPU'] = '1'
        onnx_model, dataset = inference_setup()
        result = onnx_model.inference(dataset)
        print(result)
