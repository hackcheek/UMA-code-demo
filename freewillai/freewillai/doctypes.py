from typing import Dict, Tuple, TypeAlias, TypedDict, Union, List, Optional
from transformers import (
    PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel, TFPreTrainedModel
)
from torch.nn import Module as TorchNNModule
from keras import Model as KerasModel
from sklearn.base import BaseEstimator
from onnx import ModelProto
from web3.types import Middleware as Web3Middleware
from transformers import (
    PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel, TFPreTrainedModel
)
from torch.nn import Module as TorchNNModule
from keras import Model as KerasModel
from sklearn.base import BaseEstimator
from onnx import ModelProto


Stderr = Union[bytes, str, None]
Stdout = Union[bytes, str, None]

ModelName = str
DatasetName = str

ModelName = str
DatasetName = str


class AbiDict(TypedDict):
    inputs: List[Dict[str, str]]
    name: str
    outputs: List[Dict[str, str]]
    stateMutability: str
    type: str


class AddedFile(TypedDict):
    Name: str
    Hash: str
    Size: str


class RunningConfig(TypedDict):
    preprocess: Optional[Dict]
    modellib: str
    input_size: Optional[Tuple[int, ...]]
    result_format: str
    model_kwargs: Dict
    tokenizer_method: Optional[str]
    tokenizer_kwargs: Dict
    huggingface_model_class: Optional[str]
    huggingface_model_method: Optional[str]
    generative_model: Optional[bool]
    generative_model_args: Optional[Dict]
    tokenizer: bool


Abi = List[AbiDict]
Bytecode = str

BaseHuggingFaceModel = Union[PreTrainedModel, TFPreTrainedModel]
BasePytorchModel = TorchNNModule
BaseKerasModel = KerasModel
BaseSklearnModel = BaseEstimator
BaseONNXModel = ModelProto
BaseTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Maybe use our own middlewares
Middleware: TypeAlias = Web3Middleware
