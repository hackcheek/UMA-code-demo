from __future__ import annotations

import json
import yaml
import time
import os
import onnx
import onnxruntime as rt
import numpy as np
import pandas as pd
import torch
import shutil
import tensorflow as tf
import polars as pl
import logging
import transformers
from freewillai.transforms import Transforms
from typing import Literal, Dict, Optional, Union, List, Type, cast, Tuple, Any, overload
from types import MethodType
from web3 import Web3, HTTPProvider, WebsocketProvider
from web3.middleware import geth_poa_middleware, buffered_gas_estimate_middleware
from web3.middleware.signing import construct_sign_and_send_raw_middleware
from web3.types import Timestamp
from dataclasses import dataclass, field
from dotenv import load_dotenv
from freewillai.doctypes import AddedFile, BaseHuggingFaceModel, BaseKerasModel, BaseONNXModel, BasePytorchModel, BaseSklearnModel, BaseTokenizer, RunningConfig
from freewillai.globals import Global
from freewillai.utils import (
    add_files, contract_exists, get_file, get_free_mem_of_gpu, is_ipfs_url, load_global_env,
    get_hash_from_url, get_path, get_w3 
)
from freewillai.exceptions import NotSupportedError, UnexpectedError, UserRequirement, Fatal
from freewillai.doctypes import AddedFile, Middleware
from PIL import Image as PILImage
from torchvision.transforms import ToTensor
from tensorflow import convert_to_tensor
from numpy import isin, ndarray
from csv import (
    Sniffer,
    writer as CSVWriter,
    DictReader as CSVDictReader,
    Error as CSVError
)
from tempfile import NamedTemporaryFile
from json.decoder import JSONDecodeError
from typing import List, Optional, Union
from typing_extensions import Literal
from eth_account.signers.local import LocalAccount
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, BatchEncoding, PreTrainedModel, TFPreTrainedModel
from functools import singledispatchmethod, partial
from io import BytesIO, StringIO


@dataclass
class IPFSFile:
    cid: str
    size: int
    is_dir: bool
    type: Optional[str] = None


class IPFSBucket:
    @staticmethod
    async def upload(
        path: str, 
        file_type: Optional[Literal['model', 'dataset', 'result']] = None
    ) -> IPFSFile:
        """
        Upload only one file/directory to IPFS
        IMPORTANT: only one file or directory

        If you want to upload multiple files or directories please use async
        Example:
        >>> await IPFSBucket.upload("path/to/model", 'dataset')
        >>> await IPFSBucket.upload("path/to/dataset", 'model')
        >>> await IPFSBucket.upload("path/to/result", 'result')
        >>> await IPFSBucket.upload("path/to/some_file")
        """        
        added_files: List[AddedFile] = await add_files([path])
        is_dir = False if len(added_files) == 1 else True
        ipfs_file = added_files[0]

        if file_type:
            print(f'\n[*] Uploading {file_type}...')

        if is_dir:
            for added_file in added_files:
                if '/' in added_file['Name']:
                    continue
                ipfs_file = added_file

        cid = ipfs_file['Hash']
        size = int(ipfs_file['Size'])

        print(f"  > cid = {cid}")
        return IPFSFile(cid=cid, is_dir=is_dir, type=file_type, size=size)


    @classmethod
    async def download(
        cls,
        cid_or_url: str, 
        file_type: Optional[Literal['model', 'dataset', 'result']] = None
    ) -> str:
        f"""
        Download file or directory from ipfs to {Global.working_directory}
        """
        cid = cid_or_url
        if 'ipfs' in cid_or_url:
            cid = get_hash_from_url(cid_or_url)

        if file_type:
            print(
                f'\n[*] Downloading {file_type}...\n'
                f'  > cid = {cid}'
            )

        out_path = await get_file(cid)
        print(f"  > Downloaded {cid}")
        return out_path


class FileDetector:
    """
    Functions for file type detection
    IMPORTANT: Prioritize speed
    """ 
    prompt_start = b"prompt:"

    @staticmethod
    def is_numpy(path: str) -> bool:
        """
        Detect if a file is a numpy (.npz or .npy)
        
        How it works
        ------------
        Read the first bytes and if startswith any of the valid numpy signatures
        return True else False
        """
        # if you read the first line of numpy array you should see something like
        # \x93NUMPY\x01\x00v\x00{'descr': '<f4', 'fortran_order': False...
        # So these first 6 bytes we use to detect if it's numpy without load all file
        # Added b'PK' to detect npz files
        numpy_signatures = (
            b'\x93NUMPY',  # npy
            b'PK',  # npz
        )
        with open(path, 'br') as file:
            first_bytes = file.read(6)
            for signature in numpy_signatures:
                if first_bytes.startswith(signature):
                    return True
            return False


    @staticmethod
    def is_image(path) -> bool:
        """
        Detect if a file is an image
        
        How it works
        ------------
        Read the first bytes and if startswith any of the valid image signatures
        return True else False
        """
        images_signatures = (
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'\xFF\xD8',  # JPEG
        )
        with open(path, "br") as file:
            # 8 is the max num of bytes in the signatures list
            first_bytes = file.read(8)
            for signature in images_signatures:
                if first_bytes.startswith(signature):
                    return True
            return False


    @classmethod
    def is_csv(cls, path) -> bool:
        """
        Detect if a file is a csv
        
        How it works
        ------------
        Read some of the first bytes trying detect the csv delimiter
        if the delimiter is detected return True else False
        """
        # 4096 should be enough to detect any delimiter thus detect if is csv
        # You can add more num of bytes if your csv is not detected. But 
        # notified this change please
        bytes_to_read = 4096
        with open(path, 'br') as file:
            first_bytes = file.read(bytes_to_read)
            if first_bytes.startswith(cls.prompt_start):
                return False
            try:
                readeable_buffer = str(first_bytes, 'utf-8')
                Sniffer().sniff(readeable_buffer)
                return True
            # except (CSVError, UnicodeDecodeError) as err:
            except:
                return False


    @classmethod
    def is_prompt(cls, path) -> bool:
        with open(path, 'br')  as file:
            return file.read(10).startswith(cls.prompt_start)


    @staticmethod
    def old_is_json(path) -> bool:
        """
        Old detector deprecated for now, use optimistic detector at
        FileDetector.is_json instead

        How it works
        ------------
        Read whole file trying load the json 
        if it is loaded return True else False

        NOTE: It's slow because json needs load entire file because 
        pair of brackets "[{}]"
        """
        with open(path, 'br') as file:
            try:
                json.loads(file.read())
                return True
            except (JSONDecodeError, ValueError, UnicodeDecodeError):
                return False


    @staticmethod
    def is_json(path) -> bool:
        """
        Optimistic json file detector

        How it works
        ------------
        Read the first bytes, if starts with [{ or { return True else False
        """
        with open(path, "br") as file:
            return (
                file.read(24)
                .replace(b' ', b'')
                .replace(b'\n', b'')
                .startswith((b'[{', b'{'))
            )


    @staticmethod
    def is_yaml(path) -> bool:
        """
        YAML file detector

        How it works
        ------------
        Read the file and try to load with PyYaml.
        Returns True if works else False
        (this detector is used just for config files)
        """
        with open(path, 'r') as yamlfile:
            try:
                yaml.safe_load(yamlfile)
                return True
            except yaml.YAMLError:
                return False


    @staticmethod
    def is_huggingface(path) -> bool:
        """
        Huggingface directory detector

        How it works
        ------------
        Asserts that path is a dir and if it 
        contains config.json and pytorch_model.bin or tf_model.h5
        """
        if not os.path.isdir(path):
            return False
        files = os.listdir(path)
        if (not "config.json" in files 
            and (not "pytorch_model.bin" in files
                 or not "tf_model.h5" in files)
        ):
            return False
        return True


@dataclass
class ModelConfigs:
    base: str
    input_format: str
    result_format: str = 'csv'
    tokenizer: bool = False
    preprocess: Dict = field(default_factory=dict)
    model_kwargs: Dict = field(default_factory=dict)
    tokenizer_kwargs: Dict = field(default_factory=dict)
    tokenizer_method: Optional[str] = None
    input_size: Optional[Tuple[int, ...]] = None
    huggingface_model_class: Optional[str] = None
    huggingface_model_method: Optional[str] = None
    generative_model: Optional[bool] = None
    generative_model_args: Optional[Dict] = None
    
    def dict(self):
        return self.__dict__


class FWAIDataset:
    def __new__(cls, dataset: Any, *_, **__):
        """SubClass selector"""
        subclass = cls.by_dataset(dataset)
        return object.__new__(subclass)

    def __init__(self, dataset: Any):
        self.data = dataset
    

    @classmethod
    def format(cls) -> str:
        raise NotImplementedError


    @classmethod
    @property
    def supported_formats(cls):
        """Get valid dataset formats"""
        lst = []
        for subclass in cls.__subclasses__():
            format = subclass.format()
            if isinstance(format, List):
                lst.extend(format)
                continue
            lst.append(format)
        return lst


    @classmethod
    def by_format(cls, format: str) -> Type[FWAIDataset]:
        """Get model class by format"""
        format = format.lower()
        if format == cls.format():
            return cls
        for subclass in cls.__subclasses__():
            if subclass.format() == format or format in subclass.format():
                return subclass
        raise RuntimeError(
            f"{format=} does not match with any dataset format. "
            f"Available formats name = {cls.supported_formats}"
        )


    @classmethod
    def by_dataset(cls, dataset: Any):
        """Get dataset saver by provided dataset""" 
        for subclass in cls.__subclasses__():
            if subclass.match(dataset):
                return subclass
        raise NotSupportedError(
            f'{type(dataset)=}. This dataset type is not supported yet. '
            f'Please open an issue on github if it is required'
        )

    
    @classmethod
    def match(cls, data) -> bool:
        raise NotImplementedError


    @staticmethod
    def encode_metadata(metadata: Dict) -> bytes:
        """
        Replaced by file detectors
        Maybe use in future

        Encode metadata dictionary to a human readeable string line
        """
        encoded_string = ' '.join(
            f"{k}={v}" for k, v in metadata.items()
        ) + '\n'
        return bytes(encoded_string, 'utf-8')


    @staticmethod
    def decode_metadata(encoded_metadata: Union[str, bytes]) -> Dict:
        """
        Replaced by file detectors
        Maybe use in future

        Decode metadata string to a dictionary
        """
        if isinstance(encoded_metadata, bytes):
            encoded_metadata = str(encoded_metadata, 'utf-8').strip()
        meta = {}
        for pair in encoded_metadata.strip().split(' '):
            key, _, value = pair.partition('=')
            meta[key] = value
        return meta


    @classmethod
    def add_metadata(cls, path: str, metadata: Dict):
        """
        Replaced by file detectors
        Maybe use in future
        """
        encoded_metadata = cls.encode_metadata(metadata)
        with open(path, "br") as file:
            img = file.read()

        with open(path + ".out", "wb") as file:
            img = file.write(encoded_metadata + img)

    
    @classmethod
    def parse_and_split_metadata(cls, path: str) -> Tuple[Dict, BytesIO]:
        """
        Replaced by file detectors
        Maybe use in future
        """
        with open(path, "br") as file:
            metadata = cls.decode_metadata(file.readline())
            data = file.read()
        buffered_data = BytesIO(data)
        return metadata, buffered_data

    
    @classmethod
    def load(cls, ipfs_url: str) -> ndarray:
        assert is_ipfs_url(ipfs_url), f"{ipfs_url=} must be a valid ipfs url"
        path = get_path(ipfs_url)
        assert os.path.exists(path), f"{ipfs_url=} has not been downloaded yet"
        dataset = cls.by_dataset(path)
        return dataset.loader(path)


    @classmethod
    def loader(cls, path: str) -> ndarray:
        """Load file as numpy array"""
        raise NotImplementedError
        

    def saver(self, dest_path: str) -> None:
        raise NotSupportedError(
            f'{type(self.data)=}. This dataset type is not supported yet. '
            f'Please open an issue on github if it is required'
        )
    
    def save(self, dest_path: Optional[str] = None) -> str:
        """
        Save dataset valid for FreWillAI

        Arguments
        ---------
        dest_path: Optional string
            Path to write dataset.
            If it's not setted, the dataset will be wrote in a temp file

        Returns
        -------
        String:
            Path to saved dataset
        """
        dest_path = dest_path or NamedTemporaryFile().name
        self.saver(dest_path)
        return dest_path 


class JsonDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "json"

    @classmethod
    def match(cls, data) -> bool:
        return (
            isinstance(data, str)
            and os.path.exists(data)
            and FileDetector.is_json(data)
        )

    def saver(self, dest_path: str) -> None:
        shutil.copyfile(self.data, dest_path)

    @classmethod
    def loader(cls, path: str) -> ndarray:
        # Taking for granted this json file is a list of rows
        # Example: 
        #   data = [
        #       {'user': 'Nicolas', "has_car": false},
        #       {'user': 'Felipe', "has_car": true},
        #       {'user': 'Lucas',  "has_car": false}
        #   ]
        return pl.read_json(path).to_numpy()


class CSVDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return 'csv'

    @classmethod
    def match(cls, data) -> bool:
        return (
            isinstance(data, str)
            and os.path.exists(data)
            and not FileDetector.is_prompt(data)
            and FileDetector.is_csv(data)
        )

    def saver(self, dest_path: str) -> None:
        shutil.copyfile(self.data, dest_path)

    @classmethod
    def loader(cls, path: str) -> ndarray:
        with open(path, 'r') as csvfile:
            content = csvfile.read(2048)
            delimiter = str(Sniffer().sniff(content).delimiter)
            skip_header = int(Sniffer().has_header(content))
            has_header = Sniffer().has_header(content)

        # Maybe it's better
        # return np.genfromtxt(self.path, delimiter=delimiter, skip_header=skip_header)

        lazy_data = pl.scan_csv(
            path, 
            separator=delimiter,
            has_header=has_header,
            infer_schema_length=0
        ).with_columns(pl.all().cast(pl.Float32))

        return lazy_data.collect().to_numpy()


class ImageDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "image"

    @classmethod
    def match(cls, data) -> bool:
        return (
            isinstance(data, str)
            and os.path.exists(data)
            and FileDetector.is_image(data)
        )

    def saver(self, dest_path: str) -> None:
        shutil.copyfile(self.data, dest_path)

    @classmethod
    def loader(cls, path: str) -> ndarray:
        image = PILImage.open(path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # TODO: Move it to NonTuringModel
        # image_array = self.transform_image(image, base)

        # add batch size dim
        dataset = np.expand_dims(np.array(image), 0)
        return dataset / 255


class NumpyPathDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "numpy/path"

    @classmethod
    def match(cls, data) -> bool:
        return (
            isinstance(data, str)
            and os.path.exists(data)
            and FileDetector.is_numpy(data)
        )


    def saver(self, dest_path: str) -> None:
        np.save(dest_path, self.data)
        shutil.move(dest_path + '.npy', dest_path)


    @classmethod
    def loader(cls, path: str) -> ndarray:
        return np.load(path)


class NumpyDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "numpy"

    @classmethod
    def match(cls, data) -> bool:
        return isinstance(data, ndarray)

    def saver(self, dest_path: str) -> None:
        np.save(dest_path, self.data)
        shutil.move(dest_path + '.npy', dest_path)

    @classmethod
    def loader(cls, path: str) -> ndarray:
        return np.load(path)


class TorchDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "torch"

    @classmethod
    def match(cls, data) -> bool:
        return isinstance(data, torch.Tensor)

    def saver(self, dest_path: str) -> None:
        """Save as numpy because node does not recognize Torch tensor"""
        np.save(dest_path, self.data.numpy())
        shutil.move(dest_path + '.npy', dest_path)

    @classmethod
    def loader(cls, path: str) -> ndarray:
        """Is in numpy class"""
        return np.load(path)


class TFDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "tensorflow"

    @classmethod
    def match(cls, data) -> bool:
        return isinstance(data, tf.Tensor)

    def saver(self, dest_path: str) -> None:
        """Save as numpy because node does not recognize TF tensor"""
        np.save(dest_path, self.data.numpy())
        shutil.move(dest_path + '.npy', dest_path)

    @classmethod
    def loader(cls, path: str) -> ndarray:
        """Is in numpy class"""
        return np.load(path)


class PILDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "PIL"

    @classmethod
    def match(cls, data) -> bool:
        return isinstance(data, PILImage.Image)

    def saver(self, dest_path: str) -> None:
        """Save as numpy because node does not recognize PIL tensor"""
        image_array = np.array(self.data)
        np.save(dest_path, image_array)
        shutil.move(dest_path + '.npy', dest_path)

    @classmethod
    def loader(cls, path: str) -> ndarray:
        """Is in numpy class"""
        return np.load(path)

    
class TextDataset(FWAIDataset):
    @classmethod
    def format(cls) -> str:
        return "text"

    @classmethod
    def match(cls, data) -> bool:
        return (
            isinstance(data, str)
             and (not os.path.exists(data)
                  or (os.path.exists(data) and FileDetector.is_prompt(data)))
        )

    def saver(self, dest_path: str) -> None:
        data = str(FileDetector.prompt_start, 'utf-8') + self.data
        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(data)            

    @classmethod
    def loader(cls, path: str) -> str:
        with open(path, 'r') as file:
            data =  file.read()[len(FileDetector.prompt_start):]
            return data
        

class NonTuringModel:
    def __init__(self, url: str, gpu:Optional[str]=None):
        assert is_ipfs_url(url), f"{url=} Must be an ipfs url"
        self.path = get_path(url)
        self.lib = self.get_non_turing_lib(self.path)
        self.gpu = gpu


    @staticmethod
    def get_non_turing_lib(
        path: str,
        raise_error: bool = True
    ) -> Union[str, None]:
        lib = "huggingface"
        if not FileDetector.is_huggingface(path):
            lib = "onnx"
            try:
                model = onnx.load(path)
                onnx.checker.check_model(model)
            except:
                if not raise_error:
                    return None
                raise ValueError("Is not a non-turing model")
        return lib


    def inference(self, dataset: Union[ndarray, str]):
        if self.lib == "onnx":
            assert isinstance(dataset, ndarray)
        else:
            assert isinstance(dataset, (ndarray, str))

        infer_by_lib = {
            "onnx": self.run_onnx_inference,
            "huggingface": self.run_huggingface_inference,
        }
        run_inference = infer_by_lib.get(self.lib) or False
        if not run_inference:
            raise NotSupportedError(f'{self.lib=} is not a valid')
        print(f"{dataset=}")
        return run_inference(dataset)


    def get_configs(self) -> ModelConfigs:
        if self.lib == "onnx":
            # see sess.get_metadata() to get metadata
            configs = {}
            for conf in onnx.load_model(self.path).metadata_props:
                if (conf.key == "preprocess" 
                    or conf.key == "input_size"
                    or conf.key == "tokenizer"
                ):
                    configs.update({conf.key: json.loads(conf.value)})
                    continue
                configs.update({conf.key: conf.value})

            ### TEMPORAL ISSUE
            # TODO: Allow tokenizer for all models
            if configs.get("tokenizer"):
                raise NotSupportedError(
                    "Tokenizer is only valid for huggingface models at moment"
                )

        elif self.lib == "huggingface":
            configs = AutoConfig.from_pretrained(
                self.path
            ).to_dict()["freewillai_metadata"]
        
        else:
            raise NotSupportedError(f"{self.lib=} not supported as non-turing")

        return ModelConfigs(**configs)


    def image_to_tensor(
        self,
        base_model: str,
        image: Union[PILImage.Image, ndarray],
    ) -> ndarray:
        return Transforms(base_model).image_to_tensor(image)


    def run_onnx_inference(self, dataset: ndarray):
        providers = ['CPUExecutionProvider']
        if self.gpu is not None:
            try:
                device_id = self.gpu.replace('cuda:', '').strip()
                free_mem = get_free_mem_of_gpu(self.gpu)
                print(f'[DEBUG] Free memory on {self.gpu}: {free_mem}')
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': device_id,
                        # 'arena_extend_strategy': 'kNextPowerOfTwo',
                        # 'gpu_mem_limit': free_mem,
                        # 'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ]
            except:
                print(f"[!] {self.gpu=} is not a valid gpu or system has not cuda")
        print(f"[DEBUG] onnx inference providers:", providers)
        inference_sess = rt.InferenceSession(self.path, providers=providers)
        input_name = inference_sess.get_inputs()[0].name
        label_name = inference_sess.get_outputs()[0].name
        
        configs = self.get_configs()

        if configs.input_format in ["image", "PIL"]:
            dataset = self.image_to_tensor(configs.base, dataset)
        preds = inference_sess.run(
            [label_name], {input_name: dataset.astype(np.float32)}
        )
        if isinstance(preds, list):
            preds = preds[0]

        return preds.astype(np.float32)


    def run_huggingface_inference(self, dataset: Union[ndarray, str]) -> str:
        tokenizer = base_tokenizer = None
        configs = self.get_configs()
        if configs.tokenizer:
            tokenizer = base_tokenizer = AutoTokenizer.from_pretrained(self.path)
            if method:=configs.tokenizer_method:
                tokenizer = getattr(base_tokenizer, method)

        model_class = configs.huggingface_model_class
        method_name = configs.huggingface_model_method

        # Assert to don't missing time
        assert model_class

        model = getattr(transformers, model_class).from_pretrained(self.path)
        model.eval()
        if method_name: 
            model = getattr(model, method_name)
        
        if tokenizer:
            tensor_class = 'tf' if isinstance(model, TFPreTrainedModel) else 'pt'
            _tokenizer_kwargs = {'return_tensors': tensor_class}
            _tokenizer_kwargs.update(configs.tokenizer_kwargs)
            dataset = tokenizer(
                dataset,
                **_tokenizer_kwargs
            )

        # PROBABLY THIS IS BAD
        ## TODO: Understand how to pass dataset to model
        if isinstance(dataset, (dict, BatchEncoding)):
            outputs = model(**dataset, **configs.model_kwargs)
        else:
            outputs = model(dataset, **configs.model_kwargs)

        if method_name == "generate":
            return base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # PROBABLY THIS IS BAD
            ## TODO: Understand how to send csv with all dataset
            if not outputs:
                raise UnexpectedError("Not results")
            
            with torch.no_grad():
                result = StringIO()
                array = outputs.to_tuple()[0].squeeze().numpy()
                pd.DataFrame(array).to_csv(result)
                return result.getvalue()


class FWAIModel:
    def __new__(cls, model: Any, *_, **__): 
        """SubClass selector"""
        # This is to do not load the model again after cls.by_model
        cls.cached_model = model
        subclass = cls.by_model(model)
        return object.__new__(subclass)

    def __init__(
        self, 
        model: Any, 
        input_format: str,
        tokenizer: Optional[Union[BaseTokenizer, str]] = None,
        input_size: Optional[Tuple[int, ...]] = None,
        preprocess: Optional[Dict] = None,
        model_kwargs: Dict = {},
        tokenizer_kwargs: Dict = {},
    ): 
        """
        Object to save and load models

        Arguments
        ---------
        ...
        """
        if not input_format in FWAIDataset.supported_formats:
            raise ValueError(
                f"{input_format=} is not supported. "
                f"Supported_formats = {FWAIDataset.supported_formats}"
            )
        self.model = self.cached_model or model
        self.input_format = input_format
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.preprocess = preprocess
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        if isinstance(self.model, str) and os.path.exists(self.model):
            self.model = self.from_path(self.model)

        if isinstance(self.tokenizer, str):
            ### TEMPORAL ISSUE
            # TODO: Allow tokenizer for all models
            if not self.name() == 'huggingface':
                raise NotSupportedError(
                    "Tokenizer is only valid for huggingface models at moment"
                )
                
            self.tokenizer = self.get_tokenizer_from_path(self.tokenizer)


        self.__post_init__()


    def __post_init__(self):
        """This is executed after init"""
        ...


    def to_non_turing(self, dest_path: str) -> Union[str, BaseONNXModel]:
        """
        Method that converts model to non-turing language

        Recommendations:
            - Use onnx

        Dev rules:
            - return onnx model or temp file with that onnx model
            - if returns string save in onnx format or in non-turing language
            - If you need return another datatypes 
              you will need handle it in self.add_metadata_and_save() to add metadata

        Returns
        -------
        string: Path where is the saved model
        or 
        BaseONNXModel: loaded onnx model 
        """
        raise NotImplementedError


    def _add_metadata_and_save_onnx(self, name: str, model: BaseONNXModel, out_path: str):
        model.metadata_props.extend([
            onnx.StringStringEntryProto(key="base", value=name),
            # Seems like input_size is always null/None
            # TODO: Test it
            onnx.StringStringEntryProto(key="input_format", value=self.input_format),
            onnx.StringStringEntryProto(key="input_size", value=json.dumps(self.input_size)),
            onnx.StringStringEntryProto(key="model_kwargs", value=json.dumps(self.model_kwargs)),
        ])
        if self.preprocess:
            model.metadata_props.append(
                onnx.StringStringEntryProto(
                    key="preprocess", value=json.dumps(self.preprocess)
                )
            )
        if self.tokenizer:
            model.metadata_props.extend([
                onnx.StringStringEntryProto(
                    key="tokenizer", value=json.dumps(True)
                ),
            ])
        onnx.save(model, out_path)


    def add_metadata_and_save(
        self,
        model: Union[BaseHuggingFaceModel, BaseONNXModel, str],
        out_path: Optional[str] = None
    ) -> None:
        out_path = out_path or NamedTemporaryFile().name
        name = self.name()
        if isinstance(name, list):
            name = name[0]
        if isinstance(model, BaseONNXModel):
            self._add_metadata_and_save_onnx(name, model, out_path)
        elif isinstance(model, str):
            # If model is string must be a onnx path
            self._add_metadata_and_save_onnx(name, onnx.load(model), out_path)
        else:
            raise NotSupportedError


    def save(self, dest_path: Optional[str] = None) -> str:
        """
        Save non-turing model with metadata included

        Arguments
        ---------
        path: Optional String. 
            Path where the model is to be saved
            default value is temp file

        Returns
        -------
        String: path where is saved the non-turing model
        """
        dest_path = dest_path or NamedTemporaryFile().name
        non_turing_path = NamedTemporaryFile().name
        self.add_metadata_and_save(
            self.to_non_turing(non_turing_path),
            dest_path
        )
        if os.path.exists(non_turing_path) and os.path.isdir(non_turing_path):
            os.rmdir(non_turing_path)
        elif os.path.exists(non_turing_path):
            os.remove(non_turing_path)
        return dest_path


    @classmethod
    def load(cls, ipfs_url: str, gpu:Optional[str]=None) -> NonTuringModel:
        return NonTuringModel(ipfs_url, gpu)


    @classmethod
    def from_path(cls, model_path: str):
        raise NotSupportedError(
            f"Get model from path={model_path} is not supported by {cls}. "
            f"Please pass the alredy loaded model instead of string as argument"
        )


    @classmethod
    def get_tokenizer_from_path(cls, tokenizer_path: str) -> BaseTokenizer:
        return AutoTokenizer.from_pretrained(tokenizer_path)


    @classmethod
    def name(cls) -> str:
        return 'base'


    @classmethod
    @property
    def available_models(cls):
        """Get valid models name"""
        return [subclass.name() for subclass in cls.__subclasses__()]


    @classmethod
    def by_model_name(cls, name: str) -> Type[FWAIModel]:
        """Get model class by name"""
        name = name.lower()
        if name == cls.name():
            return cls
        for subclass in cls.__subclasses__():
            if subclass.name() == name or name in subclass.name():
                return subclass
        raise RuntimeError(
            f"{name=} does not match with any model. "
            f"Available models name = {cls.available_models}"
        )


    @classmethod
    def get_model_from_path(cls, model_path: str):
        """
        Get model from path
        NOTE: That is not recommended. Is slow. Please provide the model
        TODO: Develop a FileDetector functions to detect models in path
        """
        for subclass in cls.__subclasses__():
            try:
                return subclass.from_path(model_path)
            except: ...
        raise NotSupportedError(
            f"from_path is not supported for this model filetype {model_path}. "
            "Please pass a loaded model"
        )
         

    @classmethod
    def by_model(cls, model):
        """Get model class by model type""" 
        if isinstance(model, partial):
            model = model.func
        elif isinstance(model, str) and os.path.exists(model):
            model = cls.get_model_from_path(model)
        cls.cached_model = model
        for subclass in cls.__subclasses__():
            if subclass.match(model):
                return subclass
        raise NotSupportedError(
            f"{model=} does not match with any model. "
            f"Available models = {cls.available_models}"
        )

    
    @classmethod
    def match(cls, model) -> bool:
        raise NotImplementedError

    
    @classmethod
    def image_to_tensor(cls, image: Union[PILImage.Image, ndarray]):
        """Basic pipeline where the image turn in model lib requiered format"""
        return np.array(image)


class HuggingFaceSaver(FWAIModel):
    @classmethod
    def name(cls) -> str:
        return 'huggingface'


    @classmethod
    def match(cls, model) -> bool:
        if isinstance(model, partial):
            model = model.func
        return (
            # is BaseHuggingFaceModel
            isinstance(model, BaseHuggingFaceModel) 

            # or is a method of BaseHuggingFaceModel
            or (isinstance(model, MethodType)
                and isinstance(model.__self__, BaseHuggingFaceModel))
            
            or 'transformers' in repr(type(model))
        )


    @classmethod
    def from_path(cls, model_path: str) -> BaseHuggingFaceModel:
        return AutoModel.from_pretrained(model_path)


    def to_non_turing(self, dest_path: str) -> BaseHuggingFaceModel:
        return self.model


    def model_class_parser(self, model):
        if isinstance(model, MethodType):
            # Get class name of self variable in method
            model_class = model.__self__.config.to_dict()['architectures'][0]
            method_name = model.__name__
        else:
            model_class = model.config.to_dict()['architectures'][0]
            method_name = None
        return model_class, method_name


    def tokenizer_parse_and_save(
        self, 
        tokenizer,
        metadata: ModelConfigs,
        out_path: str,
        ) -> ModelConfigs:
        if isinstance(tokenizer, BaseTokenizer):
            metadata.tokenizer = True
            tokenizer.save_pretrained(out_path)

        elif (
            isinstance(tokenizer, MethodType) 
            and isinstance(tokenizer.__self__, BaseTokenizer)
        ):
            metadata.tokenizer = True
            metadata.tokenizer_method = tokenizer.__name__
            tokenizer.__self__.save_pretrained(out_path)

        return metadata


    def add_metadata_and_save(
        self, 
        model: BaseHuggingFaceModel, 
        out_path: str,
    ) -> str:
        if isinstance(model, partial):
            self.model_kwargs = self.model.keywords
            model = model.func

        model_class, model_method = self.model_class_parser(model)

        name = self.name()
        if isinstance(name, list):
            name = name[0]

        metadata = ModelConfigs(
            base = name,
            input_format=self.input_format,
            input_size = self.input_size,
            result_format = "csv",
            preprocess = self.preprocess,
            tokenizer = False,
            tokenizer_method = None,
            model_kwargs = self.model_kwargs,
            tokenizer_kwargs = self.tokenizer_kwargs,
            huggingface_model_class = model_class,
            huggingface_model_method = model_method,
        )

        if isinstance(self.tokenizer, partial):
            metadata = self.tokenizer_parse_and_save(
                self.tokenizer.func, metadata, out_path
            )
            metadata.tokenizer_kwargs = self.tokenizer.keywords
        else:
            metadata = self.tokenizer_parse_and_save(
                self.tokenizer, metadata, out_path
            )

        if model_method == "generate":
            metadata.result_format = "string"

            if not metadata.tokenizer:
                raise UserRequirement(
                    f"{model_class}.{model_method} requires a tokenizer"
                )

        if model_method:
            model.__self__.config.update({"freewillai_metadata": metadata.dict()})
            model.__self__.save_pretrained(out_path)
        else:
            model.config.update({"freewillai_metadata": metadata.dict()})
            model.save_pretrained(out_path)
        return out_path


class PytorchSaver(FWAIModel):
    def __post_init__(self):
        if self.input_size is None:
            raise UserRequirement(
                f'{self.input_size=}. Torch model needs input_size argument.'
            )

    @classmethod
    def name(cls) -> str:
        return 'torch'

    @classmethod
    def match(cls, model) -> bool:
        return isinstance(model, BasePytorchModel) or 'torch' in repr(type(model))

    @classmethod
    def image_to_tensor(cls, image: Union[PILImage.Image, ndarray]):
        return ToTensor()(image)

    def to_non_turing(self, dest_path: str) -> str:
        logging.debug(f'Writing the onnx model to {dest_path}')
        dummy_input = torch.randn(*self.input_size, requires_grad=True)
        self.model.eval()
        torch.onnx.export(
            self.model, 
            dummy_input, 
            dest_path, 
            input_names = ['input'], 
            output_names = ['output'], 
            dynamic_axes={
                'input' : {0 : 'batch_size'}, 
                'output' : {0 : 'batch_size'}
            },
            verbose=False,
            export_params=True,
        )
        return dest_path


class KerasSaver(FWAIModel):
    @classmethod
    def name(cls) -> str:
        return 'keras'

    @classmethod
    def match(cls, model) -> bool:
        return isinstance(model, BaseKerasModel) and 'keras' in repr(type(model))

    @classmethod
    def image_to_tensor(cls, image: Union[PILImage.Image, ndarray]):
        return convert_to_tensor(image)

    @classmethod
    def from_path(cls, model_path: str) -> BaseKerasModel:
        import keras
        return keras.models.load_model(model_path)

    def to_non_turing(self, dest_path: str) -> BaseONNXModel:
        input_size = self.input_size or [
            tf.TensorSpec(self.model.input.shape, tf.float32, name='input')
        ]
        logging.debug(f'Writing the onnx model to {dest_path}')
        import tf2onnx
        onnx_model, _ = tf2onnx.convert.from_keras(
            self.model, input_size, output_path=dest_path
        )
        return onnx_model


class SklearnSaver(FWAIModel):
    @classmethod
    def name(cls) -> str:
        return 'sklearn'

    @classmethod
    def match(cls, model) -> bool:
        return isinstance(model, BaseSklearnModel)

    @classmethod
    def from_path(cls, model_path: str) -> BaseSklearnModel:
        import pickle
        return pickle.load(open(str(model_path), "rb"))

    def to_non_turing(self, dest_path: str) -> BaseONNXModel:
        input_size = self.input_size
        if input_size is None:
            try:
                input_size = self.model.coef_.shape[1]
            except AttributeError:
                input_size = self.model.n_features_in_
            except Exception as err:
                raise RuntimeError(err)
            
        logging.debug(f'Writing the onnx model to {dest_path}')
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, input_size]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type, target_opset=17)
        return cast(BaseONNXModel, onnx_model)
        # To return an string
        # with open(dest_path, "wb") as f:
        #     f.write(onx.SerializeToString())
        # return dest_path


class ONNXSaver(FWAIModel):
    @classmethod
    def name(cls) -> str:
        return "onnx"

    @classmethod
    def match(cls, model) -> bool:
        return (isinstance(model, BaseONNXModel) 
                and 'onnx' in repr(type(model)))

    @classmethod
    def from_path(cls, model_path: str) -> BaseONNXModel:
        return onnx.load_model(model_path)

    def to_non_turing(self, dest_path: str) -> BaseONNXModel:
        return self.model


@dataclass
class ContractNodeResult:
    url: str
    sender: str
    node_stake: int

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"  url = {self.url}\n"
            f"  sender = {self.sender}\n"
            f"  node_stake = {self.node_stake}\n"
            f")\n"
        )


@dataclass
class FWAIResult:
    data: Union[ndarray, str]
    url: str

    def __repr__(self):
        max_len_string = 63
        data = f"{self.data}"
        if not isinstance(data, str):
            if len(data) > max_len_string:
                data = data[:max_len_string] + "..."
        return (
            f"{type(self).__name__}(\n"
            f"  data = {data},\n"
            f"  url = {self.url}\n"
            f")\n"
        )

    @classmethod
    def load(cls, url: str):
        path = get_path(url)
        try:
            data = np.loadtxt(path, delimiter=';')
        except ValueError: 
            with open(path, 'r') as result_file:
                data = result_file.read()
        return cls(data=data, url=url)


@dataclass
class Task:
    id: int
    model_url: str
    dataset_url: str
    start_time: Optional[Timestamp] = None
    result_url: Optional[str] = None
    block_number: Optional[int] = None

    def __post_init__(self):
        self.ipfs_url = 'https://ipfs.io/ipfs/'
        self.assert_arguments()

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"  id = {self.id}\n"
            f"  model_url = {self.model_url}\n"
            f"  dataset_url = {self.dataset_url}\n"
            f"  result_url = {self.result_url}\n"
            f")\n"
        )


    def assert_arguments(self):
        # Example of url: 
        # https://ipfs.io/ipfs/QmTRzpaMbxQ2pMKS6BXWcZdeQ4aoxKX4ngGn6Qk9WujJo3
        url_valid_lenght = 67

        assert (isinstance(self.id, int)
            and self.id >= 0
        ), f"{self.id=}. {self.id} is not correct task index"

        assert (self.model_url.startswith(self.ipfs_url) 
            and len(self.model_url) == url_valid_lenght
        ), f"{self.model_url=}. {self.model_url} is not correct ipfs url"

        assert (self.dataset_url.startswith(self.ipfs_url)
            and len(self.dataset_url) == url_valid_lenght
        ), f"{self.dataset_url=}. {self.dataset_url} is not correct ipfs url"

        if self.result_url:
            assert (self.result_url.startswith(self.ipfs_url) 
                and len(self.result_url) == url_valid_lenght
            ), f"{self.result_url=}. {self.result_url} is not correct ipfs url"


    @classmethod
    def load_from_event(cls, event, network: Optional[Network] = None) -> Task:
        network = network or Global.network or Network.build()
        w3 = network.connect()
        return cls(
            id = event['args']['taskIndex'],
            model_url = event['args']['modelUrl'],
            dataset_url = event['args']['datasetUrl'],
            start_time = w3.eth.get_block(event['blockNumber']).get('timestamp'),
            block_number = w3.eth.get_block(event['blockNumber'])
        )


    def submit_result(
        self, 
        result_url: str,
        task_runner_contract: "TaskRunnerContract",
        num_retry=0
    ) -> None:
        self.result_url = result_url
        t1 = time.perf_counter()
        try:
            tx_hash = task_runner_contract.submit_result(self)
            task_runner_contract.wait_for_transaction(tx_hash)
        except Exception as err:
            max_retry = 5
            if num_retry > max_retry:
                raise UnexpectedError(
                    f"Task could not be submitted, err={err}"
                )
            print(
                f"[!] Error ocurred submitting result, "
                f"trying again {num_retry}/{max_retry}"
            )
            return self.submit_result(
                result_url, task_runner_contract, num_retry=num_retry + 1
            )

        t2 = time.perf_counter()
        # TIMER
        # print(f"[TIMER] submit_result took: {t2-t1}")


@dataclass
class AnvilAccount:
    public_key: str
    private_key: str


class Anvil:
    def __init__(
        self,
        config_path: str,
        uri="http://127.0.0.1:8545",
        build_envfile=False
    ):
        self.config_path = config_path
        self.uri = uri

        with open(self.config_path, "r") as jsonfile:
            self.config = json.loads(jsonfile.read())

        if build_envfile:
            self.build_envfile()

        self.accounts = self.config["available_accounts"]
        self.private_keys = self.config["private_keys"]
        self.base_fee = self.config['base_fee']
        self.gas_limit = self.config['gas_limit']
        self.gas_price = self.config['gas_price']
        self.mnemonic = self.config['wallet']['mnemonic']
        self.num_accounts = len(self.accounts)
        self._generate_accounts()

    def _generate_accounts(self):
        for num_node, (account, private_key) in enumerate(zip(self.accounts, self.private_keys)):
            name = f"node{num_node}"
            if num_node == 0:
                name = "master_node"
            self.__setattr__(name, AnvilAccount(account, private_key))

    def build_envfile(self, out_path="/tmp/anvil.env"):
        with open(self.config_path, "r") as jsonfile:
            data = json.loads(jsonfile.read())

        out_string = ""

        for account, public_key in enumerate(data["available_accounts"]):
            out_string += f"ANVIL_NODE{account + 1}={public_key}\n"

        private_key_builded = False
        for account, private_key in enumerate(data["private_keys"]):
            if not private_key_builded:
                out_string += f"PRIVATE_KEY={private_key}\n"
                private_key_builded = True
                
            out_string += f"ANVIL_NODE{account + 1}_PRIVATE_KEY={private_key}\n"

        out_string += f"ANVIL_BASE_FEE={data['base_fee']}\n"
        out_string += f"ANVIL_GAS_LIMIT={data['gas_limit']}\n"
        out_string += f"ANVIL_GAS_PRICE={data['gas_price']}\n"
        out_string += f"ANVIL_MNEMONIC=\"{data['wallet']['mnemonic']}\"\n"
        out_string += f"ANVIL_CONFIG_PATH={self.config_path}\n"
        
        with open(out_path, "w") as outfile:
            outfile.write(out_string)

        # Load environment
        load_global_env(out_path)


    def get_private_key_by_id(self, id: int):
        return os.environ.get(f"ANVIL_NODE{id}_PRIVATE_KEY")


class Network:
    """
    Network class with all necesaries features to 
    mount FreeWillAI on this a specific blockchain network

    Attributes:
        id: String
            It's how FreeWillAI will be get the network. 
            Must has the this format ('network type/name')
            Example:
                 'mainnet/scroll'
                 'testnet/arbitrum'
                 'devnet/anvil'
            (Only non-official network do not follows this format)
            Available network types = ['mainnet', 'testnet', 'devnet']
            More info: https://www.hiro.so/blog/devnet-vs-testnet-vs-mainnet-what-do-they-mean-for-web3-developers

        rpc_urls: List of strings
            A list of valid rpc/urls/endpoints to connect to network
            Example:
                rpc_urls = [
                    "https://rpc.sepolia.org",
                    "https://rpc2.sepolia.org"
                ]
            The first rpc will be select by Network class
            The others are to compare with the user passed rpc.
            Recommended put the minor latency rpc first
            You can search the list of rpc here: https://chainlist.org/

        explorer: Optional String
            Link to blockchain explorer
            Example:
                'https://zkevm.polygonscan.com/' 
                'https://blockscout.scroll.io'

        token_address: String
            Address of deployed FreeWillAI token contract

        task_runner_address: String
            Address of deployed TaskRunner contract
    """
    id: str = 'non-official'
    rpc_urls: List[str] = ['']
    url = Global.rpc_url
    explorer: Optional[str] = None
    token_address: str
    task_runner_address: str

    def __new__(cls, network_id_or_rpc_url:str, *_, **__):
        # If arg is not a network id return Network class
        if network_id_or_rpc_url == "non-official":
            return cls
        elif not network_id_or_rpc_url in cls.official_networks():
            return cls
        subclass = cls.by_network_id(network_id_or_rpc_url)
        return object.__new__(subclass)

    def __init__(
        self, 
        network_id_or_rpc_url: Optional[str] = None,
        middlewares: List[Middleware] = [],
        token_address: Optional[str] = None,
        task_runner_address: Optional[str] = None,
    ):
        """
        Network classrpc_urlf all necesaries features to 
        mount FreeWillAI on this a specific blockchain network

        Arguments
        ---------
            network_id_or_rpc_url: Optional string
                string used to connect to the network. 
            middlewares: List of middlewares
                Read Middleware class documentation to know more about
            token_address: String
                Address of FreeWillAI token contract deployed in this rpc
            task_runner_address: String
                Address of TaskRunner contract deployed in this rpc
        """
        self.rpc_url = network_id_or_rpc_url
        if not self.id == "non-official":
            self.rpc_url = self.rpc_urls[0]
        self._middlewares = middlewares
        self.token_address = token_address or self.token_address
        self.task_runner_address = task_runner_address or self.task_runner_address
        self.type = self.id.split('/', 1)[0]

        if self.type == "non-official" and self.rpc_url is None:
            raise UserRequirement(
                "If you want to use a non-official network you need to "
                "pass rpc url in rpc_url argument"
            )

        if (self.id == "non-official" 
            and (
                not self.token_address 
                or not self.task_runner_address
        )):
            raise UserRequirement(
                "If you want to use a non-official network "
                "you need to deploy the contracts on this rpc_url and pass "
                "addresses to Network class\n"
                "Usage:\n"
                "    network = Network(\n"
                "       'http://non-official.rpc', \n"
                "       token_address='0x70997970C51812dc3A010C7d01b50e0d17dc79C8',\n"
                "       task_runner_address='0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266',\n"
                "    )\n"
            )

        self.w3 = None
        self.asserts()

        self.__post_init__()
        
    def __post_init__(self):
        """Method executed after __init__"""
        ...

    def connect(self):
        if self.rpc_url is None:
            raise UnexpectedError("rpc_url is None")
        print(f"[*] You are connected to a {self.type} network on {self.rpc_url}")
        self.w3 = get_w3(self.rpc_url, middlewares=self._middlewares)
        self.connection_asserts(self.w3)
        self.middleware_onion = self.w3.middleware_onion
        self.add_middleware(
            cast(Middleware, buffered_gas_estimate_middleware),
            'gas_estimate'
        )
        return self.w3

    def with_custom_rpc(self, rpc_url: str):
        self.rpc_url = rpc_url
        return self
        
    @classmethod
    def build(
        cls, 
        rpc_url: Optional[str] = None,
        middlewares: List[Middleware] = [],
        token_address: Optional[str] = None,
        task_runner_address: Optional[str] = None,
    ):
        """
        Build Network class with

        IMPORTANT: If network is official it should has not errors

        Arguments
        ---------
            rpc_url: Optional string
                string used to connect to the network. 
            middlewares: List of middlewares
                Read Middleware class documentation to know more about
            token_address: String
                Address of FreeWillAI token contract deployed in this rpc
            task_runner_address: String
                Address of TaskRunner contract deployed in this rpc

        Missing args will be filled with the following environment variables:
            rpc_url -> FREEWILLAI_RPC
            token_address -> FREEWILLAI_TOKEN_ADDRESS
            task_runner_address -> FREEWILLAI_TASK_RUNNER_ADDRESS

        Example in bash, how to declare these variables:
            export FREEWILLAI_RPC=https://polygon.llamarpc.com
            export FREEWILLAI_TOKEN_ADDRESS=0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
            export FREEWILLAI_TASK_RUNNER_ADDRESS=0x70997970C51812dc3A010C7d01b50e0d17dc79C8

        Returns
        -------
            Instanced Network class
        """
        if not cls.id == "non-official":
            return cls(
                cls.rpc_urls[0],
                middlewares=middlewares,
                token_address=cls.token_address,
                task_runner_address=cls.task_runner_address,
            )

        rpc_url = rpc_url or cls.rpc_urls[0] or Global.rpc_url or None
        token_address = token_address or Global.token_address
        task_runner_address = task_runner_address or Global.task_runner_address

        err_msg = (
            "{name} is not declared. Please pass a valid {arg} argument "
            "or declare it as {env_var} environ variable "
            "(export {env_var}={example})"
        ).format

        if rpc_url is None:
            raise UserRequirement(err_msg(
                name='RPC',
                arg='rpc_url',
                env_var='FREEWILLAI_RPC',
                example='https://host:port/path'
            ))

        if token_address is None:
            raise UserRequirement(err_msg(
                name='FreeWillAI token contract address',
                arg='token_address',
                env_var='FREEWILLAI_TOKEN_ADDRESS',
                example='0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266'
            ))

        if task_runner_address is None:
            raise UserRequirement(err_msg(
                name='TaskRunner contract address',
                arg='task_runner_address',
                env_var='FREEWILLAI_TASK_RUNNER_ADDRESS',
                example='0x70997970C51812dc3A010C7d01b50e0d17dc79C8'
            ))

        return cls(
            rpc_url,
            middlewares=middlewares,
            token_address=token_address,
            task_runner_address=task_runner_address,
        )


    def asserts(self) -> None:
        """
        Develop the asserts necesaries to ensure well integrate 
        of network with the rest of code

        IMPORTANT: This code will be running in __init__ and build method

        Asserts
        -------
            - Valid rpc_url 
            - Valid middlewares 
            - Token and task runner addresses exists in network

        Variables valid to use
        ----------------------
            self.rpc_url
            self.middlewares 
            self.token_address 
            self.task_runner_address
            self.type
        """

        assert (
            self.rpc_url 
            and isinstance(self._middlewares, List) 
            and self.token_address 
            and self.task_runner_address
        ), "Bad Instance"

        # Check is valid rpc
        if (self.rpc_url is not None
            and (
                not self.rpc_url.startswith("http://")
                and not self.rpc_url.startswith("https://")
                and not self.rpc_url.startswith("ws://")
                and not self.rpc_url.startswith("wss://"))
        ):
            raise NotSupportedError(
                f"{self.rpc_url=}. Invalid rpc format\n"
                f"rpc format: <protocol>://<host>:<port>/<apikey>\n"
                f"  - valid protocols (required): http, https, ws \n"
                f"  - host (required): IP or websites to connect. \n"
                f"    Also you can using localhost\n"
                f"  - port (optional): port to connect\n"
                f"  - apikey (optional): some rpc/endpoints needs \n"
                f"    an apikey to able you to interact with them\n"
            )

    def connection_asserts(self, w3) -> None:
        # Check if contracts are deployed on rpc
        if not contract_exists(self.token_address, w3):
            raise UserRequirement(
                f"FreeWillAI token contract not found address={self.token_address} "
                f"on rpc_url={self.rpc_url}\n"
                f"Please deploy it"
            )

        if not contract_exists(self.task_runner_address, w3):
            raise UserRequirement(
                f"TaskRunner contract not found address={self.token_address} "
                f"on rpc_url={self.rpc_url}\n"
                f"Please deploy it"
            )


    @classmethod
    def official_networks(cls) -> List[str]:
        """List all official/supported networks"""
        return [subclass.id for subclass in cls.__subclasses__()]


    @classmethod
    def by_network_id(cls, id: str) -> Type[Network]:
        """Get Network class by id"""
        id = id.lower()
        if id == cls.id:
            return cls
        for subclass in cls.__subclasses__():
            if subclass.id == id or id in subclass.id:
                return subclass
        raise RuntimeError(
            f"{id=} does not match with any Network"
        )


    @classmethod
    def dump_networks(cls, dest_path: str = "./networks.json") -> None:
        data = {}
        for subclass in cls.__subclasses__():
            id = subclass.id
            if isinstance(id, list):
                id = id[0]
            data[id] = {
                "rpc_urls": subclass.rpc_urls,
                "explorer": subclass.explorer,
                "token_address": subclass.token_address,
                "task_runner_address": subclass.task_runner_address,
            }
        with open(dest_path, 'w') as jsonfile:
            jsonfile.write(json.dumps(data, indent=4))
             
    @classmethod
    def is_api_key_required(cls):
        return False

    @property
    def middlewares(self) -> List[Middleware]:
        if hasattr(self, "w3"):
            if not self.w3:
                self.w3 = self.connect()
            return cast(List[Middleware], self.w3.middleware_onion.middlewares)
        return self._middlewares

    @middlewares.setter
    def middlewares(self, middlewares: Middleware):
        if not self.w3:
            self.w3 = self.connect()
        self.w3.middleware_onion.middlewares = middlewares

    def is_mainnet(self):
        return True if self.type == "mainnet" else False

    def add_middleware(
        self, 
        middleware: Middleware, 
        name: Optional[str] = None, 
        layer: Optional[Literal[0, 1, 2]] = None
    ):
        if not self.w3:
            self.w3 = self.connect()
        if layer is None:
            self.w3.middleware_onion.add(middleware, name) 
        else:
            self.w3.middleware_onion.inject(middleware, name, layer)

    def remove_middleware(
        self,
        middleware: Union[Middleware, str], 
    ):
        if not self.w3:
            self.w3 = self.connect()
        self.w3.middleware_onion.remove(middleware)

    def is_sign_and_send_allowed(self) -> bool:
        if not self.w3:
            self.w3 = self.connect()
        if not self.middleware_onion.get("allow_sign_and_send"):
            return False
        return True

    def allow_sign_and_send(self, account: LocalAccount) -> None:
        if not self.is_sign_and_send_allowed():
            self.add_middleware(
                construct_sign_and_send_raw_middleware(account), 'allow_sign_and_send'
            )


class SepoliaNetwork(Network):
    id = "testnet/sepolia"
    rpc_urls = [
        "https://rpc2.sepolia.org"
    ]
    explorer = "https://sepolia.etherscan.io"
    # env_api_key = ["SEPOLIA_API_KEY", "API_KEY"]
    token_address = "0xDAE95F004b4B308921c8fdead101555eAB83916B"
    task_runner_address = "0x80a4C63B8201f1b69C03AbCf1525353a7F1186db"


class AnvilNetwork(Network):
    id = "devnet/anvil"
    rpc_urls = [
        Global.anvil_rpc or 'http://127.0.0.1:8545'
    ]
    token_address = "0x4Dd5F358D39A9DFfaC8F1905FcC238A04BC1f332"
    task_runner_address = "0xDAE95F004b4B308921c8fdead101555eAB83916B"
    explorer = None
    def __init__(
        self, 
        rpc_url: Optional[str] = None,
        middlewares: List[Middleware] = [],
        token_address: Optional[str] = None,
        task_runner_address: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or self.rpc_urls[0]
        self.token_address = token_address or os.environ.get("FREEWILLAI_TOKEN_ADDRESS") or self.token_address
        self.task_runner_address = task_runner_address or os.environ.get("FREEWILLAI_TASK_RUNNER_ADDRESS") or self.task_runner_address

        anvil_env = "anvil.env"
        load_global_env(anvil_env)

        self.master_account = os.environ.get("PRIVATE_KEY")

        super().__init__(
            network_id_or_rpc_url=self.rpc_url,
            middlewares=middlewares,
            token_address=self.token_address,
            task_runner_address=self.task_runner_address,
        )

    @classmethod
    def build(
        cls,
        rpc_url: Optional[str] = None,
        middlewares: List[Middleware] = [],
        *_, **__
    ):
        rpc_url = rpc_url or cls.rpc_urls[0]
        return cls(rpc_url, middlewares)


class GoerliNetwork(Network):
    id = "testnet/goerli"
    rpc_urls = [
        # temporal infura link
        # "https://goerli.infura.io/v3/55c5715ea59149e799b91e8c06463e1c"
        "https://goerli.infura.io/v3/990d1ccb908249fab01f006d3a3a9c5d"
    ]
    explorer = "https://goerli.etherscan.io/"
    token_address = "0x9b5F4d2CDD87e7b38EB35d1A6b9AaF5F03B9CdaB"
    task_runner_address = "0x4c074D6c1f2dF27d0FbDf18F3c1ad140C1cdCD2e"

    def __post_init__(self):
        # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )

class ScrollTestnetNetwork(Network):
    id = "testnet/scroll"
    rpc_urls = [
        "https://alpha-rpc.scroll.io/l2"
    ]
    explorer = "https://blockscout.scroll.io"
    token_address = "0xc365bc6d5ADd80998FFa68d0Da4925A54C43D0F6"
    task_runner_address = "0x9bDfF39Fa9Bd210629f3FEd4A4470A753268Bb6F"

    def __post_init__(self):
        # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )


class ArbitrumTestnetNetwork(Network):
    id = "testnet/arbitrum"
    rpc_urls = [
        "https://goerli-rollup.arbitrum.io/rpc"
    ]
    explorer = "https://goerli.arbiscan.io"
    token_address = "0xdDe55Bbf8bB6b13C11De0AfDb8214f245dB48d4a"
    task_runner_address = "0x9F6a708dE9cEBe7df6a7d0Fb63235fd91D0b36B0"

    def __post_init__(self):
        # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )
    

class PolygonZkEVMTestnetNetwork(Network):
    id = "testnet/polygon/zkevm"
    rpc_urls = [
        "https://rpc.public.zkevm-test.net"
    ]
    explorer = "https://testnet-zkevm.polygonscan.com"
    token_address = "0xc365bc6d5ADd80998FFa68d0Da4925A54C43D0F6"
    task_runner_address = "0x49717A5D8036be3aaf251463efeCf30bff38A209"
    
    def __post_init__(self):
        # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )


class AuroraTestnetNetwork(Network):
    id = "testnet/aurora"
    rpc_urls = [
        "https://testnet.aurora.dev"
    ]
    explorer = "https://testnet.aurorascan.dev"
    token_address = "0xc365bc6d5ADd80998FFa68d0Da4925A54C43D0F6"
    task_runner_address = "0x66714d35a0d8C585665AA73279b1828eC049eB26"

    def __post__init__(self):
        # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )

    
class NeonTestnetNetwork(Network):
    id = "testnet/neonevm"
    rpc_urls = [
        "https://devnet.neonevm.org",
    ]
    explorer = "https://devnet.neonscan.org/"
    token_address = "0xc365bc6d5ADd80998FFa68d0Da4925A54C43D0F6"
    task_runner_address = "0x749091C807660e3b89A119878204135C2803E81b"

    def __post_init__(self):
        # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )

     
class MantleTestnetNetwork(Network):
    id = "testnet/mantle"
    rpc_urls = [
        "https://rpc.testnet.mantle.xyz",
    ]
    explorer = "https://explorer.testnet.mantle.xyz/"
    token_address = "0x0BcFF346d24668DD16d760F0C9Cef45aD8580FD5"
    task_runner_address = "0xd17746a67f05D7e1E6e199d6f6e76597061E980E"

    def __post_init__(self):
        # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )
  

class AuroraNetwork(Network):
    id = "mainnet/aurora"
    rpc_urls = [
        "https://mainnet.aurora.dev"
    ]
    explorer = "https://aurorascan.dev/"
    token_address = "0x4Dd5F358D39A9DFfaC8F1905FcC238A04BC1f332"
    task_runner_address = "0xDAE95F004b4B308921c8fdead101555eAB83916B"


class GnosisNetwork(Network):
    id = "mainnet/gnosis"
    rpc_urls = [
        # TODO: create a quiknode class
        "https://shy-solitary-violet.xdai.discover.quiknode.pro/996c2ee53bdab6c1a18f039f7ac4dd6615e1e8b1/",
        "https://rpc.gnosischain.com",
    ]
    explorer = "https://gnosisscan.io"
    token_address = "0x4Dd5F358D39A9DFfaC8F1905FcC238A04BC1f332"
    task_runner_address = "0xdDe55Bbf8bB6b13C11De0AfDb8214f245dB48d4a"


class NeonEVMNetwork(Network):
    id = "mainnet/neonevm"
    rpc_urls = [
        "https://neon-proxy-mainnet.solana.p2p.org"
    ]
    explorer = "https://neonscan.org"
    token_address = "0xDAE95F004b4B308921c8fdead101555eAB83916B"
    task_runner_address = "0x80a4C63B8201f1b69C03AbCf1525353a7F1186db"


class LineaTestnetNetwork(Network):
    id = "testnet/linea"
    rpc_urls = [
        "https://linea-goerli.infura.io/v3/f4c806af885a4ff68a70b65e624f046b",
        "https://rpc.goerli.linea.build",
    ]
    explorer = "https://goerli.lineascan.build/"
    token_address = "0x70bb5CB76257579A8895a012Da0afA992eAf8c16"
    task_runner_address = "0xc74f072eE7736EfFD418A1C98cF1bc534477F86A"

    def __post_init__(self):
         # For Goerli
        self.add_middleware(
            cast(Middleware, geth_poa_middleware), 
            'proof_of_authority', layer=0
        )


class PolygonZkEVMNetwork(Network):
    id = "mainnet/polygon/zkevm"
    rpc_urls = [
        "https://zkevm-rpc.com",
    ]
    explorer = "https://zkevm.polygonscan.com"
    token_address = "0xcdcB42FeE1F1C778c48bcD199590D676c7c46b9d"
    task_runner_address = "0xC2B7682A400A618fA23f4D3E6EBa677286C8bc26"


class OptimismTestnetNetwork(Network):
    id = "testnet/optimism"
    rpc_urls = [
        "https://sepolia.optimism.io/"
    ]
    explorer = "https://sepolia-optimism.etherscan.io"
    token_address = "0x4Dd5F358D39A9DFfaC8F1905FcC238A04BC1f332"
    task_runner_address = "0xDAE95F004b4B308921c8fdead101555eAB83916B"


class ZKSyncTestnetNetwork(Network):
    id = "testnet/zksync"
    rpc_urls = [
        "wss://sepolia.era.zksync.dev/ws",
        "https://sepolia.era.zksync.dev",
    ]
    explorer = "https://sepolia.explorer.zksync.io/"
    token_address = "0x82f72f45b81b909A5d6e560896C266E5E62b3E5B"
    task_runner_address = "0xed732c073D2BD4FE426c0b2FD41e635e9D314f38"
