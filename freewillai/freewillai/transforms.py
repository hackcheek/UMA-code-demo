from __future__ import annotations
import torch
import numpy as np
from numpy import ndarray
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToTensor
from typing import Tuple, Union, Type, Callable, List
from tensorflow import convert_to_tensor


class Transforms:
    def __init__(self, model_name: str):
        """
        Using torch functions for transformations for now

        Arguments
        ---------
        model_name: String
            Name of the model library that process this array/image
            model name must be in FWAIModel.available_models
        """
        # Solve circular issue with FAWIModel
        # assert model_name in FWAIModel.available_models, f"{model_name=} Not valid"
        self.model_name = model_name


    def resize(
        self,
        image: PILImage.Image,
        size: Union[Tuple, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size: Union[Tuple, int, None] = None,
        antialias: str = "warn"
    ) -> ndarray:
        """Resize images"""
        resized_image = transforms.Resize(
            size, interpolation, max_size, antialias
        )(image)
        return  np.array(resized_image)
    

    def tf_image_to_tensor(self, image: Union[PILImage.Image, ndarray]) -> ndarray:
        return convert_to_tensor(image).numpy()


    def torch_image_to_tensor(self, image: Union[PILImage.Image, ndarray]) -> ndarray:
        if isinstance(image, PILImage.Image):
            image = np.array(image)
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image.squeeze()
        return np.expand_dims(ToTensor()(image).numpy(), 0)


    def image_to_tensor(self, image: Union[PILImage.Image, ndarray]) -> ndarray:
        """PIL/numpy images to numpy tensor"""
        if self.model_name == "tensorflow":
            return self.tf_image_to_tensor(image)
        return self.torch_image_to_tensor(image)
