from typing import Any, Literal


class BaseDataTransform:
    """
    This clase should provide preprocessing and postprocessing transformations on a specific dataset
    """

    @classmethod
    def preprocessing(cls, data: Any, for_inference: bool) -> Any:
        raise NotImplementedError

    @classmethod
    def postprocessing(cls, data: Any) -> Any:
        raise NotImplementedError
