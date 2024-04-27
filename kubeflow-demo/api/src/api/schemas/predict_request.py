from typing import Literal
from fastapi import Form, File
from dataclasses import dataclass


@dataclass
class PredictRequest:
    data: list[dict] | dict[str, dict | list] | str
    output_format: Literal['csv', 'json'] = 'json'


class PredictRequestFile:
    def __init__(
        self,
        data: bytes = File(...), 
        output_format: Literal['csv', 'json'] = Form('json')
    ):
        self.data = data
        self.output_format = output_format
