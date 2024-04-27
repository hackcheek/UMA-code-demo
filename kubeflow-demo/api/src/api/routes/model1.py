import pandas as pd

from io import BytesIO, StringIO
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from api.src.api.schemas.predict_request import PredictRequest, PredictRequestFile
from api.src.utils.model1 import predict
from api.src.utils.response import predict_response_handler


router = APIRouter()


@router.post('/csv/predict', response_class=JSONResponse)
async def api_csv_predict(request: PredictRequestFile = Depends()):
    buffer = BytesIO(request.data)
    buffer.seek(0)
    data = pd.read_csv(buffer)
    del buffer

    data = predict(data)
    
    data = predict_response_handler(data, request.output_format)

    return {
        'result': data,
        'output_format': request.output_format
    }


@router.post('/predict', response_class=JSONResponse)
async def api_predict(request: PredictRequest):
    if isinstance(request.data, str):
        buffer = StringIO(request.data)
        buffer.seek(0)
        data = pd.read_csv(buffer)
        del buffer
    elif isinstance(request.data, dict):
        data = pd.DataFrame(request.data)
    else:
        raise RuntimeError(f"Invalid data type {type(request.data)}")

    data = predict(data)
    data = predict_response_handler(data, request.output_format)

    return {
        'result': data,
        'output_format': request.output_format
    }
