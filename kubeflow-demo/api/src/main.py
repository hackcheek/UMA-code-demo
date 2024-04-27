"""
Endpoints

Request for prediction (default ch1 model and json as input)
/predict

Csv file as input
/csv/predict

Make prediction with ch2 model
/ch2/predict
/ch2/csv/predict
"""
import uvicorn
from fastapi import FastAPI
from api.src.api.main import api_router


app = FastAPI()
app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9595)
