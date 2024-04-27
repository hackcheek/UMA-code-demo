from fastapi import APIRouter
from api.src.api.routes import model2, model1


api_router = APIRouter()
api_router.include_router(model1.router, tags=['model1', 'predict'])

# NotImplemented
# api_router.include_router(ch2.router, prefix='/v2', tags=['model2 (Not Implemented)', 'predict'])
