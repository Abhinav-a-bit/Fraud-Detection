from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.db.database import Base, engine
from app.services.ml_service import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()      
    Base.metadata.create_all(bind=engine)   
    yield              

app = FastAPI(
    title='Fraud Detection API', 
    version='1.0.0', 
    lifespan=lifespan 
)
app.include_router(router, prefix='/api/v1') 

@app.get('/health') 
def health_check(): 
    return {'status': 'ok', 'model': 'loaded'} 