from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from contextlib import asynccontextmanager
from app.api.routes import router
from app.db.database import create_tables
from app.services.ml_service import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()      
    create_tables()   
    yield             

app = FastAPI(
    title='Enterprise Fraud Detection API',
    version='1.0.0',
    lifespan=lifespan
)

app.include_router(router, prefix='/api/v1')

@app.get('/health')
def health_check():
    return {'status': 'ok', 'model': 'loaded'}

@app.exception_handler(IntegrityError)
async def sqlalchemy_integrity_error_handler(request: Request, exc: IntegrityError):
    # Log the exact error to the terminal for debugging
    print(f"Database Integrity Error: {exc.orig}")
    
    return JSONResponse(
        status_code=409,
        content={
            "detail": "Conflict: A transaction with this ID already exists and has been processed."
        }
    )