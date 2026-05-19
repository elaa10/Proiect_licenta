from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.models.user import User
from app.models.analysis import AnalysisRequest, AnalysisResult, BrandReference
from app.routers import auth, analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Phishing Detector API",
    description="Automatic phishing page detection via lexical, ML, and visual analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")

app.include_router(auth.router)
app.include_router(analyze.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "Phishing Detector API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}