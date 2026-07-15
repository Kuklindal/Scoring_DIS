from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.scoring_routes import router as scoring_router
from app.api.features_routes import router as features_router

app = FastAPI(
    title="Скоринг возвращаемости клиентов",
    description="API для расчёта вероятности возврата отключённых клиентов",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scoring_router)
app.include_router(features_router)

@app.get("/")
async def root():
    return {"message": "Скоринг возвращаемости клиентов API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is running"}