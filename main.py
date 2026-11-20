from __future__ import annotations

import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from limit_article_matching.router import router as matching_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("limit_article_matching")

app = FastAPI(
    title="UzAssets - Product Limit Article Matching Service",
    version="1.0.0",
    description=(
        "«UzAssets» AJ tizimi uchun xarid qilinayotgan tovar va xizmatlarni "
        "tashkilotning byudjet xarajat moddalariga (Limit Article) moslashtiruvchi AI servisi."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "limit-article-matching", "version": "1.0.0"}

app.include_router(matching_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
