import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from sqlalchemy.exc import OperationalError

from database import engine, Base
from ml_models.Insightfulness_Model import Insightfulness
from api import router as api_router

def _wait_for_db(engine, timeout: int = 60):
    """Waits for the database to be ready before starting the app."""
    start = time.time()
    while True:
        try:
            with engine.connect() as conn:
                return
        except OperationalError:
            if time.time() - start > timeout:
                raise
            print("[startup] DB not ready, retrying in 1s...")
            time.sleep(1)


# Table Initialization (wait for Postgres to be ready)
_wait_for_db(engine, timeout=60)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Reviewer Agent API")

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".weights", "v7_frozen", "epoch_117.pt"))

# Instantiate models once and store in app state for sharing
app.state.insight_model = Insightfulness(model_path=MODEL_PATH)
app.state.rag_system = app.state.insight_model.checker

# Include API routes
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
