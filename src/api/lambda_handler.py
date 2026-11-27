"""AWS Lambda entry point: the same FastAPI app, wrapped by Mangum so it can answer
API Gateway events.

The model is loaded at import time, i.e. once per Lambda container (cold start).
Mangum's lifespan is turned off because it would otherwise re-run the startup
hook - and reload the model - on every single request.
"""
import os
from mangum import Mangum
from src.api.main import create_app
from src.api.model_store import load_forecaster

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/forecaster")

app = create_app(MODEL_DIR)
app.state.forecaster = load_forecaster(MODEL_DIR)

handler = Mangum(app, lifespan="off")
