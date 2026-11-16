"""Lambda entry point. Mangum translates Lambda events into ASGI requests for FastAPI."""

from mangum import Mangum

from api.main import app

handler = Mangum(app, lifespan="auto")