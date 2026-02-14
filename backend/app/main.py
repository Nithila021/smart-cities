from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.data_loader import initialize_data
from app.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load data on startup
    initialize_data()
    yield
    # Clean up (if any) on shutdown

app = FastAPI(
    title="Safety Analysis API",
    lifespan=lifespan
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:80",
    "http://127.0.0.1",
    "http://127.0.0.1:80",
    "http://localhost:3000",
    "http://localhost:5000",
    "*" # Helpful for dev, restrict in prod if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

@app.get("/")
def health_check():
    return {"status": "Safety Analysis Service Running"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.main:app", host='0.0.0.0', port=5000, reload=True)
