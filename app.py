from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from contextlib import asynccontextmanager
import os
import pandas as pd
from ml_models.yield_prediction import yield_predictor
from ml_models.crop_recommendation import crop_recommender
from ml_models.disease_detection import disease_detector
import gc  # Import garbage collector

# Configuration
DATA_PATH = "data/crop_yield.csv"
YIELD_MODEL_PATH = "ml_models/yield_model.joblib"
CROP_MODEL_PATH = "ml_models/crop_model.joblib"

# Pydantic Models
class HealthResponse(BaseModel):
    status: str
    message: str

class CropRecommendationRequest(BaseModel):
    region: str = Field(..., description="State/Region name")
    area: float = Field(1.0, gt=0, description="Area in hectares")
    fertilizer: float = Field(50.0, ge=0, description="Fertilizer usage in kg")
    pesticide: float = Field(10.0, ge=0, description="Pesticide usage in litres")

class YieldPredictionRequest(BaseModel):
    crop: str = Field(..., description="Crop name")
    state: str = Field(..., description="State name")
    area: float = Field(..., gt=0, description="Area in hectares")
    fertilizer: float = Field(..., ge=0, description="Fertilizer usage in kg")
    pesticide: float = Field(..., ge=0, description="Pesticide usage in litres")

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Farmer Assistance API...")
    
    # Load light models (Yield/Crop) ONLY
    if os.path.exists(YIELD_MODEL_PATH):
        yield_predictor.load_model(YIELD_MODEL_PATH)
    
    if os.path.exists(CROP_MODEL_PATH):
        crop_recommender.load_model(CROP_MODEL_PATH)

    # DO NOT load Disease model at startup
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    gc.collect()

# FastAPI App with lifespan
app = FastAPI(
    title="Farmer Assistance API",
    description="Simple agricultural assistance platform",
    version="2.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
   allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("ml_models", exist_ok=True)
os.makedirs("data", exist_ok=True)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Farmer Assistance API is running"
    )

@app.post("/api/train")
async def train_model():
    """Train both yield and crop models"""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(400, f"Data file not found at {DATA_PATH}")
    
    try:
        yield_result = yield_predictor.train_model(DATA_PATH, YIELD_MODEL_PATH)
        crop_result = crop_recommender.train_model(DATA_PATH, CROP_MODEL_PATH)
        
        return {
            "success": True,
            "yield_model": yield_result,
            "crop_model": crop_result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/recommend-crops")
async def recommend_crops(request: CropRecommendationRequest):
    """Get crop recommendations based on inputs"""
    # Auto-train if model is not trained
    if not crop_recommender.is_trained:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(400, "Data file not found for training")
        
        train_result = crop_recommender.train_model(DATA_PATH, CROP_MODEL_PATH)
        if train_result.get("status") != "success":
            raise HTTPException(400, f"Auto-training failed: {train_result.get('message')}")

    input_data = {
        'state': request.region,
        'area': request.area,
        'fertilizer': request.fertilizer,
        'pesticide': request.pesticide
    }
    
    result = crop_recommender.recommend_crops(input_data)
    
    if "error" in result:
        raise HTTPException(400, result["error"])
    
    return result

@app.post("/api/predict-yield")
async def predict_yield(request: YieldPredictionRequest):
    """Make yield prediction for a specific crop"""
    # Auto-train if model is not trained
    if not yield_predictor.is_trained:
        if not os.path.exists(DATA_PATH):
            raise HTTPException(400, "Data file not found for training")
        
        train_result = yield_predictor.train_model(DATA_PATH, YIELD_MODEL_PATH)
        if train_result.get("status") != "success":
            raise HTTPException(400, f"Auto-training failed: {train_result.get('message')}")
    
    # Make prediction
    prediction = yield_predictor.predict(request.dict())
    
    if "error" in prediction:
        raise HTTPException(400, prediction["error"])
    
    return {
        "success": True,
        "prediction": prediction,
        "input": request.dict()
    }

@app.get("/api/status")
async def get_status():
    """Check status of all models"""
    return {
        "yield_model": {
            "is_trained": yield_predictor.is_trained,
            "message": "Yield model is ready" if yield_predictor.is_trained else "Yield model needs training"
        },
        "crop_model": {
            "is_trained": crop_recommender.is_trained,
            "message": "Crop model is ready" if crop_recommender.is_trained else "Crop model needs training"
        },
        "disease_model": {
            "is_loaded": disease_detector.is_loaded,
            "message": "Disease model is ready" if disease_detector.is_loaded else "Disease model needs loading"
        }
    }

@app.get("/api/available-states")
async def get_available_states():
    """Get list of available states from training data"""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(400, "Data not available")
    
    df = pd.read_csv(DATA_PATH)
    states = sorted(df['State'].astype(str).str.strip().str.title().unique().tolist())
    
    return {"states": states}

@app.get("/api/available-crops")
async def get_available_crops():
    """Get list of available crops from training data"""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(400, "Data not available")
    
    df = pd.read_csv(DATA_PATH)
    crops = sorted(df['Crop'].astype(str).str.strip().str.title().unique().tolist())
    
    return {"crops": crops}

# Disease Detection Routes
@app.post("/api/disease-detection/load-model")
async def load_disease_model():
    """Load the disease detection model"""
    result = disease_detector.load_model()
    
    if result["status"] == "success":
        return {"success": True, "message": result["message"]}
    else:
        raise HTTPException(400, result["message"])

@app.post("/api/disease-detection/predict")
async def predict_disease(image: UploadFile = File(...)):
    """Predict disease from uploaded image"""
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # 1. Unload other models to free up RAM
        yield_predictor.model = None
        crop_recommender.model = None
        gc.collect() # Force cleanup
        
        # 2. Load Disease Model
        if not disease_detector.is_loaded:
            print("Loading disease model...")
            load_result = disease_detector.load_model()
            if load_result["status"] != "success":
                raise HTTPException(400, f"Model loading failed: {load_result['message']}")
        
        # 3. Predict
        result = await disease_detector.predict_disease(image)
        
        # 4. IMMEDIATELY Unload Disease Model to save RAM
        disease_detector.model = None
        disease_detector.is_loaded = False
        gc.collect() # Force cleanup
        print("Disease model unloaded to save RAM")

        if "error" in result:
            raise HTTPException(400, result["error"])
        
        return result

    except Exception as e:
        # Ensure cleanup even on error
        disease_detector.model = None
        disease_detector.is_loaded = False
        gc.collect()
        raise e

@app.get("/api/disease-detection/status")
async def get_disease_status():
    """Check if disease detection model is loaded"""
    return {
        "is_loaded": disease_detector.is_loaded,
        "message": "Disease detection model is ready" if disease_detector.is_loaded else "Model needs loading"
    }

@app.get("/api/disease-detection/supported-crops")
async def get_supported_crops():
    """Get list of crops supported by the disease detection model"""
    supported_crops = [
        "Tomato", "Potato", "Corn", "Grape", "Apple", "Cherry", 
        "Peach", "Pepper", "Strawberry", "Soybean"
    ]
    return {"supported_crops": supported_crops}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)