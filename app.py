from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Tuple, Optional, Dict
import os
import uvicorn
from predict import predict_cat_landmarks, load_onnx_model   

# Define the app
app = FastAPI(
    title="Cat Facial Landmark Detection API",
    description="API for detecting facial landmarks in cat images",
    version="1.0.0"
)
# Define the request model
class ImageRequest(BaseModel):
    image_url: HttpUrl
    
# Define the response model with named landmarks
class LandmarkResponse(BaseModel):
    landmarks: Dict[str, Optional[Tuple[int, int]]]
    message: str

# Define landmark names
LANDMARK_NAMES = [
    "left_eye",
    "right_eye", 
    "mouth", 
    "left_ear_1", 
    "left_ear_2", 
    "left_ear_3", 
    "right_ear_1", 
    "right_ear_2", 
    "right_ear_3"
]

# Define the model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.onnx")
session = load_onnx_model(MODEL_PATH)

@app.get("/")
async def root():
    return {"message": "Welcome to the Cat Facial Landmark Detection API"}

@app.post("/detect-landmarks", response_model=LandmarkResponse)
async def detect_landmarks(request: ImageRequest):
    try:
        # Call the prediction function from predict.py
        landmarks_list = predict_cat_landmarks(
            onxx_session=session,
            image_url=str(request.image_url)
        )
        
        # Convert list of landmarks to dictionary with named keys
        landmarks_dict = {}
        for i, landmark in enumerate(landmarks_list):
            if i < len(LANDMARK_NAMES):
                landmarks_dict[LANDMARK_NAMES[i]] = landmark
        
        # Return the landmarks with appropriate names
        return LandmarkResponse(
            landmarks=landmarks_dict,
            message="Landmarks detected successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting landmarks: {str(e)}")

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}")
        print("Make sure to place your ONNX model in the 'models' directory")
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

    #example curl:
    #curl -X POST "http://localhost:8000/detect-landmarks" -H "Content-Type: application/json" -d '{"image_url": "https://media.istockphoto.com/id/157671964/photo/portrait-of-a-tabby-cat-looking-at-the-camera.jpg?s=612x612&w=0&k=20&c=iTsJO6vuQ5w3hL5pWn42C91ziMRUsYd725oUGRRewjM="}'