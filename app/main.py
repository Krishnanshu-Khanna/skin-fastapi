from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .model import load_model
from .utils import predict_image

app = FastAPI()
model = load_model("weights/best_weights.pth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("Model loaded successfully.")
print("FastAPI server is running... at http://localhost:5000")

@app.get("/")
async def root():
    return {"message": "Welcome to the Skin Disease Prediction API!"}

@app.post("/predict")
async def predict(images: UploadFile = File(...)):
    try:
        content = await images.read()
        result = predict_image(model, content)
        return result
    except Exception as e:
        return {"error": str(e)}
