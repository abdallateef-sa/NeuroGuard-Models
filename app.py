import os
import pandas as pd
import joblib
import torch
import torch.nn as nn
import io
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
import numpy as np
from keras.models import load_model
from uuid import uuid4
from utils.functionlty import bot_func, create_bot_for_selected_bot, extract_pdf_text, analysis_text
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
from typing import Dict, List, Optional, Any

os.makedirs("models", exist_ok=True)

app = FastAPI()

stroke_model_path = hf_hub_download(repo_id="abdallateef/test", filename="prediction_model.joblib")
detection_model_path = hf_hub_download(repo_id="abdallateef/test", filename="detection_model.h5")
srgan_model_path = hf_hub_download(repo_id="abdallateef/test", filename="srgan_model.pth")
denoising_model_path = hf_hub_download(repo_id="abdallateef/test", filename="denoising_model.pth")
cyclegan_model_path = hf_hub_download(repo_id="abdallateef/test", filename="cyclegan_model.pth")

stroke_model = joblib.load(stroke_model_path)
feature_names = stroke_model.feature_names_in_
print("Model features:", feature_names)
## Image models
image_model = load_model(detection_model_path)

def create_srgan_generator():
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )
    return model

def create_denoising_generator():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, padding=1)
    )

def create_cyclegan_generator():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, padding=1)
    )

@app.post("/upload-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).resize((224, 224)).convert("RGB")
        image = np.array(image) / 255.0
        image = image.reshape((1, 224, 224, 3))
        pred = image_model.predict(image)
        pred_label = 1 if pred >= 0.50 else 0
        classes = ['Normal', 'Stroke']
        return {"prediction": classes[pred_label]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


## Chat Bot API
class ChatRequest(BaseModel):
    input: str
    session_id: str

# Store bots, chat histories, and uploaded texts for different sessions
active_bots: Dict[str, Any] = {}
chat_histories: Dict[str, List[Dict[str, str]]] = {}
uploaded_texts: Dict[str, Optional[str]] = {}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming endpoint for chat responses"""
    if not request.session_id or not request.input:
        raise HTTPException(status_code=400, detail="Session ID and input are required")

    # Get or create chat history for this session
    session_chat_history = chat_histories.setdefault(request.session_id, [])

    # Get uploaded text for this session and remove it after use
    uploaded_text = uploaded_texts.pop(request.session_id, None)
    user_input_with_pdf = f"{request.input}\n\nExtracted PDF Text:\n{uploaded_text}" if uploaded_text else request.input

    # Append user's message to chat history
    session_chat_history.append({"sender": "user", "message": user_input_with_pdf})

    # Get or create bot for this session
    if request.session_id not in active_bots:
        embeddings = "BAAI/bge-base-en-v1.5"
        vdb_dir = "Stroke_vdb"
        sys_prompt_dir = "assist/prompt.txt"
        bot = create_bot_for_selected_bot("default", embeddings, vdb_dir, sys_prompt_dir)
        active_bots[request.session_id] = bot
    else:
        bot = active_bots[request.session_id]

    async def stream_response():
        try:
            response_chunks = []
            for chunk in bot_func(bot, user_input_with_pdf, request.session_id):
                response_chunks.append(chunk)
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
            full_response = "".join(response_chunks)
            session_chat_history.append({"sender": "assistant", "message": full_response})
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/pdf/upload")
async def upload_file(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a PDF file and store its analyzed text for the session"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        text = extract_pdf_text(file.file)
        analysis = analysis_text(text)
        uploaded_texts[session_id] = analysis
        return {
            "status": "success",
            "message": "PDF Uploaded successfully",
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/chat_history")
async def get_chat_history(session_id: str = Form(...)):
    """Retrieve chat history for a specific session"""
    if session_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"chat_history": chat_histories[session_id]}



@app.post("/predict/srgan/")
async def predict_srgan(file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_srgan_generator().to(device)
    model.load_state_dict(torch.load(srgan_model_path, map_location=device))
    model.eval()
    image = Image.open(file.file).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/predict/denoising/")
async def predict_denoising(file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(denoising_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    image = Image.open(file.file).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/predict/cyclegan/")
async def predict_cyclegan(file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(cyclegan_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    image = Image.open(file.file).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")




# Initialize the FastAPI app

# Load the pre-trained model (assumed to be saved as 'model.pkl')
model = joblib.load('stroke_prediction_pipeline_optimized.joblib')

# Define the input data structure using Pydantic
class PatientData(BaseModel):
    gender: str            # 'Male', 'Female'
    age: float             # e.g., 67, 0.64
    hypertension: int      # 0 or 1
    heart_disease: int     # 0 or 1
    ever_married: str      # 'Yes', 'No'
    work_type: str         # 'Private', 'Self-employed', 'Govt_job', 'children'
    Residence_type: str    # 'Urban', 'Rural'
    avg_glucose_level: float  # e.g., 228.69
    bmi: float             # e.g., 36.6
    smoking_status: str    # 'formerly smoked', 'never smoked', 'smokes', 'Unknown'

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    # Extract input values
    gender = data.gender
    age = data.age
    hypertension = data.hypertension
    heart_disease = data.heart_disease
    ever_married = data.ever_married
    work_type = data.work_type
    residence_type = data.Residence_type
    avg_glucose_level = data.avg_glucose_level
    bmi = data.bmi
    smoking_status = data.smoking_status

    # Encode categorical variables to match model features
    gender_male = 1 if gender == 'Male' else 0
    ever_married_yes = 1 if ever_married == 'Yes' else 0
    residence_type_urban = 1 if residence_type == 'Urban' else 0
    
    # One-hot encode work_type (reference category: 'Govt_job')
    work_type_private = 1 if work_type == 'Private' else 0
    work_type_self_employed = 1 if work_type == 'Self-employed' else 0
    work_type_children = 1 if work_type == 'children' else 0
    
    # One-hot encode smoking_status (reference category: 'Unknown')
    smoking_status_formerly_smoked = 1 if smoking_status == 'formerly smoked' else 0
    smoking_status_never_smoked = 1 if smoking_status == 'never smoked' else 0
    smoking_status_smokes = 1 if smoking_status == 'smokes' else 0

    # Create feature array in the order expected by the model
    features = [
        age,
        hypertension,
        heart_disease,
        avg_glucose_level,
        bmi,
        gender_male,
        ever_married_yes,
        work_type_private,
        work_type_self_employed,
        work_type_children,
        residence_type_urban,
        smoking_status_formerly_smoked,
        smoking_status_never_smoked,
        smoking_status_smokes
    ]

    # Convert to 2D array for model prediction
    input_data = np.array([features])
    
    # Make prediction
    probabilities = model.predict_proba(input_data)[0]
    
    # Return the result as a JSON response
    stroke_prob = round(probabilities[1] * 100, 1)
    return {
        "stroke_probability": stroke_prob
    }
# Welcome endpoint
@app.get("/")
def read_root():
    return {"message": "Stroke Prediction API"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


    
