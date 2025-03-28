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

image_model = load_model(detection_model_path)

bot = create_bot_for_selected_bot(
    embeddings="BAAI/bge-base-en-v1.5",
    vdb_dir="Stroke_vdb",
    sys_prompt_dir="assist/prompt.txt",
    name="stroke RAG"
)

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

class StrokePredictionInput(BaseModel):
    age: int
    hypertension: int
    heart_disease: int
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    residence_type: str
    smoking_status: str

class ChatRequest(BaseModel):
    user_input: str

def preprocess_input(data: StrokePredictionInput):
    columns = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Female', 'gender_Male', 'gender_Other',
        'ever_married_No', 'ever_married_Yes',
        'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children',
        'Residence_type_Rural', 'Residence_type_Urban',
        'smoking_status_Unknown', 'smoking_status_formerly smoked',
        'smoking_status_never smoked', 'smoking_status_smokes'
    ]
    df = pd.DataFrame(0, index=[0], columns=columns)
    df['age'] = data.age
    df['hypertension'] = data.hypertension
    df['heart_disease'] = data.heart_disease
    df['avg_glucose_level'] = data.avg_glucose_level
    df['bmi'] = data.bmi
    df[f'gender_{data.gender}'] = 1
    df[f'ever_married_{data.ever_married}'] = 1
    df[f'work_type_{data.work_type}'] = 1
    df[f'Residence_type_{data.residence_type}'] = 1
    df[f'smoking_status_{data.smoking_status}'] = 1
    df = df[feature_names]
    return df

@app.post("/predict/")
async def predict_stroke(data: StrokePredictionInput):
    try:
        input_df = preprocess_input(data)
        prediction = stroke_model.predict(input_df)
        probability = stroke_model.predict_proba(input_df)[0][1]
        return {"prediction": int(prediction[0]), "probability": float(probability)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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



app = FastAPI(title="RAG Chat Application")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
