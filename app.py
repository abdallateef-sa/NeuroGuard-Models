import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
from keras.models import load_model
from uuid import uuid4
from utils.functionlty import bot_func, create_bot_for_selected_bot, extract_pdf_text, analysis_text
import torchvision.transforms as transforms
import torch
import torch.nn as nn


stroke_model = joblib.load(r"D:/FastAPI/models/prediction_model.joblib")
feature_names = stroke_model.feature_names_in_  # ميزات النموذج
print("Model features:", feature_names)  # Debug print

image_model = load_model("D:/FastAPI/models/detection_model.h5")

bot = create_bot_for_selected_bot(
    embeddings="BAAI/bge-base-en-v1.5",
    vdb_dir="Stroke_vdb",
    sys_prompt_dir="assist/prompt.txt",
    name="stroke RAG"
)

app = FastAPI()

#----
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

#----
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


        return {
            "prediction": int(prediction[0]),
            "probability": float(probability)
        }
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

@app.post("/chat/")
async def chat(request: ChatRequest):
    session_id = str(uuid4())  
    response = "".join(bot_func(bot, request.user_input, session_id=session_id))
    return {"response": response}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed!")

    text = extract_pdf_text(file.file)
    analysis = analysis_text(text)

    return {
        "filename": file.filename,
        "extracted_text": text,
        "analysis": analysis
    }

#---

# نقطة نهاية SRGAN بدون حفظ الصورة على القرص
@app.post("/predict/srgan/")
async def predict_srgan(file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_srgan_generator().to(device)
    model.load_state_dict(torch.load("models/srgan_model.pth", map_location=device))
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
    # تحميل النموذج كاملاً مع الأوزان
    model = torch.load("models/denoising_model.pth", map_location=device, weights_only=False)
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
    model = torch.load("models/cyclegan_model.pth", map_location=device, weights_only=False)
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


#---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
