# NeuroGuard-Models README

This is a FastAPI-based application that combines:

- **RAG Chat** (Retrieval-Augmented Generation) with streaming responses  
- **PDF Upload & Analysis**  
- **Chat History** retrieval  
- **Stroke Prediction** via a pre-trained ML pipeline  
- **Image Classification** (“Normal” vs. “Stroke”)  
- **Super-Resolution** (SRGAN)  
- **Denoising**  
- **Style-Transfer** (CycleGAN)  

Use this README to set up, run, and test all API endpoints (e.g. via Postman).

👉 [Models On Hugging Face](https://huggingface.co/abdallateef/test). 

---

## Prerequisites

1. **Python 3.8+**  
2. **Git** (to clone the repo)  
3. **Postman** (or any HTTP client) for testing  

---

## Installation

```bash
git clone https://github.com/abdallateef-sa/NeuroGuard-Models.git
cd NeuroGuard-Models

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI** available at: `http://localhost:8000/docs`   

---

## API Endpoints

### 1. Stream Chat (RAG)

- **URL**: `POST /chat/stream`  
- **Body (JSON)**  
  ```json
  {
    "input": "Hello, how can you assist me?",
    "session_id": "session_123"
  }
  ```
- **Response**: Server-Sent Events stream (`text/event-stream`)  
- **Notes**:  
  - Use Postman’s “Event Stream” view or `curl -N` to follow the stream.  
  - Maintains history per `session_id`.  

---

### 2. Upload & Analyze PDF

- **URL**: `POST /pdf/upload`  
- **Body (form-data)**  
  - `session_id`: _string_  
  - `file`: _application/pdf_  
- **Response (JSON)**  
  ```json
  {
    "status": "success",
    "message": "PDF Uploaded successfully",
    "analysis": "<extracted and summarized text>"
  }
  ```
- **Notes**:  
  - The extracted text is auto-appended to the next `/chat/stream` call for that session.  

---

### 3. Get Chat History

- **URL**: `GET /chat_history`  
- **Query/form-data**:  
  - `session_id`: _string_  
- **Response (JSON)**  
  ```json
  {
    "chat_history": [
      { "sender": "user",      "message": "Hi" },
      { "sender": "assistant", "message": "Hello!" }
    ]
  }
  ```
- **Error**: `404` if `session_id` not found.  

---

### 4. Stroke Prediction

- **URL**: `POST /predict`  
- **Body (JSON)** _(Pydantic `PatientData` model)_  
  ```json
  {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "never smoked"
  }
  ```
- **Response (JSON)**  
  ```json
  { "stroke_probability": 23.7 }
  ```
- **Notes**:  
  - Returns the % probability of a stroke.  

---

### 5. Image Classification

- **URL**: `POST /upload-image/`  
- **Body (form-data)**:  
  - `file`: image (`.png`, `.jpg`, etc.)  
- **Response (JSON)**  
  ```json
  { "prediction": "Stroke" }
  ```

---

### 6. Super-Resolution (SRGAN)

- **URL**: `POST /predict/srgan/`  
- **Body (form-data)**:  
  - `file`: image  
- **Response**: PNG image (media_type=`image/png`)  

---

### 7. Image Denoising

- **URL**: `POST /predict/denoising/`  
- **Body (form-data)**:  
  - `file`: image  
- **Response**: PNG image (media_type=`image/png`)  

---

### 8. Style Transfer (CycleGAN)

- **URL**: `POST /predict/cyclegan/`  
- **Body (form-data)**:  
  - `file`: image  
- **Response**: PNG image (media_type=`image/png`)  

---

## Testing with Postman

1. **Start a chat session**  
   - `POST /chat/stream` → follow SSE stream  
2. **Upload a PDF**  
   - `POST /pdf/upload` → confirm `analysis` in JSON  
3. **Chat with PDF context**  
   - `POST /chat/stream` again with same `session_id`  
4. **Retrieve history**  
   - `GET /chat_history?session_id=...`  
5. **Stroke prediction & image endpoints**  
   - Use their respective `POST` endpoints with JSON or form-data.  

