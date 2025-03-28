# RAG Chat Application README
This is a FastAPI-based RAG (Retrieval-Augmented Generation) Chat Application that supports streaming chat responses, PDF uploads for text extraction and analysis, and chat history retrieval. This README provides instructions on how to set up and test the API using Postman.

## Prerequisites
1. **Python 3.8+:** Ensure Python is installed on your system.
2. **Dependencies:** Install the required Python packages by running:
```bash

pip install fastapi uvicorn pydantic asyncio typing
```
(Note: Additional dependencies like `utils.functionlty` are assumed to be custom modules. Ensure they are available in your project.)
3. **Postman:** Install Postman to test the API endpoints.
4. **Run the Server:** Start the FastAPI server with:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Replace `main` with the filename containing the code (e.g., `app.py`).
The server will be available at `http://localhost:8000`.

## API Endpoints
The application exposes three main endpoints:

1. **POST /chat/stream:** Stream chat responses based on user input.
2. **POST /pdf/upload:** Upload a PDF file and store its analyzed text for a session.
3. **GET /chat_history:** Retrieve the chat history for a specific session.
Below are detailed instructions for testing each endpoint in Postman.
---
1. ### POST /chat/stream
**Description:** Streams a chat response based on user input and session ID. Supports text input and optional pre-uploaded PDF text.

- **URL:** `http://localhost:8000/chat/stream`
- **Method:** `POST`
- **Body** (raw JSON):
```json
{
  "input": "Hello, how can you assist me today?",
  "session_id": "test_session_123"
}
```
- **Response:** A stream of text in Server-Sent Events (SSE) format.
    - Example: `data: Hello! I'm here to assist you. How can I help?\n\n`
- Notes:
    - Use Postman’s “SSE” feature or a tool like `curl` to observe the streaming response.
    - The `session_id` is required to track the conversation.
2. ## POST /pdf/upload
**Description:** Uploads a PDF file, extracts its text, analyzes it, and stores it for use in the chat session.

- **URL:** `http://localhost:8000/pdf/upload`
- **Method:** `POST`
- **Body** (form-data):
    - Key: `session_id` | Value: `test_session_123` (type: text)
    - Key: `file` | Value: (select a `.pdf` file) (type: file)
- **Response** (JSON):
```json
{
  "status": "success",
  "message": "PDF Uploaded successfully",
  "analysis": "Extracted and analyzed text from the PDF"
}
```
- Notes:
    - The uploaded PDF’s analyzed text is stored temporarily and appended to the next `/chat/stream` request for the same `session_id`.
    - Only PDF files are accepted (`application/pdf`).
3. ## GET /chat_history
**Description:**Retrieves the chat history for a specific session.

- **URL**: `http://localhost:8000/chat_history`
- **Method**: `GET`
- **Headers:** None required
- **Body** (form-data)
    - Key: `session_id` | Value: `test_session_123` (type: text)

- **Response** (JSON):
```json
{
  "chat_history": [
    {"sender": "user", "message": "Hello, how can you assist me today?"},
    {"sender": "assistant", "message": "Hello! I'm here to assist you. How can I help?"}
  ]
}
```
- **Notes:**
    - Returns a 404 error if the   `session_id` does not exist.
## Testing Workflow in Postman
1. **Start a Session with Chat:**
    - Send a POST request to `/chat/stream` with a unique `session_id` and an initial message.
    - Observe the streamed response in Postman.
2. **Upload a PDF:**
    - Send a POST request to `/pdf/upload` with the same `session_id` and a PDF file.
    - Check the response for the analyzed text.
3. **Chat with PDF Context:**
    - Send another POST request to `/chat/stream` with the same `session_id`.
    - The input will include the analyzed PDF text automatically.
4. **Retrieve Chat History:**
    - Send a GET request to `/chat_history` with the `session_id` to see the full conversation.