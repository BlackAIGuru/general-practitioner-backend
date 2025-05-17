import asyncio
import os
import shutil
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.config import settings
from app.assistant import Assistant

os.makedirs("uploads", exist_ok=True)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.head('/health')
@app.get('/health')
def health_check():
    return 'ok'

@app.post('/upload')
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    """
    Upload a PDF file to be used as context for the conversation.
    Returns the file ID that can be used for the WebSocket connection.
    """
    file_path = f"uploads/{session_id}_{file.filename}"
    print(file_path)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"file_path": file_path, "filename": file.filename}

@app.websocket('/listen')
async def websocket_listen(websocket: WebSocket):
    await websocket.accept()
    assistant = Assistant(websocket)
    try:
        await asyncio.wait_for(assistant.run(), timeout=30000)
    except TimeoutError:
        print('Connection timeout')
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
