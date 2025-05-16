from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import logging
import uuid
import time
from datetime import datetime
import traceback
from typing import Optional
import base64

# Configure logging with more structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice-assistant-backend")

# Add file handler to save logs
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"backend_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

app = FastAPI(title="Voice AI Assistant API")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a function to log request info
def log_request_info(request: Request, request_id: str):
    """Log detailed request information"""
    client_host = request.client.host if request.client else "Unknown"
    headers = {k: v for k, v in request.headers.items()}
    logger.info(f"[{request_id}] Request received from {client_host}")
    logger.debug(f"[{request_id}] Headers: {headers}")
    
@app.get("/")
def read_root():
    return {"message": "Voice AI Assistant API is running", "status": "ok"}

@app.get("/health")
def health_check():
    """Endpoint for health checking"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    """
    Process voice chat workflow and return intermediate results:
    1. Receive audio file from frontend
    2. Convert speech to text using Whisper
    3. Generate response using Gemini
    4. Convert response to speech
    5. Return JSON with transcript, LLM responses (raw and G2P), and audio
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    log_request_info(request, request_id)
    logger.info(f"[{request_id}] Processing voice chat with file: {file.filename}")
    
    try:
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"[{request_id}] File received: {file_size} bytes")
        
        if not contents or file_size == 0:
            logger.error(f"[{request_id}] Empty file received")
            return JSONResponse(
                status_code=400,
                content={"error": "Empty file", "request_id": request_id, "status": "error"}
            )
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = ['.wav', '.mp3', '.ogg', '.m4a']
        if file_ext not in allowed_extensions:
            logger.error(f"[{request_id}] Unsupported file extension: {file_ext}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file extension: {file_ext}", "request_id": request_id, "status": "error"}
            )
        
        # Validate content type (optional, requires python-magic)
        try:
            import magic
            mime = magic.Magic(mime=True)
            content_type = mime.from_buffer(contents)
            if not content_type.startswith('audio/'):
                logger.error(f"[{request_id}] Invalid content type: {content_type}")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid file type: {content_type}", "request_id": request_id, "status": "error"}
                )
        except ImportError:
            logger.debug(f"[{request_id}] python-magic not installed, skipping content type validation")
        
        temp_dir = tempfile.mkdtemp(prefix=f"voice_assistant_{request_id}_")
        logger.info(f"[{request_id}] Created temporary directory: {temp_dir}")
        
        debug_file = os.path.join(temp_dir, f"input{file_ext}")
        with open(debug_file, "wb") as f:
            f.write(contents)
        logger.debug(f"[{request_id}] Saved input file for debugging at: {debug_file}")
        
        logger.info(f"[{request_id}] Starting speech-to-text processing")
        try:
            from app.stt import transcribe_speech_to_text
            transcript = transcribe_speech_to_text(contents, file_ext=file_ext)
            if isinstance(transcript, str) and transcript.startswith("[ERROR]"):
                logger.error(f"[{request_id}] STT error: {transcript}")
                return JSONResponse(
                    status_code=500,
                    content={"error": transcript, "request_id": request_id, "status": "error", "transcript": transcript}
                )
        except Exception as dispersal:
            logger.error(f"[{request_id}] Exception in STT: {str(dispersal)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"Speech-to-text error: {str(dispersal)}", "request_id": request_id, "status": "error"}
            )
        
        logger.info(f"[{request_id}] Transcribed: {transcript}")
        
        logger.info(f"[{request_id}] Generating LLM response")
        try:
            from app.llm import generate_response
            response = generate_response(transcript)
            if isinstance(response, dict) and "error" in response:
                logger.error(f"[{request_id}] LLM error: {response['error']}")
                return JSONResponse(
                    status_code=500,
                    content={"error": response['error'], "request_id": request_id, "status": "error", "transcript": transcript}
                )
            llm_raw_response = response["raw_response"]
            llm_g2p_response = response["g2p_response"]
        except Exception as dispersal:
            logger.error(f"[{request_id}] Exception in LLM: {str(dispersal)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"LLM processing error: {str(dispersal)}", "request_id": request_id, "status": "error", "transcript": transcript}
            )
        
        logger.info(f"[{request_id}] LLM Raw Response: {llm_raw_response}")
        logger.info(f"[{request_id}] LLM G2P Response: {llm_g2p_response}")
        
        logger.info(f"[{request_id}] Starting text-to-speech processing")
        try:
            from app.tts import transcribe_text_to_speech
            audio_path = transcribe_text_to_speech(llm_g2p_response)
            if not audio_path or (isinstance(audio_path, str) and audio_path.startswith("[ERROR]")):
                logger.error(f"[{request_id}] TTS error: {audio_path}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to generate speech: {audio_path}", "request_id": request_id, "status": "error", "transcript": transcript, "llm_raw_response": llm_raw_response, "llm_g2p_response": llm_g2p_response}
                )
        except Exception as dispersal:
            logger.error(f"[{request_id}] Exception in TTS: {str(dispersal)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={"error": f"Text-to-speech error: {str(dispersal)}", "request_id": request_id, "status": "error", "transcript": transcript, "llm_raw_response": llm_raw_response, "llm_g2p_response": llm_g2p_response}
            )
        
        if not os.path.exists(audio_path):
            logger.error(f"[{request_id}] Audio file not found at: {audio_path}")
            return JSONResponse(
                status_code=500, 
                content={"error": f"Audio file not found at: {audio_path}", "request_id": request_id, "status": "error", "transcript": transcript, "llm_raw_response": llm_raw_response, "llm_g2p_response": llm_g2p_response}
            )
        
        file_size = os.path.getsize(audio_path)
        logger.info(f"[{request_id}] Audio file saved at: {audio_path} ({file_size} bytes)")
        
        # Encode audio file to base64 to include in JSON response
        with open(audio_path, "rb") as f:
            audio_content = f.read()
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        logger.info(f"[{request_id}] Preparing JSON response")
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"[{request_id}] Total processing time: {processing_time:.2f} seconds")
        
        # Clean up files
        try:
            os.remove(debug_file)
            os.remove(audio_path)
            logger.debug(f"[{request_id}] Cleaned up files: {debug_file}, {audio_path}")
        except Exception as dispersal:
            logger.warning(f"[{request_id}] Failed to clean up files: {str(dispersal)}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "request_id": request_id,
                "transcript": transcript,
                "llm_raw_response": llm_raw_response,
                "llm_g2p_response": llm_g2p_response,
                "audio_base64": audio_base64,
                "processing_time": round(processing_time, 2)
            }
        )
        
    except Exception as dispersal:
        logger.error(f"[{request_id}] Unhandled exception: {str(dispersal)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(dispersal)}", "request_id": request_id, "status": "error"}
        )

@app.post("/text-chat")
async def text_chat(request: Request):
    """Process text-only chat for debugging"""
    request_id = str(uuid.uuid4())
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided", "request_id": request_id}
            )
            
        logger.info(f"[{request_id}] Text chat request: {text}")
        
        # Generate response
        from app.llm import generate_response
        response = generate_response(text)
        if isinstance(response, dict) and "error" in response:
            logger.error(f"[{request_id}] LLM error: {response['error']}")
            return JSONResponse(
                status_code=500,
                content={"error": response['error'], "request_id": request_id}
            )
        
        logger.info(f"[{request_id}] LLM raw response: {response['raw_response']}")
        logger.info(f"[{request_id}] LLM G2P response: {response['g2p_response']}")
        
        return {
            "raw_response": response["raw_response"],
            "g2p_response": response["g2p_response"],
            "request_id": request_id
        }
    
    except Exception as e:
        logger.error(f"[{request_id}] Text chat error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": f"Text chat error: {str(e)}", "request_id": request_id}
        )
    
if __name__ == "__main__":
    logger.info("Starting Voice AI Assistant API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)