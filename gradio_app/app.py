import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
import uuid
import numpy as np
import time
import logging
import json
from datetime import datetime
import traceback

# Configure logging with improved structure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice-assistant-frontend")

# Add file handler to save logs
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"frontend_{datetime.now().strftime('%Y%m%d')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Configuration constants
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
VOICE_CHAT_ENDPOINT = f"{BACKEND_URL}/voice-chat"
REQUEST_TIMEOUT = 460  # seconds

# Create temp directory for audio files
TEMP_DIR = os.path.join(tempfile.gettempdir(), "voice_assistant")
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Temporary directory created at: {TEMP_DIR}")

def check_backend_health():
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
        logger.warning(f"Backend health check failed with status code: {response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Backend health check failed: {str(e)}")
        return False

def voice_chat(audio, session_id):
    """
    Process audio input, send to backend and return audio response with intermediate results.
    
    Args:
        audio: Tuple (sample_rate, audio_data) from gr.Audio(type="numpy")
        session_id: Unique session identifier
        
    Returns:
        tuple: (output_audio_path, status, transcript, llm_raw_response, llm_g2p_response, tts_status)
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{session_id}|{request_id}] Processing new voice request")
    
    # Validate input audio
    if audio is None:
        logger.error(f"[{session_id}|{request_id}] No audio input provided")
        return None, f"Error: No audio input provided", None, None, None, None
    
    try:
        sr, audio_data = audio
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            logger.error(f"[{session_id}|{request_id}] Invalid or empty audio data")
            return None, f"Error: Invalid or empty audio data", None, None, None, None
        
        if not isinstance(sr, int) or sr <= 0:
            logger.error(f"[{session_id}|{request_id}] Invalid sample rate: {sr}")
            return None, f"Error: Invalid sample rate: {sr}", None, None, None, None
        
        logger.info(f"[{session_id}|{request_id}] Audio data shape: {audio_data.shape}, Sample rate: {sr}")
        logger.debug(f"[{session_id}|{request_id}] Audio data min: {audio_data.min()}, max: {audio_data.max()}")
        
        # Normalize audio if needed
        if audio_data.dtype != np.int16:
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                logger.info(f"[{session_id}|{request_id}] Converting audio from {audio_data.dtype} to int16")
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                logger.warning(f"[{session_id}|{request_id}] Unexpected audio data type: {audio_data.dtype}")
        
        # Save audio to temporary file
        audio_path = os.path.join(TEMP_DIR, f"input_{session_id}_{request_id}.wav")
        
        scipy.io.wavfile.write(audio_path, sr, audio_data)
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"[{session_id}|{request_id}] Saved audio file is empty")
            return None, f"Error: Could not save audio properly", None, None, None, None
        
        logger.info(f"[{session_id}|{request_id}] Audio saved to: {audio_path} ({file_size} bytes)")
        
        # Check if backend is available before proceeding
        if not check_backend_health():
            logger.error(f"[{session_id}|{request_id}] Backend is not available")
            return None, f"Error: Backend service is not available. Please check if the server is running.", None, None, None, None
        
        # Send file to backend
        try:
            with open(audio_path, "rb") as f:
                file_content = f.read()
                logger.debug(f"[{session_id}|{request_id}] File content size before sending: {len(file_content)} bytes")
                
                files = {"file": (f"voice_{request_id}.wav", file_content, "audio/wav")}
                headers = {"X-Session-ID": session_id, "X-Request-ID": request_id}
                
                logger.info(f"[{session_id}|{request_id}] Sending request to backend: {VOICE_CHAT_ENDPOINT}")
                start_time = time.time()
                
                response = requests.post(
                    VOICE_CHAT_ENDPOINT, 
                    files=files, 
                    headers=headers,
                    timeout=REQUEST_TIMEOUT
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"[{session_id}|{request_id}] Backend responded in {elapsed_time:.2f} seconds with status: {response.status_code}")
                
                if response.status_code == 200:
                    # Parse JSON response
                    response_data = response.json()
                    transcript = response_data.get("transcript", "Tidak ada transkripsi")
                    llm_raw_response = response_data.get("llm_raw_response", "Tidak ada respons AI (asli)")
                    llm_g2p_response = response_data.get("llm_g2p_response", "Tidak ada respons AI (G2P)")
                    audio_base64 = response_data.get("audio_base64", None)
                    status = response_data.get("status", "success")
                    
                    if status != "success":
                        logger.error(f"[{session_id}|{request_id}] Backend reported error: {response_data.get('error', 'Unknown error')}")
                        return None, f"Error dari server: {response_data.get('error', 'Unknown error')}", transcript, llm_raw_response, llm_g2p_response, None
                    
                    # Save response audio from base64
                    output_audio_path = os.path.join(TEMP_DIR, f"response_{session_id}_{request_id}.wav")
                    if audio_base64:
                        import base64
                        audio_content = base64.b64decode(audio_base64)
                        with open(output_audio_path, "wb") as f:
                            f.write(audio_content)
                        
                        file_size = os.path.getsize(output_audio_path)
                        if file_size == 0:
                            logger.error(f"[{session_id}|{request_id}] Output audio file is empty")
                            return None, f"Error: Received empty response from server", transcript, llm_raw_response, llm_g2p_response, None
                        
                        logger.info(f"[{session_id}|{request_id}] Output audio saved to: {output_audio_path} ({file_size} bytes)")
                        tts_status = f"Suara respons AI telah dihasilkan ({file_size} bytes)"
                    else:
                        logger.error(f"[{session_id}|{request_id}] No audio data in response")
                        return None, f"Error: No audio data received", transcript, llm_raw_response, llm_g2p_response, None
                    
                    # Clean up input file
                    try:
                        os.remove(audio_path)
                        logger.debug(f"[{session_id}|{request_id}] Cleaned up input file")
                    except Exception as e:
                        logger.warning(f"[{session_id}|{request_id}] Failed to clean up input file: {str(e)}")
                    
                    return output_audio_path, f"✅ Berhasil dalam {elapsed_time:.1f} detik", transcript, llm_raw_response, llm_g2p_response, tts_status
                else:
                    # Try to extract error message
                    try:
                        error_data = response.json()
                        error_message = error_data.get("error", "Unknown error")
                        transcript = error_data.get("transcript", None)
                        llm_raw_response = error_data.get("llm_raw_response", None)
                        llm_g2p_response = error_data.get("llm_g2p_response", None)
                        logger.error(f"[{session_id}|{request_id}] Backend error: {error_message}")
                        return None, f"Error dari server: {error_message}", transcript, llm_raw_response, llm_g2p_response, None
                    except:
                        logger.error(f"[{session_id}|{request_id}] Backend error: {response.text}")
                        return None, f"Error dari server: Status {response.status_code}", None, None, None, None
        except requests.exceptions.Timeout:
            logger.error(f"[{session_id}|{request_id}] Request to backend timed out after {REQUEST_TIMEOUT} seconds")
            return None, f"Error: Koneksi ke server timeout. Coba lagi atau hubungi administrator.", None, None, None, None
        except requests.exceptions.RequestException as e:
            logger.error(f"[{session_id}|{request_id}] Failed to connect to backend: {str(e)}")
            return None, f"Error koneksi: {str(e)}", None, None, None, None
    except Exception as e:
        logger.error(f"[{session_id}|{request_id}] Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Error tak terduga: {str(e)}", None, None, None, None

def process_submission(audio, session_state):
    """
    Process audio input from Gradio and update UI status with intermediate results.
    
    Args:
        audio: Input from gr.Audio
        session_state: Session state dictionary
        
    Yields:
        tuple: (path_audio_output, status_text, conversation_history, transcript, llm_raw_response, llm_g2p_response, tts_status)
    """
    # Get or generate session ID
    session_id = session_state.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session_state["session_id"] = session_id
        logger.info(f"New session created: {session_id}")
    
    # Handle empty audio input
    if audio is None:
        message = "Tidak ada rekaman suara. Silakan rekam suara terlebih dahulu."
        yield None, message, session_state["conversation"], "", "", "", ""
        return
    
    # Process audio
    sr, data = audio
    logger.debug(f"[{session_id}] Sample rate: {sr}, Data shape: {data.shape}")
    
    # Update UI with processing status
    status_text = "🔄 Memproses audio..."
    yield None, status_text, session_state["conversation"], "Memproses transkripsi...", "Menunggu respons AI (asli)...", "Menunggu respons AI (G2P)...", "Menyiapkan konversi suara..."
    
    # Process audio and get response
    try:
        result, status, transcript, llm_raw_response, llm_g2p_response, tts_status = voice_chat(audio, session_id)
        
        # Update conversation history
        if result:
            session_state["conversation"] = f"{session_state['conversation']}\n\n✅ Percakapan baru selesai pada {datetime.now().strftime('%H:%M:%S')}\nTranskripsi: {transcript}\nRespon AI (Asli): {llm_raw_response}\nRespon AI (G2P): {llm_g2p_response}"
        else:
            session_state["conversation"] = f"{session_state['conversation']}\n\n❌ Error pada {datetime.now().strftime('%H:%M:%S')}: {status}"
        
        # Return results to UI
        yield result, status, session_state["conversation"], transcript or "Tidak ada transkripsi", llm_raw_response or "Tidak ada respons AI (asli)", llm_g2p_response or "Tidak ada respons AI (G2P)", tts_status or "Gagal menghasilkan suara"
    except Exception as e:
        logger.error(f"[{session_id}] Exception in process_submission: {str(e)}")
        logger.error(traceback.format_exc())
        error_message = f"Error: {str(e)}"
        session_state["conversation"] = f"{session_state['conversation']}\n\n❌ Error: {str(e)}"
        yield None, error_message, session_state["conversation"], "", "", "", ""

def create_ui():
    """Create and configure the Gradio UI"""
    # Initialize session state
    session_state = {"session_id": str(uuid.uuid4()), "conversation": "🎙️ Riwayat Percakapan"}
    
    # Custom theme
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_hover="*neutral_200",
        block_title_text_weight="600",
        block_border_width="1px",
        block_shadow="0 1px 2px 0 rgb(0 0 0 / 0.05)",
    )
    
    # Create UI
    with gr.Blocks(theme=theme, analytics_enabled=False) as demo:
        gr.Markdown(
            """
            # 🎙️ Voice AI Assistant
            
            Berbicara langsung dengan asisten AI melalui perekaman suara dan dapatkan respons audio interaktif.
            Lihat prosesnya secara langsung: transkripsi suara, respons AI (asli dan G2P), dan konversi ke suara.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Left sidebar with status and history
                status = gr.Textbox(
                    label="Status",
                    value="Siap menerima perintah suara",
                    lines=1,
                    interactive=False
                )
                
                conversation_history = gr.TextArea(
                    label="Riwayat Percakapan",
                    value=session_state["conversation"],
                    lines=10,
                    interactive=False,
                    autoscroll=True
                )
                
                with gr.Accordion("ℹ️ Bantuan", open=False):
                    gr.Markdown(
                        """
                        ### Cara Menggunakan Voice Assistant
                        
                        1. Klik tombol mikrofon untuk mulai merekam
                        2. Bicara dengan suara yang jelas
                        3. Klik tombol stop untuk mengakhiri rekaman
                        4. Klik tombol "Kirim" untuk memproses
                        5. Lihat transkripsi, respons AI (asli dan G2P), dan dengar suara respons
                        
                        ### Tips
                        - Gunakan mikrofon eksternal untuk kualitas suara lebih baik
                        - Bicara dengan tempo normal dan artikulasi jelas
                        - Hindari kebisingan latar belakang
                        """
                    )
            
            with gr.Column(scale=2):
                # Main content area
                with gr.Group(elem_classes="gradio-box"):
                    gr.Markdown("## 🎤 Rekam Pertanyaan Anda")
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="",
                        interactive=True,
                        elem_classes="audio-input"
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Hapus", variant="secondary")
                        submit_btn = gr.Button("🚀 Kirim", variant="primary")
                
                with gr.Group(elem_classes="gradio-box"):
                    gr.Markdown("## 🔊 Respons Asisten")
                    audio_output = gr.Audio(
                        type="filepath",
                        label="",
                        interactive=False,
                        autoplay=True,
                        elem_classes="audio-output"
                    )
                    
                    with gr.Row():
                        download_btn = gr.Button("💾 Unduh", variant="secondary")
                
                with gr.Group(elem_classes="gradio-box"):
                    gr.Markdown("## 📝 Proses Asisten")
                    transcript_output = gr.Textbox(
                        label="Transkripsi Suara",
                        value="Menunggu transkripsi...",
                        lines=3,
                        interactive=False,
                        info="Hasil konversi suara Anda ke teks oleh sistem Speech-to-Text."
                    )
                    llm_raw_response_output = gr.Textbox(
                        label="Respon AI (Asli)",
                        value="Menunggu respons AI (asli)...",
                        lines=3,
                        interactive=False,
                        info="Teks asli dari model AI sebelum konversi G2P."
                    )
                    llm_g2p_response_output = gr.Textbox(
                        label="Respon AI (G2P)",
                        value="Menunggu respons AI (G2P)...",
                        lines=3,
                        interactive=False,
                        info="Teks dari model AI setelah konversi G2P untuk pengucapan yang lebih akurat."
                    )
                    tts_status_output = gr.Textbox(
                        label="Konversi ke Suara",
                        value="Menunggu konversi suara...",
                        lines=2,
                        interactive=False,
                        info="Status pembuatan suara respons AI oleh sistem Text-to-Speech."
                    )
        
        # Button actions
        submit_btn.click(
            fn=process_submission,
            inputs=[audio_input, gr.State(session_state)],
            outputs=[audio_output, status, conversation_history, transcript_output, llm_raw_response_output, llm_g2p_response_output, tts_status_output],
            show_progress=True
        )
        
        clear_btn.click(
            fn=lambda: (None, "Audio input cleared", session_state["conversation"], "Menunggu transkripsi...", "Menunggu respons AI (asli)...", "Menunggu respons AI (G2P)...", "Menunggu konversi suara..."),
            inputs=[],
            outputs=[audio_input, status, conversation_history, transcript_output, llm_raw_response_output, llm_g2p_response_output, tts_status_output]
        )
        
        # CSS for better styling
        demo.load(None, js="""
        () => {
            // Add custom styles
            const style = document.createElement('style');
            style.textContent = `
                .audio-input, .audio-output {
                    border-radius: 8px;
                    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                }
                .gradio-box {
                    border-radius: 12px;
                    margin-bottom: 16px;
                    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
                    padding: 16px;
                    background-color: #fff;
                }
                .gradio-button.primary {
                    background-color: #4f46e5;
                    font-weight: 600;
                }
                .gradio-button.primary:hover {
                    background-color: #4338ca;
                }
            `;
            document.head.appendChild(style);
            
            // Log that the UI has loaded completely
            console.log('Voice Assistant UI loaded successfully');
        }
        """)
        
    return demo

if __name__ == "__main__":
    # Verify dependencies are installed
    try:
        import scipy
        import numpy
        import requests
        import gradio
        logger.info("All dependencies are installed")
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.info("Install dependencies with: pip install gradio==4.43.0 scipy numpy requests")
        exit(1)
    
    # Check if backend is available
    if check_backend_health():
        logger.info(f"Successfully connected to backend at {BACKEND_URL}")
    else:
        logger.warning(f"Backend at {BACKEND_URL} is not available. UI will still launch, but processing may fail.")
    
    # Launch the app
    logger.info("Launching Voice Assistant UI")
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False,
        favicon_path="https://api.iconify.design/material-symbols:mic.svg"
    )