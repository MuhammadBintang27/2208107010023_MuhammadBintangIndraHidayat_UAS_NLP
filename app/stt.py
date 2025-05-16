import os
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path ke folder utilitas STT
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")

# Path ke binary whisper-cli
WHISPER_BINARY = os.path.join(WHISPER_DIR, "build", "bin", "whisper-cli.exe")

# Path ke file model Whisper (contoh: ggml-large-v3-turbo.bin)
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-large-v3-turbo.bin")

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Transkrip file audio menggunakan whisper.cpp CLI
    Args:
        file_bytes (bytes): Isi file audio
        file_ext (str): Ekstensi file, default ".wav"
    Returns:
        str: Teks hasil transkripsi
    """
    # Create a more permanent temp directory (not in /tmp which may be cleaned up)
    temp_dir = os.path.join(tempfile.gettempdir(), "voice_assistant_stt")
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
    result_path = os.path.join(temp_dir, "transcription.txt")

    # Simpan audio ke file temporer
    with open(audio_path, "wb") as f:
        f.write(file_bytes)

    # Validate the input file exists and has content
    if not os.path.exists(audio_path):
        return "[ERROR] Failed to create audio file"
        
    if os.path.getsize(audio_path) == 0:
        return "[ERROR] Empty audio file"

    # Jalankan whisper.cpp dengan subprocess
    cmd = [
        WHISPER_BINARY,
        "-m", WHISPER_MODEL_PATH,
        "-f", audio_path,
        "-otxt",
        "-of", os.path.join(temp_dir, "transcription")
    ]

    try:
        print(f"[INFO] Running STT command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[INFO] STT stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Whisper failed: {e}")
        print(f"[ERROR] Whisper stderr: {e.stderr}")
        return f"[ERROR] Whisper failed: {e}"
    
    # Baca hasil transkripsi
    try:
        if not os.path.exists(result_path):
            print(f"[ERROR] Transcription file not found at: {result_path}")
            return "[ERROR] Transcription file not found"
            
        with open(result_path, "r", encoding="utf-8") as result_file:
            transcript = result_file.read().strip()
            
        # Check if transcript is empty
        if not transcript:
            return "[ERROR] Empty transcript generated"
            
        return transcript
    except Exception as e:
        print(f"[ERROR] Failed to read transcription: {str(e)}")
        return f"[ERROR] Failed to read transcription: {str(e)}"