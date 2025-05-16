# import json
# import os
# import google.generativeai as genai
# from dotenv import load_dotenv
# from g2p_id import G2P

# g2p = G2P()

# load_dotenv()

# # Gunakan os.getenv("NAMA_ENV_VARIABLE") untuk mengambil API Key dari file .env.
# # Pastikan di file .env terdapat baris: GEMINI_API_KEY=your_api_key
# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GOOGLE_API_KEY:
#     print("[ERROR] GEMINI_API_KEY not found in environment variables")

# MODEL = "gemini-2.0-flash"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# # Pastikan folder untuk history ada
# os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)

# # Prompt sistem yang digunakan untuk membimbing gaya respons LLM
# system_instruction = """ 
# You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian. 
# Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice. 
 
# Your answers must: 
# - Be written in polite and easily understandable Indonesian. 
# - Be short and to the point (maximum 2–3 sentences). 
# - Avoid repeating the user's question; respond directly with the answer. 
 
# Example tone: 
# User: Cuaca hari ini gimana? 
# Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat. 
 
# User: Kamu tahu siapa presiden Indonesia? 
# Assistant: Presiden Indonesia saat ini adalah Joko Widodo. 
 
# If you're unsure about an answer, be honest and say that you don't know. 
# """ 

# try:
#     genai.configure(api_key=GOOGLE_API_KEY)
#     model = genai.GenerativeModel(model_name=MODEL)
#     print(f"[INFO] Successfully configured Gemini API with model: {MODEL}")
# except Exception as e:
#     print(f"[ERROR] Failed to configure Gemini API: {e}")
#     model = None


# def save_chat_history(chat):
#     try:
#         # Ekstrak hanya field yang relevan dari history
#         history = [
#             {
#                 "role": message.role,
#                 "parts": [part.text for part in message.parts if hasattr(part, 'text')]
#             }
#             for message in chat.history
#         ]
#         with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
#             json.dump(history, f, ensure_ascii=False, indent=2)
#         print(f"[INFO] Chat history saved to {CHAT_HISTORY_FILE}")
#     except Exception as e:
#         print(f"[ERROR] Failed to save chat history: {e}")

# def load_chat_history():
#     try:
#         if os.path.exists(CHAT_HISTORY_FILE) and os.path.getsize(CHAT_HISTORY_FILE) > 0:
#             with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
#                 history = json.load(f)
#             print(f"[INFO] Chat history loaded from {CHAT_HISTORY_FILE}")
#             # Konversi history ke format yang diterima model
#             formatted_history = [
#                 {"role": msg["role"], "parts": [{"text": part} for part in msg["parts"]]}
#                 for msg in history
#             ]
#             return model.start_chat(history=formatted_history)
#     except Exception as e:
#         print(f"[ERROR] Failed to load chat history: {e}")
    
#     print("[INFO] Starting new chat session")
#     return model.start_chat()
# # Initialize chat instance
# try:
#     chat = load_chat_history()
# except Exception as e:
#     print(f"[ERROR] Failed to initialize chat: {e}")
#     chat = None

# def generate_response(prompt: str) -> str:
#     if not model or not chat:
#         return "[ERROR] Gemini API not properly initialized"
        
#     try:
#         print(f"[INFO] Processing user prompt: '{prompt}'")
        
#         if not chat.history:
#             print("[INFO] Sending system instruction to new chat")
#             chat.send_message(system_instruction)
        
#         response = chat.send_message(prompt)
#         response_text = g2p(response.text.strip())
        
#         print(f"[INFO] Gemini response: '{response_text}'")
#         save_chat_history(chat)
        
#         return response_text
#     except Exception as e:
#         print(f"[ERROR] Failed to generate response: {e}")
#         return f"[ERROR] {e}"

import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
from g2p_id import G2P

g2p = G2P()

load_dotenv()

# Gunakan os.getenv("NAMA_ENV_VARIABLE") untuk mengambil API Key dari file .env.
# Pastikan di file .env terdapat baris: GEMINI_API_KEY=your_api_key
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("[ERROR] GEMINI_API_KEY not found in environment variables")

MODEL = "gemini-2.0-flash"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# Pastikan folder untuk history ada
os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)

# Prompt sistem yang digunakan untuk membimbing gaya respons LLM
system_instruction = """ 
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian. 
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice. 
 
Your answers must: 
- Be written in polite and easily understandable Indonesian. 
- Be short and to the point (maximum 2–3 sentences). 
- Avoid repeating the user's question; respond directly with the answer. 
 
Example tone: 
User: Cuaca hari ini gimana? 
Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat. 
 
User: Kamu tahu siapa presiden Indonesia? 
Assistant: Presiden Indonesia saat ini adalah Joko Widodo. 
 
If you're unsure about an answer, be honest and say that you don't know. 
""" 

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name=MODEL)
    print(f"[INFO] Successfully configured Gemini API with model: {MODEL}")
except Exception as e:
    print(f"[ERROR] Failed to configure Gemini API: {e}")
    model = None

def save_chat_history(chat):
    try:
        # Ekstrak hanya field yang relevan dari history
        history = [
            {
                "role": message.role,
                "parts": [part.text for part in message.parts if hasattr(part, 'text')]
            }
            for message in chat.history
        ]
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Chat history saved to {CHAT_HISTORY_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save chat history: {e}")

def load_chat_history():
    try:
        if os.path.exists(CHAT_HISTORY_FILE) and os.path.getsize(CHAT_HISTORY_FILE) > 0:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            print(f"[INFO] Chat history loaded from {CHAT_HISTORY_FILE}")
            # Konversi history ke format yang diterima model
            formatted_history = [
                {"role": msg["role"], "parts": [{"text": part} for part in msg["parts"]]}
                for msg in history
            ]
            return model.start_chat(history=formatted_history)
    except Exception as e:
        print(f"[ERROR] Failed to load chat history: {e}")
    
    print("[INFO] Starting new chat session")
    return model.start_chat()

# Initialize chat instance
try:
    chat = load_chat_history()
except Exception as e:
    print(f"[ERROR] Failed to initialize chat: {e}")
    chat = None

def generate_response(prompt: str) -> dict:
    if not model or not chat:
        return {"error": "[ERROR] Gemini API not properly initialized"}
        
    try:
        print(f"[INFO] Processing user prompt: '{prompt}'")
        
        if not chat.history:
            print("[INFO] Sending system instruction to new chat")
            chat.send_message(system_instruction)
        
        response = chat.send_message(prompt)
        raw_response = response.text.strip()
        g2p_response = g2p(raw_response)
        
        print(f"[INFO] Gemini raw response: '{raw_response}'")
        print(f"[INFO] Gemini G2P response: '{g2p_response}'")
        save_chat_history(chat)
        
        return {"raw_response": raw_response, "g2p_response": g2p_response}
    except Exception as e:
        print(f"[ERROR] Failed to generate response: {e}")
        return {"error": f"[ERROR] {e}"}