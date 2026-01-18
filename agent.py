import sounddevice as sd
import numpy as np
import os
import ollama
from faster_whisper import WhisperModel
import subprocess
import time

LLM_MODEL = "my-phi3"
WHISPER_SIZE = "tiny.en"   
SAMPLE_RATE = 16000

print(f"⏳ Loading Ears (Whisper) and Brain ({LLM_MODEL})...")
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
print(f"✅ Agent is ready! Speak clearly. Say 'Exit' to stop.")

def record_audio_fixed(duration=5):
    """Listens for fixed seconds."""
    print("\n🔴 Listening...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait() 
    print("⚫ Thinking...")
    return recording

def transcribe(audio_data):
    """Converts Audio -> Text"""
    audio_float = audio_data.flatten().astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(audio_float, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def speak(text):
    """Converts Text -> Audio (Blocking)"""
    subprocess.run(["say", "-r", "190", text])

def main():
    while True:
        try:
            audio_data = record_audio_fixed(duration=5)
            
            user_text = transcribe(audio_data)
            
            if len(user_text) < 2:
                continue
                
            print(f"👤 You: {user_text}")
            
            if "exit" in user_text.lower() or "bye" in user_text.lower():
                speak("Goodbye!")
                break

            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': "You are a helpful voice assistant. Answer in 1 short sentence."},
                    {'role': 'user', 'content': user_text}
                ],
                stream=True
            )

            print("🤖 Agent: ", end="", flush=True)
            
            buffer = ""
            for chunk in stream:
                part = chunk['message']['content']
                buffer += part
                print(part, end="", flush=True)
                
                if part in [".", "?", "!"]:
                    speak(buffer)
                    buffer = ""
            
            if buffer:
                speak(buffer)
            
            print("\n")
            
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n🛑 Stopped manually.")
            break

if __name__ == "__main__":
    main()