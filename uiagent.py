import gradio as gr
import ollama
from faster_whisper import WhisperModel
import subprocess
import os

LLM_MODEL = "my-phi3"       
WHISPER_SIZE = "tiny.en"    
SYSTEM_PROMPT = "You are a helpful voice assistant. Answer in 1 short sentence."

print(f"⏳ Loading Whisper ({WHISPER_SIZE})...")
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
print("✅ Whisper Loaded!")

def transcribe(audio_path):
    """Converts Audio File -> Text"""
    if audio_path is None:
        return ""
    
    segments, _ = whisper.transcribe(audio_path, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def generate_response(user_text):
    """Sends text to Ollama (my-phi3)"""
    if not user_text:
        return ""
    
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_text}
        ]
    )
    return response['message']['content']

def text_to_speech_mac(text):
    """
    Generates audio using Mac's native 'say' command, 
    but saves it to a WAV file so Gradio can play it.
    """
    output_file = "response.wav"
    
    
    subprocess.run([
        "say", "-r", "190", 
        "-o", output_file, 
        "--data-format=LEI16@22050", 
        text
    ])
    return output_file

def voice_chat(audio_recording):
    """The Main Pipeline: Audio -> Text -> Brain -> Audio"""
    
    user_text = transcribe(audio_recording)
    if not user_text:
        return "No audio detected", None
        
    print(f"👤 User: {user_text}")

    bot_response = generate_response(user_text)
    print(f"🤖 Bot: {bot_response}")

    audio_response_path = text_to_speech_mac(bot_response)

    return bot_response, audio_response_path

with gr.Blocks(title="Pype AI Voice Agent") as demo:
    gr.Markdown(f"## 🤖 Local Voice Agent ({LLM_MODEL})")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak Here")
            submit_btn = gr.Button("Talk", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(label="Agent Response")
            audio_output = gr.Audio(label="Audio Reply", autoplay=True) # autoplay=True makes it speak instantly

    submit_btn.click(
        fn=voice_chat,
        inputs=audio_input,
        outputs=[text_output, audio_output]
    )

    audio_input.stop_recording(
        fn=voice_chat,
        inputs=audio_input,
        outputs=[text_output, audio_output]
    )

demo.launch()