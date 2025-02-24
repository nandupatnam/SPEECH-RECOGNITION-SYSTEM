import torch
import torchaudio
import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Choose GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load optimized Wav2Vec2 model
model_name = "facebook/wav2vec2-base-960h"  # Smaller model for faster processing
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

def transcribe_audio_wav2vec(audio_path):
    """Transcribe audio using Wav2Vec2 (Optimized for speed)."""
    waveform, rate = torchaudio.load(audio_path)  # Faster than librosa
    waveform = torchaudio.functional.resample(waveform, orig_freq=rate, new_freq=16000)  # Convert to 16kHz

    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)  # Move to GPU if available

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

def transcribe_audio_speechrecognition():
    """Live speech recognition using SpeechRecognition (Google API)."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Could not request results, check your internet connection."

if __name__ == "__main__":
    choice = input("Choose method: 1 for Live Speech, 2 for Wav2Vec2 File Transcription: ")

    if choice == "1":
        print("Recognized Text:", transcribe_audio_speechrecognition())
    elif choice == "2":
        audio_file = r"D:\INTERNSHIP CODTECH\TASK 2\gen_0.wav"  # Update with your file path
        print("Recognized Text (Wav2Vec2):", transcribe_audio_wav2vec(audio_file))
    else:
        print("Invalid choice. Please enter 1 or 2.")
