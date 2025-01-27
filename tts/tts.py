from smallest import Smallest
from pydub import AudioSegment
from pydub.playback import play
import os
import io

tts_api_key = ""

def generate_and_play_speech(text, voice_id='aarav'):
    # Set up the client with your API key
    client = Smallest(api_key=tts_api_key)

    # Synthesize the speech and save it as a WAV file
    audio_bytes = client.synthesize(
        text,
        voice_id=voice_id,  # Use the provided voice ID
        speed=1.2,          # Optional: set the speed of speech
        sample_rate=24000,  # Optional: set the sample rate
        add_wav_header=True # Optional: ensure WAV header is included
    )

    # Convert bytes to AudioSegment and play the audio
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    play(audio)

if __name__ == "__main__":
    text_to_synthesize = "Hey atri how can i assist you my dear"
    generate_and_play_speech(text_to_synthesize, voice_id="aarav")  # You can change voice_id
