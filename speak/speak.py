import speech_recognition as sr

def take_command():
    """Listen to audio from the microphone and return the recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source)
            print("Recognizing...")
            query = recognizer.recognize_google(audio, language="en-in")
            print(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""

if __name__ == "__main__":
    while True:
        spoken_text = take_command()
        if spoken_text.lower() in ["quit", "exit"]:
            print("Exiting...")
            break