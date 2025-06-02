import pyttsx3

engine = pyttsx3.init()

def speak(text, voice_id='com.apple.voice.compact.en-US.Samantha', rate=180):
    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

    