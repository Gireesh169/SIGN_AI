import pyttsx3
import threading

def speak_text(text):
    """
    Decoupled Text-To-Speech engine using pyttsx3 (Native OS Engine).
    Runs asynchronously in a background thread to prevent blocking the ML inference loop.
    """
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

if __name__ == '__main__':
    speak_text("Testing sign language text to speech translation system")
