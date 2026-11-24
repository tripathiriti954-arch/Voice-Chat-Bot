
import os
import re
import sys
import time
import tempfile
import msvcrt

from concurrent.futures import ThreadPoolExecutor

import requests
import pygame
import speech_recognition as sr
from gtts import gTTS

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ================= SETTINGS =================

API_KEY = os.getenv("GROQ_API_KEY")

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

MIC_INDEX = 1

ENERGY_THRESHOLD = 200
AUTO_CALIBRATE = True

# =============================================


if not API_KEY:
    print("ERROR: GROQ_API_KEY not found!")
    print('PowerShell: $env:GROQ_API_KEY="YOUR_NEW_API_KEY"')
    sys.exit()


pygame.mixer.init()

history = [
    {
        "role": "system",
        "content": (
            "You are a friendly voice assistant. "
            "Reply naturally in the user's language. "
            "Keep answers short and conversational. "
            "Avoid markdown and unnecessary symbols."
        )
    }
]


# ================= KEYBOARD =================

def flush_keys():
    while msvcrt.kbhit():
        msvcrt.getch()


def wait_for_key():
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()

            if key == b'\r':
                return "ENTER"

            if key == b'\x1b':
                return "ESC"

        time.sleep(0.05)


# ================= LANGUAGE =================

def is_hindi(text):
    if not text:
        return False

    hindi_chars = len(
        re.findall(r'[\u0900-\u097F]', text)
    )

    return hindi_chars > 0


# ================= SPEECH RECOGNITION =================

def recognize_one(recognizer, audio, language):

    try:
        result = recognizer.recognize_google(
            audio,
            language=language
        )

        return result, 0.5

    except sr.UnknownValueError:
        return None, 0.0

    except sr.RequestError as e:
        print(f"Google recognition error ({language}): {e}")
        return None, 0.0

    except Exception as e:
        print("Recognition error:", e)
        return None, 0.0


def transcribe(audio, recognizer):

    print("Understanding your voice...")

    with ThreadPoolExecutor(max_workers=2) as executor:

        future_en = executor.submit(
            recognize_one,
            recognizer,
            audio,
            "en-IN"
        )

        future_hi = executor.submit(
            recognize_one,
            recognizer,
            audio,
            "hi-IN"
        )

        result_en = future_en.result()
        result_hi = future_hi.result()

    text_en = result_en[0]
    text_hi = result_hi[0]

    if text_hi and is_hindi(text_hi):
        return text_hi

    if text_en:
        return text_en

    if text_hi:
        return text_hi

    return None


# ================= GROQ AI =================

def ask_ai(user_text):

    history.append({
        "role": "user",
        "content": user_text
    })

    language_instruction = (
        "Reply in Hindi or Hinglish."
        if is_hindi(user_text)
        else "Reply in English."
    )

    messages = history + [
        {
            "role": "system",
            "content": language_instruction
        }
    ]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7
    }

    try:

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            print("Groq API Error:", response.text)
            return "Sorry, AI service mein problem aa rahi hai."

        answer = response.json()["choices"][0]["message"]["content"]

        history.append({
            "role": "assistant",
            "content": answer
        })

        return answer

    except requests.exceptions.Timeout:
        return "Server response mein time lag raha hai."

    except requests.exceptions.RequestException as e:
        print("Connection error:", e)
        return "Internet connection check karo."


# ================= TEXT TO SPEECH =================

def speak(text, hindi=False):

    print("\nAssistant:", text)

    filename = None

    try:

        language = "hi" if hindi else "en"

        with tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        ) as f:
            filename = f.name

        tts = gTTS(
            text=text,
            lang=language,
            slow=False
        )

        tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():

            if msvcrt.kbhit():
                key = msvcrt.getch()

                if key == b'\x1b':
                    pygame.mixer.music.stop()
                    break

            pygame.time.Clock().tick(20)

    except Exception as e:
        print("TTS Error:", e)

    finally:

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

        if filename and os.path.exists(filename):

            try:
                os.remove(filename)
            except PermissionError:
                time.sleep(0.5)
                try:
                    os.remove(filename)
                except OSError:
                    pass


# ================= MAIN PROGRAM =================

def main():

    recognizer = sr.Recognizer()

    recognizer.energy_threshold = ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = AUTO_CALIBRATE
    recognizer.pause_threshold = 1.0
    recognizer.non_speaking_duration = 0.5
    recognizer.phrase_threshold = 0.2

    try:

        with sr.Microphone(
            device_index=MIC_INDEX
        ) as source:

            print("\nCalibrating microphone...")
            recognizer.adjust_for_ambient_noise(
                source,
                duration=2
            )

            print("Microphone ready!")

            while True:

                print("\n" + "=" * 40)
                print("Press ENTER to speak")
                print("Press ESC to exit")
                print("=" * 40)

                flush_keys()
                key = wait_for_key()

                if key == "ESC":
                    break

                print("\nListening... Speak now!")

                try:

                    audio = recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=20
                    )

                    print(
                        f"Received {len(audio.get_raw_data()) // 1024} KB audio"
                    )

                    user_text = transcribe(
                        audio,
                        recognizer
                    )

                    if not user_text:

                        print(
                            "Could not understand audio. "
                            "Please speak clearly and try again."
                        )

                        continue

                    print("\nYou:", user_text)

                    if user_text.lower().strip() in [
                        "exit",
                        "quit",
                        "stop",
                        "bye",
                        "बंद करो"
                    ]:
                        print("Goodbye!")
                        break

                    hindi = is_hindi(user_text)

                    print("\nThinking...")

                    answer = ask_ai(user_text)

                    speak(answer, hindi)

                except sr.WaitTimeoutError:
                    print("No speech detected. Try again.")

                except sr.UnknownValueError:
                    print("Could not understand audio. Try again.")

                except sr.RequestError as e:
                    print("Speech recognition internet error:", e)

                except Exception as e:
                    print("Error:", e)

    except OSError as e:
        print("Microphone error:", e)
        print("Check MIC_INDEX and microphone permissions.")

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
