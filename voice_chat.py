
import os
import sounddevice as sd
import pyttsx3
from scipy.io.wavfile import write
from groq import Groq

from dotenv import load_dotenv

load_dotenv()

# Connect to Groq API
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# Initialize voice engine
speaker = pyttsx3.init()
speaker.setProperty("rate", 170)


# Conversation history
messages = [
    {
        "role": "system",
        "content": "You are a friendly AI voice assistant. Give short and simple answers."
    }
]


# STEP 1: Record voice
def listen():

    input("\nPress ENTER to start speaking...")

    print("Listening... Speak now!")

    duration = 5
    sample_rate = 16000

    # Record voice
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    print("Recording completed!")

    # Check microphone sound
    volume = abs(audio.astype("int32")).max()

    print("Sound level:", volume)

    if volume < 200:
        print("No voice detected. Try again.")
        return None

    # Save audio file
    write("audio.wav", sample_rate, audio)

    return "audio.wav"


# STEP 2: Convert voice into text using Whisper
def speech_to_text(filename):

    print("Converting speech to text...")

    with open(filename, "rb") as file:

        result = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo"
        )

    text = result.text

    print("You:", text)

    return text


# STEP 3: Generate AI response
def get_response(user_text):

    messages.append({
        "role": "user",
        "content": user_text
    })

    print("Thinking...")

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )

    answer = response.choices[0].message.content

    messages.append({
        "role": "assistant",
        "content": answer
    })

    print("Assistant:", answer)

    return answer


# STEP 4: Convert text into voice
def speak(text):

    print("Speaking...")

    speaker.say(text)
    speaker.runAndWait()


# STEP 5: Main program
def main():

    print("\n================================")
    print("       AI VOICE ASSISTANT")
    print("================================")

    print("Press ENTER to speak.")
    print("Say 'exit' to stop.")

    while True:

        try:

            # Record voice
            filename = listen()

            if filename is None:
                continue

            # Convert voice into text
            user_text = speech_to_text(filename)

            if not user_text.strip():

                print("Could not understand. Try again.")
                continue

            # Exit condition
            if "exit" in user_text.lower():

                speak("Goodbye! Have a nice day.")
                break

            # Generate AI response
            answer = get_response(user_text)

            # Speak AI response
            speak(answer)

        except KeyboardInterrupt:

            print("\nAssistant stopped.")
            break

        except Exception as e:

            print("Error:", e)


# Run program
if __name__ == "__main__":
    main()
