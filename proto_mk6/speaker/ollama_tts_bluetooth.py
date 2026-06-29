from pathlib import Path
import os
import subprocess

from ollama import chat
from gtts import gTTS


BLUETOOTH_SINK = "bluez_sink.B8_D5_0B_9B_F3_33.a2dp_sink"
OUTPUT_MP3 = Path("/home/aidl/work/speaker/ollama_response.mp3")


def get_ollama_text():
    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": (
                    "반드시 한국어로만 답하세요. "
                    "운전자에게 출력할 경고 문장을 두줄 작성하세요. "
                    "설명하지 말고 문장만 출력하세요. "
                    "따옴표 없이 출력하세요. "
                    "예: 전방을 주시하세요."
                ),
            }
        ],
    )

    return response.message.content.strip().replace('"', "")


def make_mp3(text, mp3_path):
    tts = gTTS(text=text, lang="ko")
    tts.save(str(mp3_path))


def play_mp3(mp3_path, volume="50%"):
    mp3_path = Path(mp3_path)
    warmup_mp3 = Path("/home/aidl/work/speaker/warmup_silence.mp3")

    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    # 무음 mp3가 없으면 한 번만 생성
    if not warmup_mp3.exists():
        silence_tts = gTTS(text=".", lang="ko")
        silence_tts.save(str(warmup_mp3))

    subprocess.run(["pactl", "set-default-sink", BLUETOOTH_SINK], check=True)
    subprocess.run(["pactl", "set-sink-volume", BLUETOOTH_SINK, volume], check=True)
    subprocess.run(["pactl", "set-sink-mute", BLUETOOTH_SINK, "0"], check=True)

    env = os.environ.copy()
    env["PULSE_SINK"] = BLUETOOTH_SINK

    # 스피커 깨우기용 짧은 mp3
    subprocess.run(
        ["mpg123", "-q", "-a", "pulse", str(warmup_mp3)],
        env=env,
        check=False,
    )

    # 실제 경고 mp3 재생
    subprocess.run(
        ["mpg123", "-q", "-a", "pulse", str(mp3_path)],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    text = get_ollama_text()
    print("Ollama response:", text)

    make_mp3(text, OUTPUT_MP3)
    print("MP3 saved:", OUTPUT_MP3)

    play_mp3(OUTPUT_MP3)