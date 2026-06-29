from pathlib import Path
import os
import subprocess

from ollama import chat
from gtts import gTTS


# 1. 현재 이 코드가 적힌 파이썬 파일의 위치를 가져옵니다.
CURRENT_DIR = Path(__file__).parent

# 2. 현재 폴더 기준 한 단계 위(..)의 speaker 폴더를 지정하고 resolve()로 경로를 정리합니다.
BASE_DIR = (CURRENT_DIR / ".." / "speaker").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

MIC_SOURCE = "bluez_source.B8_D5_0B_9B_F3_33.handsfree_head_unit"
BLUETOOTH_SINK = "bluez_sink.B8_D5_0B_9B_F3_33.handsfree_head_unit"

# 3. 마찬가지로 whisper.cpp 폴더도 동적으로 매칭합니다.
WHISPER_DIR = (CURRENT_DIR / ".." / "whisper.cpp").resolve()
WHISPER_BIN = WHISPER_DIR / "build/bin/whisper-cli"
WHISPER_MODEL = WHISPER_DIR / "models/ggml-small.bin"

INPUT_WAV = BASE_DIR / "user_input.wav"
STT_OUTPUT_BASE = BASE_DIR / "stt_result"
STT_TXT = BASE_DIR / "stt_result.txt"
OUTPUT_MP3 = BASE_DIR / "ollama_response.mp3"
WARMUP_MP3 = BASE_DIR / "warmup_silence.mp3"

BT_MAC = "B8_D5_0B_9B_F3_33"

def get_current_bt_sink():
    '''현재 켜져있는 블루투스 스피커의 정확한 이름을 알아서 찾아옵니다.'''
    try:
        # 리눅스 오디오 장치 목록을 텍스트로 뽑아옵니다.
        result = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            # 목록 중에 우리 스피커의 MAC 주소가 포함된 줄이 있다면?
            if BT_MAC in line:
                # 탭(Tab)으로 구분된 텍스트 중 2번째(인덱스 1)가 진짜 Sink 이름입니다.
                return line.split()[1] 
    except Exception as e:
        print(f"[경고] 오디오 장치 목록 검색 실패: {e}")
    
    return None # 못 찾으면 None 반환

def play_mp3_rev2(mp3_path, volume="80%"):
    '''음성 출력 함수 (자동 탐지 및 에러 방어 적용)'''
    mp3_path = Path(mp3_path)

    if not mp3_path.exists():
        print(f"[경고] 재생할 MP3 파일이 없습니다: {mp3_path}")
        return

    # 1. 스피커가 잘 연결되어 있는지 먼저 찾습니다.
    current_sink = get_current_bt_sink()
    if current_sink is None:
        print(f"🚨 [오디오 에러] 스피커({BT_MAC})가 연결되어 있지 않습니다. 블루투스 연결을 확인하세요!")
        return # 스피커가 없으면 메인 코드가 안 죽도록 그냥 넘어갑니다.

    # 2. 웜업 파일이 없으면 만듭니다.
    if not WARMUP_MP3.exists():
        silence_tts = gTTS(text=".", lang="ko")
        silence_tts.save(str(WARMUP_MP3))

    # 3. 찾아낸 이름(current_sink)으로 세팅하고 재생합니다.
    try:
        subprocess.run(["pactl", "set-default-sink", current_sink], check=True)
        subprocess.run(["pactl", "set-sink-volume", current_sink, volume], check=True)
        subprocess.run(["pactl", "set-sink-mute", current_sink, "0"], check=True)

        env = os.environ.copy()
        env["PULSE_SINK"] = current_sink

        # 스피커 깨우기 (에러 무시)
        subprocess.run(["mpg123", "-q", "-a", "pulse", str(WARMUP_MP3)], env=env, check=False)
        # 진짜 파일 재생
        subprocess.run(["mpg123", "-q", "-a", "pulse", str(mp3_path)], env=env, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"🚨 [오디오 시스템 에러] 재생 중 문제가 발생했습니다: {e}")


def record_audio(seconds=5):
    '''예약어를 녹음하는 함수'''
    if INPUT_WAV.exists():
        INPUT_WAV.unlink()

    print(f"{seconds}초 동안 말하세요...")

    subprocess.run(
        [
            "timeout",
            str(seconds),
            "parecord",
            "--device", MIC_SOURCE,
            "--rate", "16000",
            "--channels", "1",
            "--format", "s16le",
            str(INPUT_WAV),
        ],
        check=False,
    )

    if not INPUT_WAV.exists():
        raise FileNotFoundError("녹음 파일이 생성되지 않았습니다.")

    print("녹음 저장:", INPUT_WAV)


def transcribe_audio():
    '''STT -> TXT'''
    if STT_TXT.exists():
        STT_TXT.unlink()

    subprocess.run(
        [
            str(WHISPER_BIN),
            "-m", str(WHISPER_MODEL),
            "-f", str(INPUT_WAV),
            "-l", "ko",
            "-otxt",
            "-of", str(STT_OUTPUT_BASE),
        ],
        cwd=str(WHISPER_DIR),
        check=True,
    )

    if not STT_TXT.exists():
        raise FileNotFoundError("Whisper 결과 텍스트 파일이 생성되지 않았습니다.")

    text = STT_TXT.read_text(encoding="utf-8").strip()
    print("Whisper text:", text)
    return text


def get_ollama_text(user_text, user_data_txt):
    '''LLM 소통 (시스템 프롬프트와 사용자 데이터 분리)'''
    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": (
                    "You are an AI assistant for driver drowsiness prevention. "
                    "Use the provided system status data to answer user questions naturally. "
                    "When answering, explicitly mention relevant metrics (e.g., eye closure time, closure ratio) "
                    "and the specific cause of the alert if available. "
                    "If the question is unrelated, reply with 'I cannot answer that.' "
                    "Keep your response concise, within two sentences.\n\n"
                    "Only speak Korean"
                    f"driver sleep data:{user_data_txt}\n\nUser Question: {user_text}"
                ),
            }
        ],
    )
    return response.message.content.strip().replace('"', "")


def make_mp3(text, mp3_path):
    '''mp3 만드는 함수'''
    if mp3_path.exists():
        mp3_path.unlink()

    tts = gTTS(text=text, lang="ko")
    tts.save(str(mp3_path))


def play_mp3(mp3_path, volume="80%"):
    '''음성 출력 함수'''
    mp3_path = Path(mp3_path)

    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    if not WARMUP_MP3.exists():
        silence_tts = gTTS(text=".", lang="ko")
        silence_tts.save(str(WARMUP_MP3))

    subprocess.run(["pactl", "set-default-sink", BLUETOOTH_SINK], check=True)
    subprocess.run(["pactl", "set-sink-volume", BLUETOOTH_SINK, volume], check=True)
    subprocess.run(["pactl", "set-sink-mute", BLUETOOTH_SINK, "0"], check=True)

    env = os.environ.copy()
    env["PULSE_SINK"] = BLUETOOTH_SINK

    subprocess.run(["mpg123", "-q", "-a", "pulse", str(WARMUP_MP3)], env=env, check=False)
    subprocess.run(["mpg123", "-q", "-a", "pulse", str(mp3_path)], env=env, check=True)

def speak_text(text, volume="100%"):
    '''통합함수 text->mp3->sound 출력'''
    print("AI:", text)
    make_mp3(text, OUTPUT_MP3)
    play_mp3(OUTPUT_MP3, volume=volume)

def is_wake_word(text):
    '''예약어 지정 함수'''
    wake_words = ["마이카", "아이카", "아이가", "마카","마이크"]
    return any(word in text for word in wake_words)

if __name__ == "__main__":
    print("음성 대기 모드 시작")
    print("'마이카'이라고 말하면 질문 모드로 들어갑니다.")
    print("종료하려면 Ctrl+C를 누르세요.")

    try:
        while True:
            print("\n==============================")
            print("트리거 대기 중... '마이카'이라고 말하세요.")
            print("==============================")

            record_audio(seconds=5)
            trigger_text = transcribe_audio()

            print("\n트리거 인식 결과:", trigger_text)

            if not is_wake_word(trigger_text):
                print("'안녕' 트리거가 인식되지 않았습니다. 다시 대기합니다.")
                continue

            #speak_text("네 질문하세요.")

            print("\n질문을 6초 동안 말하세요.")
            record_audio(seconds=6)
            user_text = transcribe_audio()

            print("\n==============================")
            print("질문 인식 결과")
            print("==============================")
            print(user_text)
            print("==============================")

            if not user_text or user_text.strip() in ["(끝)", "[BLANK_AUDIO]"]:
                speak_text("질문을 인식하지 못했습니다. 다시 말씀해주세요.")
                continue

            answer = get_ollama_text(user_text)

            print("\n==============================")
            print("Ollama 응답")
            print("==============================")
            print(answer)
            print("==============================")

            speak_text(answer)

    except KeyboardInterrupt:
        print("\n음성 대기 모드를 종료합니다.")


