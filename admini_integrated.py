#!/usr/bin/env python3
"""
Admini Assistant — integrated (no live_module, no wake_module)
Single-file: arecord -> ffmpeg streaming -> wakeword inference -> command capture (forward) -> transcription -> TTS
LED behavior:
 - blue_run: running blue while waiting for wakeword
 - blue_breath: breathing blue while recording command
 - yellow: processing /TTS output
 - green/red/off available for notifications/errors
"""
import os
import sys
import time
import signal
import threading
import warnings
import subprocess
import queue
import json
import io
import re
import shutil
from collections import deque
warnings.filterwarnings("ignore")

# External libs (ensure installed)
from vtt import translate_audio_to_english, transcribe_and_detect_language, translate_audio_to_englishopen
import speech_recognition as sr
import numpy as np
import RPi.GPIO as GPIO
from pydub import AudioSegment
from pydub.playback import play
import requests
from chat_api import chat
# Google Cloud TTS
from google.cloud import texttospeech
from clean import preprocess_wav
from langdetect import detect, LangDetectException
import pika
import soundfile as sf

# ML / audio libs for wake inference
import torch
import torch.nn as nn
import torchaudio

# DotStar LED (optional)
try:
    import adafruit_dotstar
    import board
    DOTSTAR_AVAILABLE = True
except Exception:
    DOTSTAR_AVAILABLE = False

# -----------------------------
# CONFIG (tweak as needed)
# -----------------------------
WAKE_WORD = "admini"
_stop = False
model_paused = False
chat_mode_active = False
exit_chat_mode = False
current_language = "en"

notification_pending = False
notification_lock = threading.Lock()

skip_current_notification = False
skip_all_notifications = False
skip_lock = threading.Lock()

# RabbitMQ (keep as you had)
RABBITMQ_HOST = "ec2-13-203-69-0.ap-south-1.compute.amazonaws.com"
RABBITMQ_PORT = 5672
RABBITMQ_USER = "admini"
RABBITMQ_PASS = "Anton@123"
RABBITMQ_QUEUE = "notification_queue"

# TTS credentials (path)
JSON_CREDENTIALS_PATH = "fast-planet-479910-k9-88fd07aefb58.json"

# Audio capture / wake inference config
ARECORD_DEVICE = os.environ.get("ARECORD_DEVICE", "hw:3,0")  # set as needed
CAPTURE_SR = 48000     # arecord capture in 48k
MODEL_SR = 16000       # model uses 16k
CLIP_DURATION = 2.0
TARGET_LEN = int(MODEL_SR * CLIP_DURATION)  # 32000
CHUNK_SECONDS = 2
CHUNK_FRAMES = MODEL_SR * CHUNK_SECONDS
CHUNK_BYTES = CHUNK_FRAMES * 2  # s16le -> 2 bytes/sample (int16)
# Path to model
MODEL_PATH = os.environ.get("MODEL_PATH", "wakeword_lstm_2class.pt")

# Mel params
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160

# Wake thresholds
THRESHOLD = 0.90
VERY_HIGH_THRESHOLD = 0.95
REQUIRED_HITS = 1
HITS_WINDOW = 2
COOLDOWN = 10.0

# Capture exactly 10 seconds after wake
AFTER_WAKE_RECORD_SECONDS = 10  # 10 seconds after wake

# GPIO pins
GPIO.setwarnings(False)
SERVO_PIN = 13
SWITCH_PIN = 27
NOTIFICATION_BUTTON_PIN = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(NOTIFICATION_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)
# servo state tracking
servo_lock = threading.Lock()
servo_raised = False

# DotStar setup
NUM_LEDS = 12
LED_CLOCK_PIN_NAME = "D16"
LED_DATA_PIN_NAME = "D23"

if DOTSTAR_AVAILABLE:
    try:
        LED_CLOCK_PIN = getattr(board, LED_CLOCK_PIN_NAME)
        LED_DATA_PIN = getattr(board, LED_DATA_PIN_NAME)
        dots = adafruit_dotstar.DotStar(LED_CLOCK_PIN, LED_DATA_PIN, NUM_LEDS, brightness=0.5, auto_write=True)
    except Exception:
        class _DummyDots:
            def __setitem__(self, idx, val): pass
        dots = _DummyDots()
else:
    class _DummyDots:
        def __setitem__(self, idx, val): pass
    dots = _DummyDots()

_led_running = True
_stop_event = threading.Event()

def led_thread_fn():
    """Single LED thread that reacts to _led_mode"""
    global _led_mode, _led_running, dots
    i = 0.0
    while not _stop_event.is_set():
        mode = _led_mode
        if mode == "blue_run":
            # simple rotating blue segments
            for j in range(NUM_LEDS):
                try:
                    if j == int(i) % NUM_LEDS:
                        dots[j] = (0, 0, 200)
                    else:
                        dots[j] = (0, 0, 30)
                except Exception:
                    pass
            i += 1
            time.sleep(0.08)
        elif mode == "blue_breath":
            level = (np.sin(i / 10.0) + 1.0) / 2.0
            val = int(30 + 200 * level)
            for j in range(NUM_LEDS):
                try:
                    dots[j] = (0, 0, val)
                except Exception:
                    pass
            i += 1
            time.sleep(0.05)
        elif mode == "yellow":
            for j in range(NUM_LEDS):
                try:
                    dots[j] = (200, 160, 0)
                except Exception:
                    pass
            time.sleep(0.15)
        elif mode == "green":
            for j in range(NUM_LEDS):
                try:
                    dots[j] = (0, 200, 0)
                except Exception:
                    pass
            time.sleep(0.15)
        elif mode == "red":
            for j in range(NUM_LEDS):
                try:
                    dots[j] = (200, 0, 0)
                except Exception:
                    pass
            time.sleep(0.15)
        else:  # off
            for j in range(NUM_LEDS):
                try:
                    dots[j] = (0, 0, 0)
                except Exception:
                    pass
            time.sleep(0.2)

def set_led_mode(mode:str):
    """Request an LED mode. This function enforces notification priority:
    - If `mode` is 'blue_breath' or 'yellow' (active listening/processing), it always sets that mode.
    - Otherwise, if notifications are pending, the LED will show 'green'.
    - Otherwise the requested mode is used.
    Use this from code to request a visual mode; the function will decide final mode
    based on notification state.
    """
    global _led_mode, notification_pending
    # Modes that should always override notification green
    active_override = ('blue_breath', 'yellow')
    if mode in active_override:
        _led_mode = mode
    else:
        if notification_pending:
            _led_mode = 'green'
        else:
            _led_mode = mode

# convenience: set LED immediately to a mode regardless of notification (force)
def force_led_mode(mode:str):
    global _led_mode
    _led_mode = mode

# TTS init
try:
    tts_client = texttospeech.TextToSpeechClient.from_service_account_file(JSON_CREDENTIALS_PATH)
    print("✅ Google Cloud TTS initialized")
except Exception as e:
    print(f"⚠️ TTS init failed: {e}")
    tts_client = None

def detect_language(text):
    try:
        tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
        if tamil_pattern.search(text):
            return 'ta'
        try:
            detected = detect(text)
            return 'ta' if detected == 'ta' else 'en'
        except:
            return 'en'
    except:
        return 'en'

def speak_text(text, force_language=None, led_color='yellow', enable_skip=False):
    """
    Speak text using Google Cloud TTS with Tamil (ta-IN) or English (en-IN)
    force_language: 'en', 'ta', 'en-IN', 'ta-IN', or None (auto-detect)
    led_color: 'yellow' during playback (default)
    enable_skip: True for notifications (chunked playback), False for chat mode (smooth playback)
    Returns True if completed, False if skipped
    """
    global current_language, skip_current_notification, skip_all_notifications

    # Check if we should skip before starting (only if skip is enabled)
    if enable_skip:
        with skip_lock:
            if skip_current_notification or skip_all_notifications:
                print("⏭️ Skipping TTS (skip flag detected)")
                skip_current_notification = False
                return False

    # Determine language
    if force_language:
        lang = force_language
    else:
        lang = detect_language(text)

    # Update current language for context
    if lang in ['ta', 'ta-roman', 'ta-IN']:
        current_language = 'ta'
    else:
        current_language = 'en'

    print(f"🗣️ Speaking ({len(text)} chars, {'Tamil' if current_language == 'ta' else 'English'}): {text[:80]}...")

    # Check TTS client
    if not tts_client:
        print("⚠️ Google Cloud TTS not available!")
        return False

    try:
        # Set LED to yellow during playback
        set_led_mode('yellow')

        # Use Tamil or English depending on detection
        if current_language == 'ta':
            language_code = "ta-IN"
        else:
            language_code = "en-IN"

        # Prepare text input
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Select FEMALE voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        # Configure audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            volume_gain_db=10.0
        )

        # Generate speech
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with ignore_stderr():
            audio_data = io.BytesIO(response.audio_content)
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")

            # Use chunking for notifications (to enable skip)
            if enable_skip:
                # Play in chunks of 500ms to check for skip
                chunk_length = 1000  # milliseconds
                for i in range(0, len(audio_segment), chunk_length):
                    # Check skip flags before each chunk
                    with skip_lock:
                        if skip_current_notification or skip_all_notifications:
                            print("⏭️ TTS playback interrupted (skip detected)")
                            skip_current_notification = False
                            set_led_mode('blue_run')
                            return False

                    chunk = audio_segment[i:i + chunk_length]
                    play(chunk)
            else:
                # For chat mode: play full audio smoothly (no chunking)
                play(audio_segment)

        # Return to blue_run after speaking
        set_led_mode('blue_run')
        return True

    except Exception as e:
        print(f"⚠️ Google Cloud TTS failed: {e}")
        set_led_mode('blue_run')
        return False

# small helper to suppress noisy stderr from ffmpeg/alsa
import contextlib
@contextlib.contextmanager
def ignore_stderr():
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()

# Servo helpers
def set_servo_angle_smooth(start, end, step=2, delay=0.05):
    angle_range = range(start, end + 1, step) if start < end else range(start, end - 1, -step)
    for angle in angle_range:
        duty = 2.5 + (angle / 18)
        servo.ChangeDutyCycle(duty)
        time.sleep(delay)
    servo.ChangeDutyCycle(0)

def servo_mode():
    set_servo_angle_smooth(0, 180)
    time.sleep(4)
    set_servo_angle_smooth(180, 0)

# New helpers to raise / lower servo and keep it raised
def raise_servo():
    """Move servo to raised position and mark state. Non-blocking caller should use a thread.
    If already raised, this is a no-op."""
    global servo_raised
    with servo_lock:
        if servo_raised:
            return
        try:
            set_servo_angle_smooth(0, 180, step=4, delay=0.05)
            servo_raised = True
            print("[Servo] raised to 180°")
        except Exception as e:
            print(f"[Servo] raise failed: {e}")

def lower_servo():
    """Move servo to lowered (rest) position and mark state. If already lowered, no-op."""
    global servo_raised
    with servo_lock:
        if not servo_raised:
            return
        try:
            set_servo_angle_smooth(180, 0, step=4, delay=0.05)
            servo_raised = False
            print("[Servo] lowered to 0°")
        except Exception as e:
            print(f"[Servo] lower failed: {e}")

# Notification helpers (unchanged logic)
def check_notifications():
    global notification_pending
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials, heartbeat=600, blocked_connection_timeout=300))
        channel = connection.channel()
        queue_state = channel.queue_declare(queue=RABBITMQ_QUEUE, durable=False)
        if queue_state.method.message_count > 0:
            with notification_lock:
                if not notification_pending:
                    notification_pending = True
                    # keep green LED to indicate pending notifications
                    set_led_mode('green')
                    # raise servo and keep it raised while notifications are pending
                    threading.Thread(target=raise_servo, daemon=True).start()
            connection.close()
            return True
        connection.close()
        return False
    except Exception as e:
        print(f"⚠️ Notification check: {e}")
        return False

def fetch_and_consume_all_notifications():
    notifications = []
    try:
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, credentials=credentials))
        channel = connection.channel()
        while True:
            method, properties, body = channel.basic_get(queue=RABBITMQ_QUEUE, auto_ack=False)
            if method:
                data = json.loads(body)
                notifications.append({'text': data.get('notification', ''), 'type': data.get('type', 'general'), 'org': data.get('org_name', '')})
                channel.basic_ack(method.delivery_tag)
            else:
                break
        connection.close()
        return notifications
    except Exception as e:
        print(f"⚠️ Fetch error: {e}")
        return []

def notification_monitor():
    global _stop
    while not _stop:
        with notification_lock:
            should_check = not notification_pending
        if should_check:
            check_notifications()
        # check every second for more responsive behavior
        time.sleep(1)

def notification_button_monitor():
    global _stop, skip_current_notification, skip_all_notifications
    button_pressed = False
    start_time = 0
    while not _stop:
        current_state = GPIO.input(NOTIFICATION_BUTTON_PIN) == GPIO.LOW
        if current_state and not button_pressed:
            button_pressed = True
            start_time = time.time()
        elif not current_state and button_pressed:
            duration = time.time() - start_time
            button_pressed = False
            if duration >= 3.0:
                with skip_lock:
                    skip_all_notifications = True
                    skip_current_notification = True
            elif duration >= 0.3:
                with skip_lock:
                    skip_current_notification = True
        time.sleep(0.05)

def read_notifications_with_skip():
    global notification_pending, skip_current_notification, skip_all_notifications, model_paused
    # ensure servo is raised when we start reading
    try:
        raise_servo()
    except Exception:
        pass

    # Pause inference to reduce playback/recording conflicts
    prev_model_paused = model_paused
    model_paused = True

    notifications = fetch_and_consume_all_notifications()
    if not notifications:
        # nothing to read: lower servo and clear pending state
        lower_servo()
        with notification_lock:
            notification_pending = False
        set_led_mode('off')
        # speak that there are no notifications; allow user to skip
        speak_text("At present there is no notification.", force_language=None, led_color='yellow', enable_skip=True)
        # restore inference pause state
        model_paused = prev_model_paused
        return

    for idx, notif in enumerate(notifications, 1):
        with skip_lock:
            if skip_all_notifications:
                skip_all_notifications = False
                skip_current_notification = False
                break
        # speak the notification (enable skipping mid-playback)
        speak_text(f"{notif['type']} notification. {notif['text']}", force_language=None, led_color='yellow', enable_skip=True)
        with skip_lock:
            if skip_current_notification:
                skip_current_notification = False
                continue
        if idx < len(notifications):
            # small gap between notifications
            time.sleep(0.5)

    # finished reading: lower servo, turn off green, resume wake
    lower_servo()
    set_led_mode('blue_run')
    with notification_lock:
        notification_pending = False
    with skip_lock:
        skip_current_notification = False
        skip_all_notifications = False

    # restore inference pause state
    model_paused = prev_model_paused


def notification_button_press_handler():
    """Monitors the notification button and controls reading/skipping behavior.
    - Short press (while not reading): start reading if notifications present, otherwise speak 'no notifications'.
    - Short press (while reading): skip current notification.
    - Long press (>=3s): skip all notifications, lower servo, clear pending and resume wake.
    """
    global notification_pending, skip_current_notification, skip_all_notifications, model_paused
    button_pressed = False
    start_time = 0
    reading_started = False
    reader_thread = None
    while not _stop:
        state = GPIO.input(NOTIFICATION_BUTTON_PIN) == GPIO.LOW
        now = time.time()
        if state and not button_pressed:
            # button down
            button_pressed = True
            start_time = now
        elif not state and button_pressed:
            # button released
            duration = now - start_time
            button_pressed = False
            if duration >= 3.0:
                # long press: skip all notifications and stop reading
                with skip_lock:
                    skip_all_notifications = True
                    skip_current_notification = True
                # if reading thread is running, it will observe skip_all_notifications and stop
                if reader_thread and reader_thread.is_alive():
                    time.sleep(0.2)
                # lower servo and clear pending
                lower_servo()
                set_led_mode('blue_run')
                with notification_lock:
                    notification_pending = False
                with skip_lock:
                    skip_current_notification = False
                    skip_all_notifications = False
                reading_started = False
                reader_thread = None
            else:
                # short press
                with notification_lock:
                    has = notification_pending
                if has:
                    if not reading_started:
                        # start reading notifications
                        reading_started = True
                        with skip_lock:
                            skip_current_notification = False
                            skip_all_notifications = False
                        reader_thread = threading.Thread(target=read_notifications_with_skip, daemon=True)
                        reader_thread.start()
                    else:
                        # if already reading, skip current notification
                        with skip_lock:
                            skip_current_notification = True
                else:
                    # No notifications present — give audible feedback
                    # Pause inference briefly while we say "no notifications"
                    prev_model_paused = model_paused
                    model_paused = True
                    speak_text("At present there is no notification.", force_language=None, led_color='yellow', enable_skip=True)
                    model_paused = prev_model_paused
        # if notifications were cleared externally, reset reading flag
        with notification_lock:
            if not notification_pending:
                reading_started = False
                reader_thread = None
        time.sleep(0.05)


# -----------------------------
# Streaming / capture internals
# -----------------------------

_BACK_BUFFER_SECONDS = 30  # short history in seconds (keeps memory small)
_back_buffer = deque(maxlen=int(_BACK_BUFFER_SECONDS / CHUNK_SECONDS) + 2)

_record_q = queue.Queue(maxsize=200)  # holds upcoming chunks for forward capturing
_inference_q = queue.Queue(maxsize=16)

def start_audio_stream_process():
    arecord_cmd = ["arecord", "-D", ARECORD_DEVICE, "-f", "S32_LE", "-r", str(CAPTURE_SR), "-c", "2", "-q"]
    ffmpeg_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s32le", "-ar", str(CAPTURE_SR), "-ac", "2", "-i", "-", "-ar", str(MODEL_SR), "-ac", "1", "-f", "s16le", "-"]
    arecord = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=arecord.stdout, stdout=subprocess.PIPE)
    arecord.stdout.close()
    return arecord, ffmpeg

def audio_reader_worker(ffmpeg_proc):
    print(f"[AudioReader] reading {CHUNK_BYTES} byte chunks ({CHUNK_SECONDS}s @ {MODEL_SR}Hz)")
    try:
        while not _stop_event.is_set():
            data = ffmpeg_proc.stdout.read(CHUNK_BYTES)
            if not data or len(data) == 0:
                time.sleep(0.01)
                continue
            if len(data) < CHUNK_BYTES:
                data = data.ljust(CHUNK_BYTES, b'\x00')
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # mono frames at MODEL_SR
            try:
                _inference_q.put_nowait(arr)
            except queue.Full:
                pass
            try:
                _back_buffer.append(arr.copy())
            except Exception:
                pass
            try:
                _record_q.put(arr.copy(), timeout=0.2)
            except queue.Full:
                try:
                    _ = _record_q.get_nowait()
                    _record_q.put_nowait(arr.copy())
                except Exception:
                    pass
    except Exception as e:
        print("[AudioReader] Exception:", e)
    finally:
        print("[AudioReader] exiting")

# -----------------------------
# Capture helpers
# -----------------------------
def capture_forward(duration_seconds, timeout_per_chunk=10.0):
    try:
        needed_frames = int(MODEL_SR * float(duration_seconds))
        frames = []
        frames_collected = 0
        start_t = time.time()
        while frames_collected < needed_frames and not _stop_event.is_set():
            try:
                chunk = _record_q.get(timeout=1.0)
                frames.append(chunk)
                frames_collected += len(chunk)
            except queue.Empty:
                if time.time() - start_t > timeout_per_chunk:
                    break
                continue
        if frames_collected == 0:
            print("[capture_forward] no frames collected")
            return None
        data = np.concatenate(frames, axis=0)
        if len(data) > needed_frames:
            data = data[:needed_frames]
        return data.astype(np.float32)
    except Exception as e:
        print(f"⚠️ capture_forward error: {e}")
        return None

def capture_forward_from_now(duration_seconds, timeout_per_chunk=10.0):
    """Capture audio starting from the next new chunk arriving in _record_q.
    This ensures we do not return already-buffered audio that precedes the wake event.
    It drains any pre-existing queued chunks, then waits for the first new chunk and
    records for the requested *wall-clock* duration_seconds (so it truly records
    for ~duration_seconds seconds after the wake event).
    Returns a float32 numpy array or None on timeout/error.
    """
    try:
        needed_frames = int(MODEL_SR * float(duration_seconds))
        # Drain any old buffered chunks so we only capture fresh audio
        while True:
            try:
                _ = _record_q.get_nowait()
            except queue.Empty:
                break

        # wait for the first *new* chunk (timeout to avoid hanging forever)
        try:
            first = _record_q.get(timeout=timeout_per_chunk)
        except queue.Empty:
            print("[capture_forward_from_now] timeout waiting for first new chunk")
            return None

        frames = [first]
        frames_collected = len(first)
        # Start time-based recording to ensure we capture the requested wall-clock duration
        start_t = time.time()
        while (time.time() - start_t) < float(duration_seconds) and not _stop_event.is_set():
            try:
                chunk = _record_q.get(timeout=1.0)
                frames.append(chunk)
                frames_collected += len(chunk)
            except queue.Empty:
                # no chunk this second, continue waiting until duration elapses
                continue

        if frames_collected == 0:
            print("[capture_forward_from_now] no frames collected")
            return None

        data = np.concatenate(frames, axis=0)
        # trim to exact needed frames if longer
        if len(data) > needed_frames:
            data = data[:needed_frames]
        # if shorter, it's still valid (partial capture)
        return data.astype(np.float32)
    except Exception as e:
        print(f"⚠️ capture_forward_from_now error: {e}")
        return None


def capture_backward(duration_seconds):
    try:
        needed_frames = int(MODEL_SR * float(duration_seconds))
        pieces = []
        frames_collected = 0
        for chunk in reversed(_back_buffer):
            pieces.append(chunk)
            frames_collected += len(chunk)
            if frames_collected >= needed_frames:
                break
        if frames_collected == 0:
            return None
        data = np.concatenate(list(reversed(pieces)), axis=0)
        if len(data) > needed_frames:
            data = data[-needed_frames:]
        return data.astype(np.float32)
    except Exception as e:
        print(f"⚠️ capture_backward error: {e}")
        return None

def save_captured_audio(audio, out_path, sample_rate=None):
    try:
        if isinstance(audio, str) and os.path.exists(audio):
            shutil.copy(audio, out_path)
            return True
        arr = np.asarray(audio)
        if sample_rate is None:
            sample_rate = MODEL_SR
        sf.write(out_path, arr, int(sample_rate), subtype='PCM_16')
        return True
    except Exception as e:
        print(f"⚠️ save_captured_audio error: {e}")
        return False


# -----------------------------
# ChatGPT mode (uses capture_forward for live listening)
# -----------------------------
# -----------------------------
# Voice Activity Detection Capture
# -----------------------------
def capture_until_silence(
    speech_threshold=0.02,      # RMS threshold to detect speech start
    silence_threshold=0.004,     # RMS threshold to detect silence
    min_speech_duration=0.3,     # Minimum speech duration to consider valid
    silence_duration=1.5,        # Duration of silence after speech to end recording
    max_duration=30.0,           # Maximum recording duration
    initial_wait_timeout=15.0    # Timeout waiting for speech to start
):
    """
    Capture audio until silence is detected after speech.
    LED stays blue breathing during capture.
    Returns numpy array of audio or None if no speech detected.
    """
    global exit_chat_mode
    
    # Drain any old buffered chunks
    while True:
        try:
            _ = _record_q.get_nowait()
        except queue.Empty:
            break
    
    frames = []
    speech_detected = False
    speech_start_time = None
    last_speech_time = None
    start_time = time.time()
    
    # Keep LED blue breathing while listening
    set_led_mode('blue_breath')
    
    print("👂 Listening... (waiting for speech)")
    
    while not _stop_event.is_set() and not exit_chat_mode:
        elapsed = time.time() - start_time
        
        # Check timeouts
        if not speech_detected and elapsed > initial_wait_timeout:
            print("⏱️ Timeout waiting for speech")
            return None
        if elapsed > max_duration:
            print("⏱️ Max duration reached")
            break
        
        # Get audio chunk
        try:
            chunk = _record_q.get(timeout=0.5)
        except queue.Empty:
            continue
        
        frames.append(chunk)
        
        # Calculate RMS (energy level)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        
        # Detect speech start
        if rms > speech_threshold:
            if not speech_detected:
                speech_detected = True
                speech_start_time = time.time()
                print(f"🎤 Speech detected! (RMS: {rms:.4f})")
            last_speech_time = time.time()
        
        # Check if speech ended (silence after speech)
        if speech_detected and last_speech_time:
            silence_elapsed = time.time() - last_speech_time
            if silence_elapsed > silence_duration:
                speech_duration = last_speech_time - speech_start_time
                if speech_duration >= min_speech_duration:
                    print(f"🔇 End of speech detected ({speech_duration:.1f}s of speech)")
                    break
                else:
                    # Too short, might be noise, reset and continue listening
                    print(f"⚠️ Speech too short ({speech_duration:.1f}s), might be noise. Continuing...")
                    speech_detected = False
                    speech_start_time = None
                    last_speech_time = None
    
    if not frames or not speech_detected:
        return None
    
    audio = np.concatenate(frames, axis=0)
    return audio.astype(np.float32)


# -----------------------------
# ChatGPT mode (uses VAD-based capture)
# -----------------------------
def chatgpt_chat_mode():
    """
    ChatGPT mode that uses the existing arecord->ffmpeg->_record_q capture pipeline.
    - Pauses wake inference while active
    - Blue breathing LED while listening for voice
    - Automatically stops when silence is detected after speech
    - Transcribes, sends to chat API, speaks reply
    """
    global chat_mode_active, model_paused, _model_paused, exit_chat_mode, current_language

    # If already active, signal exit and return
    if chat_mode_active:
        print("⚠️ ChatGPT mode already active. Button press will exit chat mode.")
        exit_chat_mode = True
        return

    chat_mode_active = True
    exit_chat_mode = False

    # Pause wake inference
    model_paused = True
    _model_paused = True

    print("\n" + "="*50)
    print("🧠 ChatGPT MODE ACTIVATED")
    print("Speak in Tamil or English. Press the ChatGPT button again to exit.")
    print("="*50 + "\n")

    # Welcome phrase (best-effort)
    try:
        if current_language == 'ta':
            speak_text("Chat mode activated. நான் உங்களுக்கு எப்படி உதவ முடியும்?", force_language='ta-IN')
        else:
            speak_text("Chat mode activated. How can I help?", force_language='en-IN')
    except Exception as e:
        print("⚠️ Welcome TTS failed:", e)

    # OpenRouter settings
    OPENROUTER_API_KEY = "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"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    system_prompt = """You are Admini, a smart Raspberry Pi assistant. You can respond in both English and Tamil.Follow below instructions clearly and strictly.

    Instructions:

    1) If the user speaks in Tamil or uses Tamil words, respond in Tamil.
    2) If the user speaks in English, respond in English.
    3) Keep responses brief, clear, and conversational ans answer fully.
    4) Do not use any special tags or symbols like * .
    5) Answer should be like a human friend and conversational.
    6) Think deep and make sure the answer is fully completed.
    ***Important rule***: Always respond only in the language the user used and do not use special tags or symbols like * ."""


    conversation_history = [{"role": "system", "content": system_prompt}]

    last_activity_time = time.time()
    no_speech_count = 0  # Track consecutive no-speech detections
    
    try:
        while not _stop and not exit_chat_mode:
            print("\n" + "-"*40)
            print("🎤 Chat mode: Listening... (LED breathing blue)")
            print("   Speak now - recording will stop automatically when you pause")
            print("-"*40)
            
            # Capture audio with blue breathing LED until silence after speech
            audio = capture_until_silence(
                speech_threshold=0.02,      # Adjust based on your mic sensitivity
                silence_threshold=0.004,
                min_speech_duration=0.3,     # At least 0.3s of speech
                silence_duration=1.5,        # 1.5 seconds of silence ends recording
                max_duration=30.0,           # Max 30 seconds per utterance
                initial_wait_timeout=15.0    # Wait up to 15 seconds for speech to start
            )

            # Check for exit
            if exit_chat_mode:
                break

            # If nothing captured
            if audio is None or (hasattr(audio, "size") and audio.size == 0):
                no_speech_count += 1
                
                # After 2 failed attempts, prompt user
                if no_speech_count >= 2:
                    if time.time() - last_activity_time > 30:
                        try:
                            speak_text("Are you still there? Say something or press the button to exit.")
                        except Exception:
                            pass
                        last_activity_time = time.time()
                        no_speech_count = 0
                continue

            # Reset counters on successful capture
            no_speech_count = 0
            last_activity_time = time.time()
            
            print(f"📝 Captured {len(audio)/MODEL_SR:.1f}s of audio. Processing...")

            # Transcribe & detect language - Yellow LED during processing
            set_led_mode('yellow')
            text = ""
            detected_lang = "en"
            
            try:
                tmp_file = "/tmp/chat_mode_capture.wav"
                saved = save_captured_audio(audio, tmp_file, sample_rate=MODEL_SR)
                if saved:
                    try:
                        result = transcribe_and_detect_language(tmp_file)
                        if isinstance(result, dict):
                            text = (result.get("text") or "").strip()
                            detected_lang = result.get("language", "en")
                        else:
                            text = str(result).strip()
                    except Exception as e:
                        print("⚠️ transcribe_and_detect_language failed:", e)
                        # Fallback attempt
                        
                else:
                    print("⚠️ Could not save captured audio for transcription")
            except Exception as e:
                print("⚠️ Transcription error:", e)

            if not text:
                print("🔇 No clear speech recognized — try again.")
                # Go back to listening (blue breathing)
                set_led_mode('blue_breath')
                time.sleep(0.3)
                continue

            # Show recognized text
            print(f"\n🗣 You ({detected_lang}): {text}")

            # Exit if user asked to quit
            exit_phrases = ['exit', 'quit', 'bye', 'goodbye', 'stop', 'end chat', 
                           'வெளியேறு', 'நிறுத்த��', 'போய்வருகிறேன்']
            if any(w in text.lower() for w in exit_phrases):
                try:
                    if detected_lang.startswith('ta'):
                        speak_text("சரி, போய்வருகிறேன்!", force_language='ta-IN')
                    else:
                        speak_text("Goodbye! Talk to you later.", force_language='en-IN')
                except Exception:
                    pass
                break
            text= text + "Answer in " + ("Tamil" if detected_lang.startswith('ta') else "English")
            # Append user message to conversation history
            conversation_history.append({"role": "user", "content": text})
            # Keep history bounded
            if len(conversation_history) > 11:
                conversation_history = [conversation_history[0]] + conversation_history[-10:]

            # Call chat API (yellow LED continues)
            print("🔄 Getting response from AI...")
            reply = None
            try:
                payload = {
                    "model": "amazon/nova-premier-v1",
                    "messages": conversation_history,
                    "max_tokens": 500
                }
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
                
                if resp.status_code == 200:
                    body = resp.json()
                    if isinstance(body, dict) and "choices" in body and body["choices"]:
                        try:
                            reply = body["choices"][0]["message"]["content"].strip()
                        except Exception:
                            reply = str(body["choices"][0]).strip()
                    else:
                        if isinstance(body, dict) and body.get("message"):
                            reply = str(body.get("message"))
                        elif isinstance(body, str):
                            reply = body
                        else:
                            reply = "Sorry, I didn't get a valid response."
                else:
                    print(f"⚠️ Chat API returned {resp.status_code}: {resp.text[:400]}")
                    try:
                        err = resp.json()
                        if isinstance(err, dict) and err.get("error"):
                            reply = err["error"].get("message", "Chat service error.")
                        else:
                            reply = "Chat service returned an error."
                    except Exception:
                        reply = "Chat service returned an error."
            except requests.exceptions.Timeout:
                print("⚠️ Chat API timeout")
                reply = "Sorry, the response took too long. Please try again."
            except Exception as e:
                print("⚠️ Chat API request failed:", e)
                reply = "Sorry, I couldn't reach the chat service right now."

            # Speak reply (yellow LED during TTS)
            if reply:
                # Add to conversation history
                conversation_history.append({"role": "assistant", "content": reply})
                
                print(f"\n🤖 Admini: {reply}")
                force_lang = 'ta-IN' if detected_lang.startswith('ta') else 'en-IN'
                try:
                    speak_text(reply, force_language=force_lang, led_color='yellow')
                except Exception as e:
                    print("⚠️ TTS failed:", e)

            # Small pause before next listening cycle
            time.sleep(0.3)

    except Exception as e:
        print("⚠️ chatgpt_chat_mode fatal error:", e)
        import traceback
        traceback.print_exc()

    finally:
        # Exit message
        try:
            exit_msg = "Exiting chat mode" if current_language != 'ta' else "Chat mode முடிவடைகிறது"
            speak_text(exit_msg, force_language=('ta-IN' if current_language == 'ta' else 'en-IN'))
        except Exception:
            pass

        # Restore state
        chat_mode_active = False
        exit_chat_mode = False
        model_paused = False
        _model_paused = False
        set_led_mode('blue_run')
        print("\n✅ ChatGPT mode exited — wake detection resumed.\n")

# -----------------------------
# Wake command (processing) helpers
# -----------------------------
def listen_mic_command_from_audio(audio_array):
    """
    Process a captured audio numpy array (mono float32 @ MODEL_SR),
    run preprocess -> transcription -> chat_api -> speak_text
    """
    global current_language
    if chat_mode_active:
        return
    if audio_array is None:
        print("⚠️ No audio provided to listen_mic_command_from_audio()")
        return
    audio = np.asarray(audio_array).astype(np.float32)
    temp_wav = "/tmp/capture_raw_from_stream.wav"
    saved = save_captured_audio(audio, temp_wav, sample_rate=MODEL_SR)
    if not saved:
        print("⚠️ Could not save captured audio to disk")
        return
    processed_wav = "/tmp/capture_proc_from_stream.wav"
    try:
        preprocess_wav(temp_wav, processed_wav, target_fs=MODEL_SR, do_vad=True, do_noise_reduction=True)
        audio_for_transcribe = processed_wav
    except Exception as e:
        print(f"⚠️ Preprocessing failed, falling back to raw: {e}")
        audio_for_transcribe = temp_wav
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_for_transcribe) as source:
        aud = recognizer.record(source)
    set_led_mode('yellow')
    text = None
    detected_lang = None
    audio_path = audio_for_transcribe
    try:
        text = translate_audio_to_english(audio_path)
        detected_lang = 'en'
        print(f"🗣 Recognized (EN via translate_audio_to_english): {text}")
    except Exception:
        try:
            text = recognizer.recognize_google(aud, language="ta-IN")
            detected_lang = 'ta'
            print(f"🗣 Recognized (TA): {text}")
            print("🔄 Translating from Tamil...")
            try:
                english_text = translate_audio_to_english(audio_path)
                if english_text and english_text.strip():
                    text = english_text
                    detected_lang = 'en'
                    print(f"🗣 Translated: {english_text}")
            except Exception as ex2:
                print(f"⚠️ Tamil->EN translate failed: {ex2}")
        except sr.UnknownValueError:
            print("🔇 No clear speech detected after preprocessing.")
            set_led_mode('blue_run')
            return
        except Exception as e2:
            print(f"⚠️ Recognizer error: {e2}")
            set_led_mode('blue_run')
            return
    if text:
        current_language = detected_lang
        result = chat(text)
        if result.get("status") == "success":
            api_response = result.get("api_response")
            print(f"🤖 Response: {api_response}")
            speak_text(str(api_response["message"]), force_language='ta-IN', led_color='yellow')
    set_led_mode('blue_run')

# -----------------------------
# Wake detection internals (model)
# -----------------------------
class ConvLSTMWakeWord(nn.Module):
    def __init__(self, n_mels: int, num_classes: int = 2, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.lstm = nn.LSTM(input_size=32 * n_mels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(2 * hidden_size, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        B, T, F = x.shape
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv(x)
        B, C, F, Tprime = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, Tprime, C * F)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        logits = self.classifier(last)
        return logits

# load model
device = torch.device("cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device) if os.path.exists(MODEL_PATH) else {"model_state": None}
label_map = checkpoint.get("label_map", None)
if label_map is not None:
    num_classes = len(label_map)
else:
    num_classes = checkpoint.get("num_classes", None) or 2
model = ConvLSTMWakeWord(n_mels=N_MELS, num_classes=num_classes)
if checkpoint.get("model_state") is not None:
    try:
        model.load_state_dict(checkpoint["model_state"])
        print("[Model] Loaded model_state")
    except Exception as e:
        print("[Model] Partial load fallback:", e)
        state = model.state_dict()
        ck_state = checkpoint["model_state"]
        matched = 0
        for k in list(state.keys()):
            if k in ck_state and ck_state[k].shape == state[k].shape:
                state[k] = ck_state[k]
                matched += 1
        model.load_state_dict(state)
        print(f"[Model] Partially loaded {matched}/{len(state)} tensors.")
model.to(device).eval()

inv_label_map = {i: f"class_{i}" for i in range(num_classes)}
wake_idx = 0
if label_map:
    for candidate in ("admini", "wake", "wakeword", "keyword"):
        if candidate in label_map:
            wake_idx = label_map[candidate]; break
print(f"[Model] wake_idx={wake_idx}")

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=MODEL_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, center=True, power=2.0)
db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

def audio_to_features(audio_np):
    wav = torch.from_numpy(audio_np).float()
    if wav.numel() > TARGET_LEN:
        wav = wav[-TARGET_LEN:]
    elif wav.numel() < TARGET_LEN:
        pad_len = TARGET_LEN - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad_len))
    wav = wav.unsqueeze(0)
    mel = mel_transform(wav)
    mel_db = db_transform(mel)
    mean = mel_db.mean()
    std = mel_db.std().clamp(min=1e-5)
    mel_db = (mel_db - mean) / std
    feat = mel_db.squeeze(0).transpose(0, 1)
    return feat

# inference state
_hit_times = deque()
_last_confirm = 0.0
_model_paused = False

def on_wakeword_detected():
    """
    Called when wake detected. Pause inference, set breathing blue,
    record AFTER_WAKE_RECORD_SECONDS from live stream (forward capture),
    process synchronously (transcribe + chat + TTS), then resume inference.
    """
    global model_paused, _model_paused

    print("\n✅ WAKE WORD DETECTED! (callback)\n")
    model_paused = True
    _model_paused = True

    try:
        # visual & servo feedback
        # servo: raise for wake unless notifications have priority
        try:
            if not notification_pending:
                # raise servo for this wake event (non-blocking)
                threading.Thread(target=raise_servo, daemon=True).start()
            else:
                # notification pending takes priority; ensure servo is raised
                threading.Thread(target=raise_servo, daemon=True).start()
        except Exception:
            pass

        # set breathing blue for recording
        set_led_mode('blue_breath')

        # capture forward N seconds
        print(f"[on_wakeword_detected] starting forward capture for {AFTER_WAKE_RECORD_SECONDS} seconds...")
        # Capture strictly *after* wake: ignore already-buffered chunks and wait for new incoming audio
        audio_full = capture_forward_from_now(AFTER_WAKE_RECORD_SECONDS, timeout_per_chunk=60.0)
        if audio_full is None:
            print("[on_wakeword_detected] forward capture returned None")
            set_led_mode('yellow')
            # fallback to backward buffer
            fallback = capture_backward(10)
            if fallback is not None:
                print("[on_wakeword_detected] using fallback last-10s buffer")
                # process synchronously
                listen_mic_command_from_audio(fallback)
            else:
                print("[on_wakeword_detected] no fallback available")
        else:
            print(f"[on_wakeword_detected] captured {len(audio_full)} samples ({len(audio_full)/MODEL_SR:.1f}s). Processing now...")
            # process synchronously (block until finished) so inference remains paused
            listen_mic_command_from_audio(audio_full)

    except Exception as e:
        print(f"[on_wakeword_detected] error: {e}")
    finally:
        # resume inference and set running blue
        model_paused = False
        _model_paused = False
        # lower servo only if there are no pending notifications
        try:
            if not notification_pending:
                lower_servo()
        except Exception:
            pass
        set_led_mode('blue_run')
    print("[on_wakeword_detected] returning to listen state\n")

def _infer_worker():
    global _hit_times, _last_confirm, _model_paused
    _hit_times = deque()
    audio_buffer = deque(maxlen=2)
    print("[Wake] warming up...")
    dummy = np.random.randn(TARGET_LEN).astype(np.float32) * 0.1
    _ = audio_to_features(dummy)
    print("[Wake] listening (buffering 2s)...")
    chunk_count = 0
    while not _stop_event.is_set():
        try:
            chunk = _inference_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if _model_paused or model_paused:
            _inference_q.task_done()
            continue
        chunk_count += 1
        ts = time.time()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        audio_buffer.append(chunk)
        if len(audio_buffer) < 2:
            print(f"[{chunk_count}] buffering {len(audio_buffer)}/2")
            _inference_q.task_done()
            continue
        audio_2s = np.concatenate(list(audio_buffer), axis=0)
        total_rms = float(np.sqrt(np.mean(audio_2s ** 2)))
        print(f"[Chunk {chunk_count}] samples={len(audio_2s)} RMS={total_rms:.6f}")
        if total_rms < 0.003:
            _inference_q.task_done()
            continue
        try:
            feat = audio_to_features(audio_2s)
            feat_b = feat.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(feat_b).squeeze(0)
                if logits.ndim == 0 or logits.numel() == 1:
                    prob_wake = float(torch.sigmoid(torch.tensor(float(logits.item()))).item())
                else:
                    probs = torch.softmax(logits, dim=0).cpu().numpy()
                    prob_wake = float(probs[wake_idx]) if 0 <= wake_idx < len(probs) else float(probs.max())
        except Exception as e:
            print("[Inference] error:", e)
            _inference_q.task_done()
            continue
        in_cooldown = (ts - _last_confirm) < COOLDOWN
        confirmed = False
        if not in_cooldown:
            if prob_wake >= VERY_HIGH_THRESHOLD:
                confirmed = True
            elif prob_wake >= THRESHOLD:
                _hit_times.append(ts)
                while _hit_times and (ts - _hit_times[0]) > HITS_WINDOW:
                    _hit_times.popleft()
                if len(_hit_times) >= REQUIRED_HITS:
                    confirmed = True
        bar_len = int(prob_wake * 30)
        bar = '█' * min(bar_len, 30) + '░' * max(0, 30 - bar_len)
        icon = "🔥" if prob_wake >= 0.6 else "🟡" if prob_wake >= 0.4 else "🟠" if prob_wake >= 0.2 else "⚪"
        hits = len(_hit_times)
        print(f"\n  {icon} [{bar}] wake_prob: {prob_wake:.3f}  Hits: {hits}/{REQUIRED_HITS}")
        if confirmed:
            _last_confirm = ts
            _hit_times.clear()
            print("[Wake] DETECTED -> calling on_wakeword_detected()")
            threading.Thread(target=on_wakeword_detected, daemon=True).start()
        _inference_q.task_done()

# -----------------------------
# Button monitors & signal
# -----------------------------
def chatgpt_button_monitor():
    global _stop, exit_chat_mode, chat_mode_active
    pressed = False
    last = 0
    while not _stop:
        state = GPIO.input(SWITCH_PIN) == GPIO.LOW
        now = time.time()
        if state and not pressed and (now - last > 0.5):
            pressed = True
            last = now
            if chat_mode_active:
                exit_chat_mode = True
            else:
                threading.Thread(target=chatgpt_chat_mode, daemon=True).start()
        elif not state:
            pressed = False
        time.sleep(0.05)

def signal_handler(sig, frame):
    global _stop
    print("\n⛔ Shutting down...")
    _stop = True
    _stop_event.set()

signal.signal(signal.SIGINT, signal_handler)

# -----------------------------
# Main
# -----------------------------
def main():
    global _stop, _led_running
    print("\n" + "="*60)
    print("🚀 ADMINI ASSISTANT - Integrated (no live_module/wake_module)")
    print("="*60)
    print(f"📍 Wake model: {MODEL_PATH}")
    print(f"📍 Audio device: {ARECORD_DEVICE} (arecord->ffmpeg pipeline)")
    print("="*60 + "\n")

    # start background threads
    threading.Thread(target=chatgpt_button_monitor, daemon=True).start()
    threading.Thread(target=notification_button_monitor, daemon=True).start()
    threading.Thread(target=notification_monitor, daemon=True).start()
    threading.Thread(target=notification_button_press_handler, daemon=True).start()
    print("✅ Background threads started")

    # start LED thread
    set_led_mode('blue_run')
    led_t = threading.Thread(target=led_thread_fn, daemon=True)
    led_t.start()

    # start wakeword capture pipeline
    try:
        arecord_proc, ffmpeg_proc = start_audio_stream_process()
        reader_t = threading.Thread(target=audio_reader_worker, args=(ffmpeg_proc,), daemon=True)
        reader_t.start()
        infer_t = threading.Thread(target=_infer_worker, daemon=True)
        infer_t.start()
    except Exception as e:
        print("[Main] failed to start audio pipeline:", e)
        _stop = True

    print("✅ System ready!")
    # ensure servo starts lowered at boot if no notification pending
    try:
        if notification_pending:
            threading.Thread(target=raise_servo, daemon=True).start()
        else:
            # ensure starting position is down
            lower_servo()
    except Exception:
        pass
    print("💡 Speak 'Admini' to activate")
    print("💡 Press GPIO27 for ChatGPT mode")
    print("💡 Press GPIO22 to read notifications\n")

    try:
        while not _stop:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        _stop = True
        _stop_event.set()
        try:
            ffmpeg_proc.terminate()
        except Exception:
            pass
        try:
            arecord_proc.terminate()
        except Exception:
            pass
        GPIO.cleanup()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()