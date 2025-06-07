import cv2
import mediapipe as mp
import os
import numpy as np
import pyttsx3
import pickle
import websocket
import keyboard
import json
import threading
import webbrowser
import shutil
import stat
import time
import pygame
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue

WS_URL = "ws://localhost:1880/ws/keyboard"
NODE_RED_UI_URL = "http://127.0.0.1:1880/ui/#!/0"
FACE_DATA_DIR = "face_data"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"
BGM_FILE = "work.mp3"
VOSK_MODEL_PATH = "model"

engine = pyttsx3.init()
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {}
greeted = set()
gesture_state = None
ws = None
app_running = True
mode_selected = None
name_collected = False
current_gesture = None
pygame.mixer.init()

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"[AUDIO STATUS] {status}")
    q.put(bytes(indata))

def play_bgm():
    if os.path.exists(BGM_FILE):
        pygame.mixer.music.load(BGM_FILE)
        pygame.mixer.music.play(-1)

def stop_bgm():
    pygame.mixer.music.stop()

def background_voice_listener():
    global current_gesture
    model = Model(VOSK_MODEL_PATH)
    recognizer_vosk = KaldiRecognizer(model, 16000)
    stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=audio_callback)
    stream.start()
    print("[VOICE] Offline voice listener started...")
    active_audio_key = None

    while app_running:
        data = q.get()
        if recognizer_vosk.AcceptWaveform(data):
            result = json.loads(recognizer_vosk.Result())
            command = result.get("text", "").lower()
            if not command:
                continue
            if "jarvis start the music" in command:
                print("[VOICE] start music")
                play_bgm()
            elif "jarvis mute" in command:
                print("[VOICE] stop music")
                stop_bgm()
            elif "move forward" in command:
                if active_audio_key and active_audio_key != "right":
                    send_key_payload(active_audio_key, False)
                send_key_payload("space", False)
                send_key_payload("right", True)
                active_audio_key = "right"
            elif "move backward" in command:
                if active_audio_key and active_audio_key != "left":
                    send_key_payload(active_audio_key, False)
                send_key_payload("space", False)
                send_key_payload("left", True)
                active_audio_key = "left"
            elif "stop the belt" in command:
                send_key_payload("right", False)
                send_key_payload("left", False)
                send_key_payload("up", False)
                send_key_payload("down", False)
                send_key_payload("space", True)
                active_audio_key = "space"

def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    wrist = landmarks[0]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    if thumb_tip.y < thumb_ip.y < wrist.y:
        return "up"
    elif thumb_tip.y > wrist.y and thumb_ip.y > wrist.y:
        return "down"
    elif all(landmarks[i].y < wrist.y for i in [8, 12, 16, 20]) and thumb_tip.x < index_tip.x:
        return "left"
    elif (index_tip.y < wrist.y and all(landmarks[i].y > wrist.y for i in [12, 16, 20])):
        return "right"
    elif (thumb_tip.y < wrist.y and pinky_tip.y < wrist.y and all(landmarks[i].y > wrist.y for i in [8, 12, 16])):
        return "space"
    return None

def greet(name):
    msg = f"Hey {name}, how are you doing? Welcome to G B S I O T. Let's work on the CP system!"
    print(msg)
    engine.say(msg)
    engine.runAndWait()

def capture_and_save(name, cap, face_loc):
    save_path = os.path.join(FACE_DATA_DIR, name)
    os.makedirs(save_path, exist_ok=True)
    x, y, w, h = face_loc
    count = 0
    while count < 10:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = gray[y:y+h, x:x+w]
        if face.size:
            resized = cv2.resize(face, (200, 200))
            cv2.imwrite(f"{save_path}/{count+1}.jpg", resized)
            cv2.putText(frame, f"Capturing {count+1}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            count += 1
        cv2.imshow("Capture", frame)
        cv2.waitKey(100)

def train_model():
    faces, ids, label_map = [], [], {}
    for i, person in enumerate(os.listdir(FACE_DATA_DIR)):
        for img in os.listdir(os.path.join(FACE_DATA_DIR, person)):
            path = os.path.join(FACE_DATA_DIR, person, img)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                faces.append(image)
                ids.append(i)
        label_map[i] = person
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)
    return {v: k for k, v in label_map.items()}

def send_key_payload(key, state):
    global gesture_state, ws
    if ws:
        try:
            ws.send(json.dumps({key: state}))
            print(f"WebSocket Sent: {key} -> {state}")
        except Exception as e:
            print(f"WebSocket error: {e}")

def run_ws_keyboard_control():
    global ws
    try:
        ws = websocket.create_connection(WS_URL)
        print("[✓] WebSocket connected to Node-RED (Manual Mode).")
        keyboard.on_press_key("right", lambda e: send_key_payload("right", True))
        keyboard.on_release_key("right", lambda e: send_key_payload("right", False))
        keyboard.on_press_key("left", lambda e: send_key_payload("left", True))
        keyboard.on_release_key("left", lambda e: send_key_payload("left", False))
        keyboard.on_press_key("up", lambda e: send_key_payload("up", True))
        keyboard.on_release_key("up", lambda e: send_key_payload("up", False))
        keyboard.on_press_key("down", lambda e: send_key_payload("down", True))
        keyboard.on_release_key("down", lambda e: send_key_payload("down", False))
        keyboard.on_press_key("space", lambda e: send_key_payload("space", True))
        keyboard.on_release_key("space", lambda e: send_key_payload("space", False))
    except Exception as e:
        print(f"WebSocket Thread Error: {e}")

def run_ws_connection_only():
    global ws
    try:
        ws = websocket.create_connection(WS_URL)
        print("[✓] WebSocket connected for gesture mode.")
    except Exception as e:
        print(f"WebSocket Connection Error: {e}")

def ask_to_clear_face_data():
    ans = input("Do you want to clear all face data? (y/n): ").strip().lower()
    if ans == 'y':
        try:
            for root, dirs, files in os.walk(FACE_DATA_DIR):
                for file in files:
                    filepath = os.path.join(root, file)
                    os.chmod(filepath, stat.S_IWRITE)
            shutil.rmtree(FACE_DATA_DIR, ignore_errors=True)
            if os.path.exists(TRAINER_FILE):
                os.remove(TRAINER_FILE)
            if os.path.exists(LABELS_FILE):
                os.remove(LABELS_FILE)
            print("[✓] Face data cleared successfully.")
        except Exception as e:
            print(f"[!] Failed to clear face data: {e}")

cap = cv2.VideoCapture(1)
print("[INFO] System running...")

while app_running:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = face_detector.process(rgb)

    if result.detections and not name_collected:
        ih, iw = frame.shape[:2]
        det = result.detections[0]
        box = det.location_data.relative_bounding_box
        x, y, w, h = int(box.xmin * iw), int(box.ymin * ih), int(box.width * iw), int(box.height * ih)

        if 0 <= x < iw and 0 <= y < ih:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "New Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("Face System", frame)
            cv2.waitKey(500)
            name = input("Enter your name: ").strip()
            if name:
                capture_and_save(name, cap, (x, y, w, h))
                labels = train_model()
                greet(name)
                name_collected = True
                mode = input("Select mode: [M]anual or [G]esture: ").strip().lower()
                webbrowser.open(NODE_RED_UI_URL)
                if mode == 'm':
                    mode_selected = "manual"
                    play_bgm()
                    threading.Thread(target=run_ws_keyboard_control, daemon=True).start()
                elif mode == 'g':
                    mode_selected = "gesture"
                    play_bgm()
                    run_ws_connection_only()
                threading.Thread(target=background_voice_listener, daemon=True).start()

    if mode_selected == "gesture":
        hand_result = hands_detector.process(rgb)
        if hand_result.multi_hand_landmarks:
            landmarks = hand_result.multi_hand_landmarks[0].landmark
            gesture_key = detect_gesture(landmarks)
            if gesture_key:
                if gesture_key != current_gesture:
                    if current_gesture:
                        send_key_payload(current_gesture, False)
                    send_key_payload(gesture_key, True)
                    current_gesture = gesture_key

    if keyboard.is_pressed("esc"):
        app_running = False
        break

    cv2.imshow("Face + Gesture System", frame)
    if cv2.getWindowProperty("Face + Gesture System", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
stop_bgm()
ask_to_clear_face_data()
