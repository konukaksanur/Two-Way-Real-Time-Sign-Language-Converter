import os
import sys
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model
import threading
import time

# Proje kök dizinini path'e ekle
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

# sign_language_recognition içinden importlar
from sign_language_recognition.main import mp_holistic, mediapipe_detection, draw_styled_landmarks
from sign_language_recognition.feature_extraction import extract_features

# Flask başlat
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Dosya yolları
CSV_PATH = os.path.join(BASE_DIR, "sign_language_recognition/class_map.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sign_language_recognition/action2.h5")
KEYPOINT_PATH = os.path.join(BASE_DIR, "sign_language_recognition/keypoints_175_40")

# Verileri yükle
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["ClassID"] = df["ClassID"].astype(str).str.zfill(4)
id_to_name = dict(zip(df["ClassID"], df["ClassName_tr"]))
folder_ids = [d for d in os.listdir(KEYPOINT_PATH) if os.path.isdir(os.path.join(KEYPOINT_PATH, d))]
actions = np.array([id_to_name[d] for d in folder_ids])

# Modeli yükle
model = load_model(MODEL_PATH)

# Küresel değişkenler
sequence = []
sentence = []
threshold = 0.99
lock_duration = 20
frame_lock_counter = 0

# Kamera kontrolü için
camera_active = False
camera_lock = threading.Lock()
current_camera = None

def release_camera():
    """Kamerayı güvenli bir şekilde serbest bırak"""
    global current_camera, camera_active
    with camera_lock:
        if current_camera is not None:
            try:
                current_camera.release()
                print("🔴 Kamera serbest bırakıldı")
            except:
                pass
            current_camera = None
        camera_active = False

def gen_frames():
    global sequence, sentence, frame_lock_counter, camera_active, current_camera
    print("🎥 gen_frames() başladı")
    
    with camera_lock:
        if current_camera is not None:
            release_camera()
        
        current_camera = cv2.VideoCapture(0)
        if not current_camera.isOpened():
            print("❌ Kamera açılamadı!")
            return
        
        current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_active = True

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while camera_active:
                with camera_lock:
                    if not camera_active or current_camera is None:
                        break
                    
                    success, frame = current_camera.read()
                    if not success:
                        break

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                image, features = extract_features(image, results)

                sequence.append(features)
                sequence = sequence[-40:]

                if len(sequence) == 40:
                    sequence_arr = np.array(sequence)

                    X_r = np.concatenate([
                        sequence_arr[:, 162:225], sequence_arr[:, 229:230],
                        sequence_arr[:, 231:246], sequence_arr[:, 261:277]
                    ], axis=-1)

                    X_l = np.concatenate([
                        sequence_arr[:, 99:162], sequence_arr[:, 230:231],
                        sequence_arr[:, 246:261], sequence_arr[:, 277:293]
                    ], axis=-1)

                    X_p = np.concatenate([
                        sequence_arr[:, 0:99], sequence_arr[:, 225:229],
                        sequence_arr[:, 299:307], sequence_arr[:, 293:299]
                    ], axis=-1)

                    res = model.predict([
                        np.expand_dims(X_r, axis=0),
                        np.expand_dims(X_l, axis=0),
                        np.expand_dims(X_p, axis=0)
                    ], verbose=0)[0]

                    best_action = actions[np.argmax(res)]
                    if res[np.argmax(res)] > threshold:
                        if frame_lock_counter == 0:
                            if len(sentence) == 0 or best_action != sentence[-1]:
                                sentence.append(best_action)
                                print("🧠 Prediction:", best_action)
                            frame_lock_counter = lock_duration
                        else:
                            frame_lock_counter -= 1

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    print("❌ Görüntü encode edilemedi")
                    continue

                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
    except Exception as e:
        print(f"❌ gen_frames hatası: {e}")
    finally:
        release_camera()
        print("✅ gen_frames() tamamlandı, kamera kapatıldı")

# Rotalar

@app.route('/')
def index():
    # Ana sayfaya geçildiğinde kamerayı kapat
    release_camera()
    return render_template("mainmenu.html")

@app.route('/SignToText')
def index2():
    # SignToText sayfasına geçildiğinde önceki kamerayı kapat
    release_camera()
    return render_template("live_sign.html")

@app.route('/TextToSign')
def index3():
    # TextToSign sayfasına geçildiğinde kamerayı kapat
    release_camera()
    return render_template("animation.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_predictions')
def get_predictions():
    return jsonify(sentence=sentence)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Kamerayı manuel olarak durdurma endpoint'i"""
    release_camera()
    return jsonify(status="camera_stopped")

# Uygulama kapatılırken kamerayı serbest bırak
import atexit
atexit.register(release_camera)

# Uygulamayı çalıştır
if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)