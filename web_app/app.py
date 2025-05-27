import os
import sys
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model

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

# Kamera görüntü üreten fonksiyon
def gen_frames():
    global sequence, sentence, frame_lock_counter
    print("🎥 gen_frames() başladı")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, frame = cap.read()
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


# Rotalar

@app.route('/')
def index():
    return render_template("mainmenu.html")

@app.route('/SignToText')
def index2():
    return render_template("live_sign.html")

@app.route('/TextToSign')
def index3():
    return render_template("animation.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_predictions')
def get_predictions():
    return jsonify(sentence=sentence)

# Uygulamayı çalıştır
if __name__ == '__main__':
    import webbrowser
    webbrowser.open("http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)