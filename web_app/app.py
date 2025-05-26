from flask import Flask, request, render_template, jsonify
import base64
import cv2
import numpy as np
import os
import sys
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sign_language_recognition'))
sys.path.append(BASE_DIR)  

# 🔁 Ortak modülleri şimdi import et
from main import mp_holistic, mediapipe_detection, draw_styled_landmarks
from feature_extraction import extract_features


# ✅ Yol tanımları
CSV_PATH = os.path.join(BASE_DIR, "class_map.csv")
MODEL_PATH = os.path.join(BASE_DIR, "action2.h5")
KEYPOINT_PATH = os.path.join(BASE_DIR, "keypoints_175_40")

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["ClassID"] = df["ClassID"].astype(str).str.zfill(4)
id_to_name = dict(zip(df["ClassID"], df["ClassName_tr"]))

folder_ids = [d for d in os.listdir(KEYPOINT_PATH) if os.path.isdir(os.path.join(KEYPOINT_PATH, d))]
actions = np.array([id_to_name[d] for d in folder_ids if d in id_to_name])

model = load_model(MODEL_PATH)

sequence = []
sentence = []
threshold = 0.99
lock_duration = 20
frame_lock_counter = 0

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("mainmenu.html")
@app.route('/SignToText')
def index2():
    return render_template("camera.html")
@app.route('/TextToSign')
def index3():
    return render_template("animation.html")

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global sequence, sentence, frame_lock_counter
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        image, features = extract_features(image, results)

        sequence.append(features)
        sequence = sequence[-40:]

        if len(sequence) == 40:
            seq = np.array(sequence)
            X_r = np.concatenate((seq[:, 162:225], seq[:, 229:230], seq[:, 231:246], seq[:, 261:277]), axis=-1)
            X_l = np.concatenate((seq[:, 99:162], seq[:, 230:231], seq[:, 246:261], seq[:, 277:293]), axis=-1)
            X_p = np.concatenate((seq[:, 0:99], seq[:, 225:229], seq[:, 299:307], seq[:, 293:299]), axis=-1)

            res = model.predict([
                np.expand_dims(X_r, axis=0),
                np.expand_dims(X_l, axis=0),
                np.expand_dims(X_p, axis=0)
            ], verbose=0)[0]

            if res[np.argmax(res)] > threshold:
                predicted_word = actions[np.argmax(res)]
                if frame_lock_counter == 0:
                    if len(sentence) == 0 or predicted_word != sentence[-1]:
                        sentence.append(predicted_word)
                        print("Tahmin:", predicted_word)
                    frame_lock_counter = lock_duration
                else:
                    frame_lock_counter -= 1

            if len(sentence) > 5:
                sentence = sentence[-5:]

    _, buffer = cv2.imencode('.jpg', image)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'processed': f"data:image/jpeg;base64,{encoded}"})


import webbrowser

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
