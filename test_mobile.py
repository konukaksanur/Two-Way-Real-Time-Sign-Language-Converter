import base64
from flask import Flask, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from main import mp_holistic, mediapipe_detection, extract_features,draw_styled_landmarks
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Model ve action sınıfları
DATA_PATH = "keypoints"
actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
model = load_model('action3.h5')
threshold = 0.96

# Global durumlar
sequence = []
sentence = []
lock_duration = 20
frame_lock_counter = 0

def predict_from_frame_with_image(frame):
    global sequence, sentence, frame_lock_counter

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
        image, results = mediapipe_detection(image, holistic)
        image, features = extract_features(image, results)

        # Landmark çizimlerini ekle
        image = draw_styled_landmarks(image, results)

    if features is None:
        return "", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    sequence.append(features)
    sequence = sequence[-60:]

    if len(sequence) < 60:
        return "", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    sequence_arr = np.array(sequence)

    X_r = np.concatenate((sequence_arr[:, 162:225], sequence_arr[:, 229:230],
                          sequence_arr[:, 231:246], sequence_arr[:, 261:277]), axis=-1)
    X_l = np.concatenate((sequence_arr[:, 99:162], sequence_arr[:, 230:231],
                          sequence_arr[:, 246:261], sequence_arr[:, 277:293]), axis=-1)
    X_p = np.concatenate((sequence_arr[:, 0:99], sequence_arr[:, 225:229],
                          sequence_arr[:, 299:307], sequence_arr[:, 293:299]), axis=-1)

    res = model.predict([
        np.expand_dims(X_r, axis=0),
        np.expand_dims(X_l, axis=0),
        np.expand_dims(X_p, axis=0)
    ], verbose=0)[0]

    best_action_idx = np.argmax(res)
    best_action_name = actions[best_action_idx]
    best_action_prob = res[best_action_idx]

    if best_action_prob > threshold:
        if frame_lock_counter == 0:
            if not sentence or best_action_name != sentence[-1]:
                sentence.append(best_action_name)
            frame_lock_counter = lock_duration
        else:
            frame_lock_counter -= 1

    if len(sentence) > 5:
        sentence = sentence[-5:]

    return sentence[-1] if sentence else "", cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    try:
        pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(pil_image)
        frame = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": "Image processing failed", "details": str(e)}), 500

    # Tahmin ve görsel işleme
    prediction, annotated_image = predict_from_frame_with_image(frame)

    # Görseli base64'e çevir
    _, buffer = cv2.imencode('.jpg', annotated_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "prediction": prediction,
        "image": img_base64
    })


if __name__ == "__main__":
    # Tüm ağdan erişilebilir olsun (mobil cihaz için)
    app.run(host="0.0.0.0", port=5001, debug=False)
