import os
import sys
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
import threading
import atexit

# Add the project root to the Python path
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

# Import sign language recognition utilities
from sign_language_recognition.main import mp_holistic, mediapipe_detection, draw_styled_landmarks
from sign_language_recognition.feature_extraction import extract_features


# Initialize Flask application
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Define file paths
CSV_PATH = os.path.join(BASE_DIR, "sign_language_recognition/class_map.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sign_language_recognition/action4.h5")

# Load class names in the same order as model training
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["ClassID"] = pd.to_numeric(df["ClassID"])
df = df.sort_values("ClassID")

actions = df["ClassName_tr"].to_numpy()


# Load the trained model
model = load_model(MODEL_PATH, compile=False)
sequence = []
sentence = []
threshold = 0.99
lock_duration = 20
frame_lock_counter = 0

# Camera management
camera_active = False
camera_lock = threading.RLock()
current_camera = None

def release_camera():
    global current_camera, camera_active
    with camera_lock:
        if current_camera is not None:
            try:
                current_camera.release()
                print("🔴 Camera released")
            except:
                pass
            current_camera = None
        camera_active = False

def gen_frames():
    global sequence, sentence, frame_lock_counter, camera_active, current_camera
    print("🎥 Frame generation started")
    
    with camera_lock:
        if current_camera is not None:
            release_camera()
        
        current_camera = cv2.VideoCapture(0)
        if not current_camera.isOpened():
            print("❌ Failed to open camera")
            return
        
        current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

                    best_idx = np.argmax(res)
                    best_action = actions[best_idx]
                    confidence = res[best_idx]

                    print(f"🧠 Top prediction: {best_action} | Confidence: {confidence:.2%}")

                    # Decrease the lock counter on every processed frame
                    if frame_lock_counter > 0:
                        frame_lock_counter -= 1

                    # Add a prediction only when confidence is high and the lock has expired
                    if confidence > threshold and frame_lock_counter == 0:
                        if len(sentence) == 0 or best_action != sentence[-1]:
                            sentence.append(best_action)
                            print(f"✅ Added: {best_action} | Confidence: {confidence:.2%}")
                            frame_lock_counter = lock_duration

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    print("❌ Failed to encode frame")
                    continue

                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
    except Exception as e:
        print(f"❌ Frame generation error: {e}")
    finally:
        release_camera()
        print("✅ Frame generation completed, camera closed")

# Flask routes
@app.route('/')
def index():
    release_camera()
    return render_template("mainmenu.html")

@app.route('/SignToText')
def index2():
    release_camera()
    return render_template("live_sign.html")

@app.route('/TextToSign')
def index3():
    release_camera()
    return render_template("animation.html")

@app.route('/video_feed')
def video_feed():
    if camera_active:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("", mimetype='text/plain')

@app.route('/get_predictions')
def get_predictions():
    return jsonify(sentence=sentence)

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    
    try:
        if not camera_active:
            camera_active = True
            print("🎥 Starting camera...")
            return jsonify(status="camera_started")
        else:
            return jsonify(status="camera_already_active")
    except Exception as e:
        print(f"❌ Camera startup error: {e}")
        return jsonify(status="error", message=str(e))

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    release_camera()
    return jsonify(status="camera_stopped")

@app.route('/clear_predictions', methods=['POST'])
def clear_predictions():
    global sentence, sequence
    sentence = []
    sequence = []  
    print("🗑️ Predictions cleared")
    return jsonify(status="predictions_cleared")

@app.route('/camera_status')
def camera_status():
    return jsonify(
        active=camera_active,
        predictions_count=len(sentence)
    )

atexit.register(release_camera)

# Run the Flask application
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)