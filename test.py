import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# Mediapipe models
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Model ve etiketleri yükle
MODEL_PATH = "actions.h5"
DATA_PATH = "keypoints"  # Eğitimde kullanılan veri yolu
actions = np.array(os.listdir(DATA_PATH))  # Sınıf isimlerini al
model = load_model(MODEL_PATH)

# **Tahmin için eşik değeri**
THRESHOLD = 0.8

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı.")
        return

    sequence = []  # Model için giriş dizisi

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Landmarkları çıkar
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Modelin kabul ettiği uzunluktan fazla olursa, eski frame'leri sil
            if len(sequence) > 60:
                sequence.pop(0)

            # En az 30 frame biriktiyse tahmin yap
            if len(sequence) == 60:
                input_data = np.expand_dims(sequence, axis=0)  # Model için boyutları ayarla
                input_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())  # Normalize et

                prediction = model.predict(input_data)[0]  # Modelden tahmin al
                max_prob = np.max(prediction)  # En yüksek olasılığı al
                predicted_label = actions[np.argmax(prediction)]  # En yüksek olasılıklı sınıf

                # **Threshold (Eşik) Kontrolü**
                if max_prob > THRESHOLD:
                    label = f"{predicted_label} ({max_prob*100:.2f}%)"
                else:
                    label = "Bilinmeyen Hareket"

                # Tahmini ekrana yazdır
                cv2.putText(image, f"Prediction: {label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(label)

            cv2.imshow('Video Classification', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Kullanım
video_path = "User_7_005.mp4"  # Tahmin yapılacak video dosyası
classify_video(video_path)
