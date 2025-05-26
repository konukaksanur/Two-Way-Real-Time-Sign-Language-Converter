# 1. New detection variables
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image as PILImage, ImageDraw, ImageFont
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from main import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_features
import pandas as pd


def draw_text_matplotlib(img, text, position=(10, 10), font_size=20):
    # DejaVu Sans fontunun sistem yolunu otomatik bul
    font_path = font_manager.findfont("DejaVu Sans")  # Bu yol kesin çalışır
    img_pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)



# CSV'deki eşleştirme tablosunu yükle
import os

BASE_DIR = os.path.dirname(__file__)

csv_path = os.path.join(BASE_DIR, "class_map.csv")
df = pd.read_csv(csv_path)



id_to_name = dict(zip(df["ClassID"], df["ClassName_tr"]))


sequence = []
sentence = []
threshold = 0.99

DATA_PATH = os.path.join(BASE_DIR, "keypoints_175_40")

df.columns = df.columns.str.strip()                      
df["ClassID"] = df["ClassID"].astype(str).str.zfill(4)  
id_to_name = dict(zip(df["ClassID"], df["ClassName_tr"]))


lock_duration = 20
frame_lock_counter = 0 

folder_ids = [
    d for d in os.listdir(DATA_PATH) 
    if os.path.isdir(os.path.join(DATA_PATH, d))
]
actions = np.array([id_to_name[d] for d in folder_ids])


MODEL_PATH = os.path.join(BASE_DIR, "action2.h5")
model = load_model(MODEL_PATH)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)] * (len(actions) // 3 + 1)
colors = colors[:len(actions)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        image = draw_text_matplotlib(image, ' '.join(sentence), position=(10, 5), font_size=36)


    return output_frame

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)  # Sol-sağ düzeltme

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        image, features = extract_features(image, results)

        sequence.append(features)
        sequence = sequence[-40:]

        if len(sequence) == 40:
            sequence_arr = np.array(sequence)
            # --- Sağ, Sol, Pose için ayır ---
            X_r_keypoint = sequence_arr[:, 162:225]
            X_r_wrist = sequence_arr[:, 229:230]
            X_r_angle = sequence_arr[:, 231:246]
            X_r_interaction = sequence_arr[:, 261:277]
            X_right = np.concatenate((X_r_keypoint, X_r_wrist, X_r_angle, X_r_interaction), axis=-1)

            X_l_keypoint = sequence_arr[:, 99:162]
            X_l_wrist = sequence_arr[:, 230:231]
            X_l_angle = sequence_arr[:, 246:261]
            X_l_interaction = sequence_arr[:, 277:293]
            X_left = np.concatenate((X_l_keypoint, X_l_wrist, X_l_angle, X_l_interaction), axis=-1)

            X_p_keypoint = sequence_arr[:, 0:99]
            X_p_angle = sequence_arr[:, 225:229]
            X_p_distance = sequence_arr[:, 299:307]
            X_p_intra_hand = sequence_arr[:, 293:299]
            X_pose = np.concatenate((X_p_keypoint, X_p_angle, X_p_distance, X_p_intra_hand), axis=-1)

            res = model.predict([
                np.expand_dims(X_right, axis=0),
                np.expand_dims(X_left, axis=0),
                np.expand_dims(X_pose, axis=0)
            ], verbose=0)[0]

            best_action_idx = np.argmax(res)
            best_action_name = actions[best_action_idx]
            best_action_prob = res[best_action_idx]

            # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                # if len(sentence) > 0:
                #     if actions[np.argmax(res)] != sentence[-1]:
                #         sentence.append(actions[np.argmax(res)])
                #         print(f" {actions[np.argmax(res)]}: {res[np.argmax(res)]}")
                #         # sequence.clear()
                # else:
                #     sentence.append(actions[np.argmax(res)])

                if frame_lock_counter == 0:
                    predicted_word = actions[np.argmax(res)]
                    if len(sentence) == 0 or predicted_word != sentence[-1]:
                        sentence.append(predicted_word)
                        print(f"Debounce Prediction: {predicted_word}, {res[np.argmax(res)]}")
                    frame_lock_counter = lock_duration  # Lock'u başlat
                else:
                    frame_lock_counter -= 1  # Lock süresi dolana kadar başka tahmin kabul etme

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            # image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        image = draw_text_matplotlib(image, ' '.join(sentence), position=(10, 5), font_size=36)


        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

