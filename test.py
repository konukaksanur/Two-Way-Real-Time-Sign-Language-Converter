# 1. New detection variables
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from main import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints, extract_angles, calculate_wrist_rotation

sequence = []
sentence = []
threshold = 0.9

DATA_PATH = "keypoints"

actions = np.array([
    d for d in os.listdir(DATA_PATH) 
    if os.path.isdir(os.path.join(DATA_PATH, d))
])
model = load_model('actions.h5')
# /Users/aksanurkonuk/Programming_Projects/Two-Way-Real-Time-Sign-Language-Converter/data/ad/User_2_001.mp4
# /Users/aksanurkonuk/Programming_Projects/Two-Way-Real-Time-Sign-Language-Converter/data/anne/User_2_001.mp4
# /Users/aksanurkonuk/Programming_Projects/Two-Way-Real-Time-Sign-Language-Converter/data/birSeyDegil/User_2_001.mp4
# /Users/aksanurkonuk/Programming_Projects/Two-Way-Real-Time-Sign-Language-Converter/data/tesekkur/User_5_003.mp4
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)] * (len(actions) // 3 + 1)
colors = colors[:len(actions)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        #frame = cv2.flip(frame, 1)  # Sol-sağ düzeltme

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:40]

        angles = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = list(extract_angles(landmarks))
        else: 
            angles = [0, 0] 

        angles += list(calculate_wrist_rotation(results)) 


        combined_data = np.concatenate([keypoints, np.array(angles)])



        sequence.append(combined_data)
        sequence = sequence[-40:]

        if len(sequence) == 40:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                        # sequence.clear()
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
