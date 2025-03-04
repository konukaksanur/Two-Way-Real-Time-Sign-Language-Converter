import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from main import mp_holistic, mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Load actions and model
DATA_PATH = "keypoints"
actions = np.array(os.listdir(DATA_PATH))
model_path = 'actions.h5'

try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    raise FileNotFoundError(f"Failed to load the model: {e}")

# Initialize variables
sequence = []
sentence = []
threshold = 0.8
target_frame_count = 60  # Desired frame count for the model
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 245, 245)]

def interpolate_sequence(sequence, target_length):
    """
    Adjust the sequence length using interpolation to match the target length.
    """
    current_length = len(sequence)
    if current_length == target_length:
        return sequence
    elif current_length < target_length:
        # Interpolation: Fill with interpolated values
        indices = np.linspace(0, current_length - 1, num=target_length)
        interpolated_sequence = [sequence[int(i)] for i in indices]
        return np.array(interpolated_sequence)
    else:
        # Truncate: Cut down to the target length
        return np.array(sequence[-target_length:])

def prob_viz(res, actions, input_frame, colors):
    """
    Visualize prediction probabilities on the frame.
    """
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 300), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Open video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam. Please check your device.")

# Set Mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Detect mediapipe landmarks
        image, results = mediapipe_detection(frame, holistic)
        if not results.pose_landmarks:  # Ensure detections are valid
            continue

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Update sequence buffer
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Normalize sequence length
        normalized_sequence = interpolate_sequence(sequence, target_frame_count)

        # Perform prediction if the sequence is ready
        if len(normalized_sequence) == target_frame_count:
            res = model.predict(np.expand_dims(normalized_sequence, axis=0))[0]
            print(f"Prediction: {actions[np.argmax(res)]}, Confidence: {res[np.argmax(res)]:.2f}")

            # Update sentence if confidence exceeds threshold
            if res[np.argmax(res)] > threshold:
                if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])

            # Limit sentence length
            sentence = sentence[-5:]

            # Visualize probabilities
            image = prob_viz(res, actions, image, colors)

        # Display predicted sentence
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show output frame
        cv2.imshow('OpenCV Feed', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
