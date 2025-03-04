import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Define data path and load actions
DATA_PATH = "keypoints"
actions = np.array(os.listdir(DATA_PATH))  # Get class names from subdirectories
label_map = {label: num for num, label in enumerate(actions)}

# Load the trained model
model = load_model('actions.h5')


# Function to load and process the video
def process_video(video_path, max_sequence_length):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Read video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        # Here you would extract keypoints from the frame
        # For example, use Mediapipe or other methods to extract the keypoints (landmarks) of the hand
        # For simplicity, let's assume we have extracted keypoints as a numpy array (this part is pseudocode)
        # frame_keypoints = extract_keypoints_from_frame(frame)

        # You need to replace `frame_keypoints` with the actual keypoints array from Mediapipe
        frame_keypoints = np.random.rand(21, 3)  # Random values as a placeholder for the keypoints

        frames.append(frame_keypoints)

    cap.release()  # Close the video capture

    if len(frames) > 0:
        # Pad the sequence to match max sequence length
        frames = np.pad(frames, ((0, max_sequence_length - len(frames)), (0, 0)), mode='constant')
        frames = (frames - frames.min()) / (frames.max() - frames.min())  # Normalize data
        return np.expand_dims(frames, axis=0)  # Add batch dimension for model input
    return None


# Example usage
video_path = "User_2_002.mp4"  # Replace with the path to your video file
max_sequence_length = 30  # Use the same max sequence length as in training

# Process the video and make a prediction
processed_video = process_video(video_path, max_sequence_length)
if processed_video is not None:
    # Predict the class for the video
    prediction = model.predict(processed_video)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_class_name = actions[predicted_class][0]  # Convert the class index to the class name
    print(f"The predicted class for this video is: {predicted_class_name}")
else:
    print("No frames found or video is too short.")
