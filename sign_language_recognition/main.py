import os
import cv2
import mediapipe as mp
import numpy as np

from .feature_extraction import extract_features

# Mediapipe models and utilities
mp_holistic = mp.solutions.holistic  # Detect pose, hand, and face landmarks
mp_drawing = mp.solutions.drawing_utils   # Draw detected landmarks


def mediapipe_detection(image, model): 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Make image non-writeable for performance
    results = model.process(image)  # Process the image with the Mediapipe model
    image.flags.writeable = True  # Re-enable image writeability
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

def process_videos():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "keypoints_175_40")

    VIDEO_PATH = os.path.join(BASE_DIR, "data")

    SEQUENCE_LENGTH = 40

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for folder_name in os.listdir(VIDEO_PATH):  
            folder_path = os.path.join(VIDEO_PATH, folder_name)
            if not os.path.isdir(folder_path):  
                continue

            for video_name in os.listdir(folder_path): 
                video_path = os.path.join(folder_path, video_name)
                if not video_path.endswith('.mp4'):
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    continue


                video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                skip_frames_window = max(int(video_frame_count / SEQUENCE_LENGTH), 1)

                if not video_frame_count < SEQUENCE_LENGTH: 
                    for frame_counter in range(SEQUENCE_LENGTH):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                        success, frame = cap.read()

                        if not success:
                            break                       

                        image, results = mediapipe_detection(frame, holistic)
                        print(f"Processing {folder_name}/{video_name} Frame {frame_counter}")

                        image, features = extract_features(image, results)

                        save_folder = os.path.join(DATA_PATH, folder_name, video_name.split('.')[0])
                        os.makedirs(save_folder, exist_ok=True)
                        np.save(os.path.join(save_folder, f"frame_{frame_counter}.npy"), features)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                else:
                    print('Eliminated video: ', video_name)

                cap.release()
                cv2.destroyAllWindows()

def main():
    process_videos()

if __name__ == "__main__":
    main()