import os
import cv2
import mediapipe as mp
import numpy as np

from feature_extraction import extract_features  # Import the feature extraction function

# Mediapipe models and utilities
mp_holistic = mp.solutions.holistic  # To get hand body and face gesture
mp_drawing = mp.solutions.drawing_utils  # for drawing that features


def mediapipe_detection(image, model): #opencv bgr mediapipe rgb kullanıyo diye olmuş, non-writeable yapıyoruz ki mediapipe image'ı değiştirmesin
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Make image non-writeable for performance
    results = model.process(image)  # Process the image with the Mediapipe model
    image.flags.writeable = True  # Re-enable image writeability
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

def draw_styled_landmarks(image, results):
    # # Draw face connections
    # if results.face_landmarks:
    #     mp_drawing.draw_landmarks( 
    #         image,
    #         results.face_landmarks,
    #         mp.solutions.face_mesh.FACEMESH_TESSELATION,
    #         mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
    #         mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    #     )
    # Draw pose connections
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
    DATA_PATH = "keypoints"  # Keypoint'leri kaydetmek için klasör
    VIDEO_PATH = "data_10"  # Videoların ana klasörü

    SEQUENCE_LENGTH = 60

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for folder_name in os.listdir(VIDEO_PATH):  # Ana klasördeki her alt klasörü kontrol et
            folder_path = os.path.join(VIDEO_PATH, folder_name)
            if not os.path.isdir(folder_path):  # Sadece klasörleri işleme al
                continue

            for video_name in os.listdir(folder_path):  # Her klasördeki videoları kontrol et
                video_path = os.path.join(folder_path, video_name)
                if not video_path.endswith('.mp4'):  # Sadece .mp4 dosyalarını işleme al
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    continue

#
                video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                skip_frames_window = max(int(video_frame_count / SEQUENCE_LENGTH), 1)

                if not video_frame_count < SEQUENCE_LENGTH: 
                    for frame_counter in range(SEQUENCE_LENGTH):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                        success, frame = cap.read()

                        if not success:
                            break                       

                        # Frame işleme
                        image, results = mediapipe_detection(frame, holistic)

                        # Görselleştirme için landmark çizimi
                        draw_styled_landmarks(image, results)

                        image, features = extract_features(image, results)

                        # Her bir frame'in keypoint'ini ayrı dosyaya kaydet
                        save_folder = os.path.join(DATA_PATH, folder_name, video_name.split('.')[0])
                        os.makedirs(save_folder, exist_ok=True)
                        np.save(os.path.join(save_folder, f"frame_{frame_counter}.npy"), features)

                        # Video işleme durumunu göster
                        cv2.putText(image, f"Processing {folder_name}/{video_name} Frame {frame_counter}", (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Processing Video', image)

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
