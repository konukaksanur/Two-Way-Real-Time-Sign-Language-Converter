import os
import cv2
import mediapipe as mp
import numpy as np

# Mediapipe models and utilities
mp_holistic = mp.solutions.holistic  # To get hand body and face gesture
mp_drawing = mp.solutions.drawing_utils  # for drawing that features

def calculate_angle(pts1, pts2, pts3):

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    pts3 = np.array(pts3)

    # create vectors
    pts2_1 = pts1 - pts2
    pts2_3 = pts3 - pts2

    cosine_angle = np.dot(pts2_1, pts2_3) / (np.linalg.norm(pts2_1)* np.linalg.norm(pts2_3))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle) / 360

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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    # face = np.array([[res.x, res.y, res.z] for res in
                    # results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in
                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                    results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    # return np.concatenate([pose,face,lh,rh])
    return np.concatenate([pose, lh,rh])

def calculate_wrist_rotation(results):
    angle_wrist_right, angle_wrist_left = 0, 0  # Varsayılan olarak sıfırla başlat

    if results.right_hand_landmarks:  # Eğer sağ el landmark'ları varsa
        landmarks = results.right_hand_landmarks.landmark

        wrist_right = [landmarks[mp_holistic.HandLandmark.WRIST.value].x,
                       landmarks[mp_holistic.HandLandmark.WRIST.value].y,
                       landmarks[mp_holistic.HandLandmark.WRIST.value].z]

        middle_finger_right = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x,
                               landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y,
                               landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].z]

        index_finger_right = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x,
                              landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y,
                              landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].z]

        angle_wrist_right = calculate_angle(wrist_right, index_finger_right, middle_finger_right)

    if results.left_hand_landmarks:  # Eğer sol el landmark'ları varsa
        landmarks = results.left_hand_landmarks.landmark

        wrist_left = [landmarks[mp_holistic.HandLandmark.WRIST.value].x,
                      landmarks[mp_holistic.HandLandmark.WRIST.value].y,
                      landmarks[mp_holistic.HandLandmark.WRIST.value].z]

        middle_finger_left = [landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].x,
                              landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].y,
                              landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP.value].z]

        index_finger_left = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x,
                             landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y,
                             landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].z]

        angle_wrist_left = calculate_angle(wrist_left, index_finger_left, middle_finger_left)

    return angle_wrist_right, angle_wrist_left

         

def extract_angles(landmarks):
    if (mp_holistic.PoseLandmark.RIGHT_SHOULDER.value in range(len(landmarks)) and
        mp_holistic.PoseLandmark.RIGHT_ELBOW.value in range(len(landmarks)) and
        mp_holistic.PoseLandmark.RIGHT_WRIST.value in range(len(landmarks))):

        shoulder_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y,
                        landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].z]

        elbow_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y,
                    landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].z]

        wrist_right = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y,
                    landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].z]

        angle_right_arm = calculate_angle(shoulder_right, elbow_right, wrist_right)

    else:
        angle_left_arm = 0
         
    if (mp_holistic.PoseLandmark.LEFT_SHOULDER.value in range(len(landmarks)) and
        mp_holistic.PoseLandmark.LEFT_ELBOW.value in range(len(landmarks)) and
        mp_holistic.PoseLandmark.LEFT_WRIST.value in range(len(landmarks))):

        shoulder_left = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y,
                            landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].z]

        elbow_left = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y,
                        landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].z]

        wrist_left = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y,
                        landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].z]

        angle_left_arm = calculate_angle(shoulder_left, elbow_left, wrist_left)

    else:
        angle_left_arm = 0

    return angle_right_arm, angle_left_arm


def process_videos():
    DATA_PATH = "keypoints"  # Keypoint'leri kaydetmek için klasör
    VIDEO_PATH = "data"  # Videoların ana klasörü

    SEQUENCE_LENGTH = 40

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
                        
                        #normalized_frame = frame/255

                        # Frame işleme
                        image, results = mediapipe_detection(frame, holistic)

                        # Görselleştirme için landmark çizimi
                        draw_styled_landmarks(image, results)

                        # Keypoint çıkarma
                        keypoints = extract_keypoints(results)

                        angles = []
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            angles = list(extract_angles(landmarks))
                        else: 
                            angles = [0, 0] 

                        angles += list(calculate_wrist_rotation(results)) 


                        combined_data = np.concatenate([keypoints, np.array(angles)])


                        # Her bir frame'in keypoint'ini ayrı dosyaya kaydet
                        save_folder = os.path.join(DATA_PATH, folder_name, video_name.split('.')[0])
                        os.makedirs(save_folder, exist_ok=True)
                        np.save(os.path.join(save_folder, f"frame_{frame_counter}.npy"), combined_data)

                        # Video işleme durumunu göster
                        cv2.putText(image, f"Processing {folder_name}/{video_name} Frame {frame_counter}", (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow('Processing Video', image)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                else:
                    print('elendi ')


                cap.release()
                cv2.destroyAllWindows()
                                    # resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                    # normalized_frame = resized_frame/255

                # cap.release()
                # return frame_list
            

            # if len(frames) == SEQUENCE_LENGTH:
            #     features.append(frames)
            #     labels.append(class_index)
            #     video_files_paths.append(video_files_path)

            # features = np.asarray(features)
            # labels = np.array(labels)

            # return features, labels, video_files_paths



def video_capture():
    # Video capture
    cap = cv2.VideoCapture(0)  # access camera
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()  # read from camera
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            # Process the frame
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Draw landmarks on the frame
            draw_styled_landmarks(image, results)

            # Display the output
            cv2.imshow('Bitto Deneme', image)  # show frame

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
def main():
    process_videos()

if __name__ == "__main__":
    main()
