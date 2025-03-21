import os
import cv2
import mediapipe as mp
import numpy as np

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

def draw_angle(image, angle, pts2):
    h, w, _ = image.shape
    cv2.putText(image, str(round(angle, 2)), 
                    tuple(np.multiply(pts2[:2], [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
def calculate_angle(pts1, pts2, pts3):

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    pts3 = np.array(pts3)

    # create vectors
    pts2_1 = pts1 - pts2
    pts2_3 = pts3 - pts2

    cosine_angle = np.dot(pts2_1, pts2_3) / (np.linalg.norm(pts2_1)* np.linalg.norm(pts2_3))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle) / 180
    # return np.degrees(angle) 

# def calculate_finger_angles(image, hand_landmarks):
def calculate_finger_angles(image, results):

    joint_list = [
        [8, 7, 6],      # Index finger (Tip - DIP - PIP)
        [12, 11, 10],   # Middle finger (Tip - DIP - PIP)
        [16, 15, 14],   # Ring finger (Tip - DIP - PIP)
        [20, 19, 18],   # Pinky finger (Tip - DIP - PIP)
        [4, 3, 2]       # Thumb (Tip - IP - MCP)
    ]

    hands = [results.right_hand_landmarks, results.left_hand_landmarks]
    finger_angles = []

    for landmarks in hands:
        if landmarks:
            landmark = landmarks.landmark

            for j0, j1, j2 in joint_list:
                pt1 = [landmark[j0].x, landmark[j0].y, landmark[j0].z]
                pt2 = [landmark[j1].x, landmark[j1].y, landmark[j1].z]
                pt3 = [landmark[j2].x, landmark[j2].y, landmark[j2].z]

                angle = calculate_angle(pt1, pt2, pt3)
                finger_angles.append(angle)
                draw_angle(image, angle, pt2)

        else: 
            finger_angles.extend([0] * 5)

    return image, np.array(finger_angles)

def calculate_wrist_rotation(image, results):

    hands = [results.right_hand_landmarks, results.left_hand_landmarks]
    wrist_rotation = []

    for landmarks in hands:
        if landmarks:
            landmark = landmarks.landmark
            
            thumb_base = [landmark[1].x, landmark[1].y, landmark[1].z]
            wrist = [landmark[0].x, landmark[0].y, landmark[0].z]
            pinky_mcp = [landmark[17].x, landmark[17].y, landmark[17].z]

            angle = calculate_angle(thumb_base, wrist, pinky_mcp)
            draw_angle(image, angle, wrist)
            wrist_rotation.append(angle)        
        else:
            wrist_rotation.append(0)

    return image, np.array(wrist_rotation)
         

def calculate_elbow_angles(image, results):
    joint_list = [[12, 14, 16], [11, 13, 15]]
    elbow_angles = []

    if results.pose_landmarks:
        for shoulder_id, elbow_id, wrist_id in joint_list:
            
            shoulder = results.pose_landmarks.landmark[shoulder_id]
            elbow = results.pose_landmarks.landmark[elbow_id]
            wrist = results.pose_landmarks.landmark[wrist_id]

            shoulder_coords = [shoulder.x, shoulder.y, shoulder.z]
            elbow_coords = [elbow.x, elbow.y, elbow.z]
            wrist_coords = [wrist.x, wrist.y, wrist.z]

            angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
            draw_angle(image, angle, elbow_coords)
            elbow_angles.append(angle)
    else:
        elbow_angles.extend([0] *2)  # BU KISMI KONTROL ET
    
    return image, np.array(elbow_angles)

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

def extract_features(image, results):

    keypoints = extract_keypoints(results)
    image, elbow_angles = calculate_elbow_angles(image, results)
    image, wrist_angles = calculate_wrist_rotation(image, results)
    image, finger_angles = calculate_finger_angles(image, results)

    return image, np.concatenate([keypoints, elbow_angles, wrist_angles, finger_angles])

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
                    print('elendi ')

                cap.release()
                cv2.destroyAllWindows()

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
