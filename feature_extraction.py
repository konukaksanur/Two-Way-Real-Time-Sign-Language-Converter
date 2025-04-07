import numpy as np
import cv2

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

def calculate_finger_angles(image, results):

    joint_list = [
        [4, 3, 2],        # Thumb (TIP - IP - MCP)
        [3, 2, 1],        # Thumb (IP - MCP - CMC)
        [2, 1, 0],        # Thumb (MCP - CMC - WRIST)

        [8, 7, 6],        # Index finger (TIP - DIP - PIP)
        [7, 6, 5],        # Index finger (DIP - PIP - MCP)
        [6, 5, 0],        # Index finger (PIP - MCP - WRIST)

        [12, 11, 10],     # Middle finger (TIP - DIP - PIP)
        [11, 10, 9],      # Middle finger (DIP - PIP - MCP)
        [10, 9, 0],       # Middle finger (PIP - MCP - WRIST)

        [16, 15, 14],     # Ring finger (TIP - DIP - PIP)
        [15, 14, 13],     # Ring finger (DIP - PIP - MCP)
        [14, 13, 0],      # Ring finger (PIP - MCP - WRIST)

        [20, 19, 18],     # Pinky finger (TIP - DIP - PIP)
        [19, 18, 17],     # Pinky finger (DIP - PIP - MCP)
        [18, 17, 0]       # Pinky finger (PIP - MCP - WRIST)
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
            finger_angles.extend([0] * 15)

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
        elbow_angles.extend([0] *2)
    
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
