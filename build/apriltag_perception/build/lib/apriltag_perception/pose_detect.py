import cv2
import mediapipe as mp
import socket
import json
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ROS_BRIDGE_IP = "172.29.165.220"
ROS_BRIDGE_PORT = 5006  # different port from apriltag

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract the 3 arm keypoints we care about
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow    = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist    = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        payload = json.dumps({
            "shoulder": {"x": round(shoulder.x, 4), "y": round(shoulder.y, 4), "z": round(shoulder.z, 4)},
            "elbow":    {"x": round(elbow.x, 4),    "y": round(elbow.y, 4),    "z": round(elbow.z, 4)},
            "wrist":    {"x": round(wrist.x, 4),    "y": round(wrist.y, 4),    "z": round(wrist.z, 4)},
        })

        sock.sendto(payload.encode(), (ROS_BRIDGE_IP, ROS_BRIDGE_PORT))
        print(f"[SENT] Shoulder:{shoulder.x:.3f},{shoulder.y:.3f} "
              f"Elbow:{elbow.x:.3f},{elbow.y:.3f} "
              f"Wrist:{wrist.x:.3f},{wrist.y:.3f}")

        # Draw the skeleton on screen
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
