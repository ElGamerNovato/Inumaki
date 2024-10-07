import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Use the correct landmark connections
mp_face_mesh_connections = mp_face_mesh.FACEMESH_TESSELATION

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw all landmarks for debugging
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh_connections)

            # Get the coordinates for the lips
            upper_lip_indices = [61, 62, 63, 64, 65, 66, 67]
            lower_lip_indices = [78, 79, 80, 81, 82, 13, 312]

            # Extract the upper and lower lip landmarks
            upper_lip_landmarks = [face_landmarks.landmark[i] for i in upper_lip_indices]
            lower_lip_landmarks = [face_landmarks.landmark[i] for i in lower_lip_indices]

            # Convert normalized coordinates to pixel values
            upper_lip_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in upper_lip_landmarks]
            lower_lip_points = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in lower_lip_landmarks]

            # Draw the lip landmarks
            for point in upper_lip_points + lower_lip_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Lip Movement Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

