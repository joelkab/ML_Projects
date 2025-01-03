import cv2
import mediapipe as mp
import json
import numpy as np 


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load the hand movement 
input_file = "hand_movement.json"
with open(input_file, "r") as f:
    hand_data = json.load(f)


canvas_height, canvas_width = 720, 1280
canvas = None

# Replay the movement
for frame_data in hand_data:
    # Create a blank canvas for each frame
    canvas = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)

    for hand_landmarks in frame_data:
        # Convert normalized coordinates to pixel coordinates
        landmarks = []
        for landmark in hand_landmarks:
            x = int(landmark["x"] * canvas_width)
            y = int(landmark["y"] * canvas_height)
            landmarks.append((x, y))

        # Draw the hand 
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(canvas, start, end, (0, 0, 255), thickness=2)

        # Draw the landmarks
        for landmark in landmarks:
            cv2.circle(canvas, landmark, 5, (0, 255, 0), thickness=-1)

    
    cv2.imshow("Hand Movement Replay", canvas)

    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

