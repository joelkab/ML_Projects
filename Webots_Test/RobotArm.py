import cv2
import mediapipe as mp
import json


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)


output_file = "hand_movement.json"
hand_data = []

# MediaPipe Hands settings
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = hands.process(rgb_frame)

        frame_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Save hand landmark positions to the frame data
                landmarks = [{
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                } for landmark in hand_landmarks.landmark]
                frame_data.append(landmarks)

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

        # Add the frame data to the overall hand data
        hand_data.append(frame_data)

        
        cv2.imshow('Hand Tracking', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


with open(output_file, "w") as f:
    json.dump(hand_data, f)


cap.release()
cv2.destroyAllWindows()

print(f"Hand movement saved to {output_file}")


