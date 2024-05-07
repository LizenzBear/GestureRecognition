import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Use the default webcam

screen_width, screen_height = pyautogui.size()  # Get the size of the screen
frame_reduction = 100  # Frame reduction to manage sensitivity

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the tip of the index finger and the tip of the thumb
            # Use the wrist landmark for tracking the cursor
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            x, y = int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0])

            # Convert coordinates to screen size
            screen_x = np.interp(x, (frame_reduction, image.shape[1] - frame_reduction), (0, screen_width))
            screen_y = np.interp(y, (frame_reduction, image.shape[0] - frame_reduction), (0, screen_height))

            # Move the mouse cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Calculate the distance between index finger tip and thumb tip
            distance = calculate_distance(index_tip, thumb_tip)

            # Click if the distance is small enough
            if distance < 0.05:  # Adjust the threshold based on testing
                pyautogui.click()

            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
