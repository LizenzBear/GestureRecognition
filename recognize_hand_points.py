import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Capture video from the first camera connected
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image)

    # Convert the image color back so it can be displayed
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting image
    cv2.imshow('Hand Gesture Recognition', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

