import cv2
import mediapipe as mp

# Initialize MediaPipe solutions for hands and face mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
hand_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)
face_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Capture video from the first camera connected
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Process the image and find hands and faces
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    hands_results = hands.process(image)
    face_results = face_mesh.process(image)

    # Draw the hand and face annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                hand_drawing_spec, hand_drawing_spec)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=face_drawing_spec,
                connection_drawing_spec=face_drawing_spec)

    # Display the resulting image
    cv2.imshow('Hand and Face Recognition', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

