import mediapipe as mp
import cv2
import pyautogui
import time
import numpy as np

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

# Festlegen der Fensterposition
cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("Video Feed", 0, 0)

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    gestures = result.gestures
    for gesture in gestures:
        print(gesture)
        x = [category.category_name for category in gesture]
        if x[0] == "click":
            pyautogui.click()
            time.sleep(2)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="./rock_paper_scissors.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

timestamp = 0
frame_reduction = 200  # Frame reduction to manage sensitivity
screen_width, screen_height = pyautogui.size()  # Get the size of the screen
with GestureRecognizer.create_from_options(options) as recognizer:
    try:
        while video.isOpened():
            ret, frame = video.read()
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(frame)

            if not ret:
                print("Ignoring empty frame")
                break

            if results:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        x, y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])

                        # Convert coordinates to screen size
                        screen_x = np.interp(x, (frame_reduction, frame.shape[1] - frame_reduction), (0, screen_width))
                        screen_y = np.interp(y, (frame_reduction, frame.shape[0] - frame_reduction), (0, screen_height))

                        # Move the mouse cursor
                        pyautogui.moveTo(screen_x, screen_y)

            timestamp += 1
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            recognizer.recognize_async(mp_image, timestamp)

            cv2.imshow("Video Feed", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        video.release()
        cv2.destroyAllWindows()
