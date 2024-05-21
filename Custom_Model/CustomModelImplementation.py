import mediapipe as mp
import cv2
import pyautogui
import time

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
    print(gestures[0].index)
    # print(result.gestures[0])

    if "click" in gestures:  
        pyautogui.click()  
        time.sleep(2)  

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="C:/Users/Moritz/OneDrive/Dokumente/KI/GestureRecognition/Custom_Model/rock_paper_scissors.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, timestamp)

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

video.release()
cv2.destroyAllWindows()
