# Gesture Recognition

# Setup

```bash
git clone https://github.com/LizenzBear/GestureRecognition/tree/main
python3 -m pip install -r requirements.txt
python3 {OS}_CustomModelImplementation.py # With OS being MacOS (= MacOS & Linux) or Windows
```

## Mediapipe

Mediapipe ist eine Open-Source-Bibliothek von Google, die Werkzeuge für die Implementierung von Multimedia- und maschinellem Lernen-Pipelines bereitstellt. Sie wird häufig für die Echtzeit-Verarbeitung von Videostreams verwendet, um Aufgaben wie Gesichtserkennung, Handtracking und Körperpose-Tracking zu ermöglichen. Durch vorgefertigte Lösungen und optimierte Performance unterstützt Mediapipe Entwickler bei der schnellen Umsetzung komplexer Computer Vision-Anwendungen.

### Anwendung

Mediapipe wird von uns verwendet um ein digitales Skelett unserer Hand aus dem Live-Video der Webcam herauszubekommen. (Siehe Abbildung) 

![Untitled](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/f8ba831a-0306-41ef-8311-7c0a48686d22.png)

Durch Mediapipe kann man zusätzlich auch die Koordinaten jeder Punkte (Siehe Abbildung rote Punkte) herauslesen, was uns das trainieren leichter macht, da wir nicht auf Grundlage von Bildern sondern auf Grundlage von Koordinaten trainieren können. 

### Andere Anwendungen (Für Projekt irrelevant)

Mediapipe ist nicht nur für das Erkennen einer Hand geeignet, sondern kann auch den ganzen Körper erkennen oder detailliert das Gesicht. (Siehe Abbildungen)

![Untitled](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/6aac48be-3f11-4005-a37b-47cf958309fc.png)

![Untitled](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/c7d00044-60c6-40b9-b64d-bf5b666c5827.png)

**Anwendungsbeispiele:**

Die Gesichtserkennung lässt sich gut dazu verwenden um Gesichtszüge und somit den Ausdruck von Emotionen zu erkennen da man nur mit einem Bild Unmengen von Daten erhält.

Die Erkennung vom ganzen Körper ist zwar an den Extrempunkten (Hände und Gesicht) zwar wesentlich ungenauer, allerdings kann man diese Daten gut verwenden um beispielsweise ein Modell zu trainieren welches dabei hilft Sportübungen (Bsp. Liegestütz) korrekt auszuführen ohne einen persönlichen Trainer zu haben. 

## Trainingsdaten

Eines der wichtigsten Themen bei einem Modell welches durch “Supervised-Training” trainiert wird sind Trainingsdaten. Da es im Internet allerdings keine Daten gibt, welche passend für unser Modell sind haben wir uns kurzerhand entschieden eigene Trainingsdaten aufzunehmen. Dafür haben wir ein kleines Python-Script geschrieben: 

```python
import cv2
import os

def take_photos(directory, start_number):
    # Zugriff auf die Webcam
    cam = cv2.VideoCapture(0)

    # Überprüfen, ob die Kamera verfügbar ist
    if not cam.isOpened():
        print("Kann nicht auf die Kamera zugreifen")
        return

    photo_count = start_number
    while True:
        # Ein Foto aufnehmen
        ret, frame = cam.read()

        # Überprüfen, ob das Foto erfolgreich aufgenommen wurde
        if not ret:
            print("Kann kein Foto aufnehmen")
            continue

        # Das aktuelle Bild anzeigen
        cv2.imshow('Webcam', frame)

        # Auf Tastendruck warten
        key = cv2.waitKey(1) & 0xFF

        # Wenn die Leertaste gedrückt wird, das Foto speichern
        if key == ord(' '):
            # Den Pfad für das Foto erstellen
            photo_path = os.path.join(directory, f"{photo_count}.jpg")

            # Das Foto speichern
            cv2.imwrite(photo_path, frame)

            # Den Zähler erhöhen
            photo_count += 1

        # Wenn die Taste "Q" gedrückt wird, das Programm beenden
        elif key == ord('q'):
            break

    # Die Kamera freigeben
    cam.release()

    # Alle Fenster schließen
    cv2.destroyAllWindows()

# Verzeichnis und Fotonummer festlegen
directory = "clickData"
start_number = 1
```

Dieses Script macht nichts anders als bei jedem Anschlag auf der Leertaste ein Foto mit der Webcam zu machen um diese dann nummeriert in das vorgegebene Verzeichnis zu speichern. (Siehe Abbildung)

![Untitled](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/f764bb3b-c59a-4c7b-bed8-a34669492e91.png)

Durch diese strukturierten Daten lässt sich das Modell dann gut trainieren. Insgesamt haben wir uns so jeweils für “click” und “drag” ungefähr 200 Bilder aufgenommen. Das hat soweit gereicht um eine sehr hohe Genauigkeit zu erreichen. Bräuchte man allerdings mehr kann man entweder mehr aufnehmen oder allerdings “Data Augmentation” anwenden.

**Data Augmentation:** 

Data Augmentation ist ein Verfahren in welchem Bilder “manipuliert” werden mit entweder:

- Rotation
- Spiegelung
- Skalierung
- Translation (Verschieben von Bildern in X oder Y Richtung)
- Helligkeit und Kontrastanpassung
- künstliches Rauschen hinzufügen
- Zufällige Ausschnitte
- Farbkanäle manipulieren

Für unser Projekt würden 7 von 8 Varianten in Frage kommen, da wir nicht zufällige Ausschnitte verwenden können, da immer die ganze Hand auf dem Bild sein sollte. 

## Trainieren des Modells

(Training.ipynb)

Mit den gewonnen Trainingsdaten wird jetzt unser eigenes Model trainiert:

**Importieren der benötigten Libraries**

```python
import cv2
# Install with: python3 -m pip install 'keras<3.0.0' mediapipe-model-maker
from mediapipe_model_maker import gesture_recognizer
```

**Daten Laden und in Test- und Trainingsdaten aufteilen**

```python
# Load the rock-paper-scissor image archive.
IMAGES_PATH = "rps_data_sample"
data = gesture_recognizer.Dataset.from_folder(
    dirname=IMAGES_PATH,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)

# Split the archive into training, validation and test dataset.
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
```

**Modell bauen und trainieren**

```python
# Train the model
hparams = gesture_recognizer.HParams(export_dir="rock_paper_scissors_model", epochs=200)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)
```

![Untitled](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/Untitled.png)

![Untitled](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/Untitled%201.png)

**Modell evaluieren**

```python
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")
```

![Untitled](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/Untitled%202.png)

Nach diesem Durchgang hat das Modell also eine Genauigkeit von 96,43%. 

**Modell Speichern**

Um nicht jedes mal von vorne anzufangen speichert man das Modell als .task File ab, auf dieses kann man dann in anderen Programmen zugreifen.

```python
# Export the model bundle.
model.export_model()

# Rename the file to be more descriptive. (Works only in Jupyter)
!mv rock_paper_scissors_model/gesture_recognizer.task rock_paper_scissors.task
```

## Anwendung des Modells

### MacOS

```python
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

click_cooldown = 0.5
last_click_time = 0

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global last_click_time
    gestures = result.gestures
    for gesture in gestures:
        gesture_name = [category.category_name for category in gesture]
        if gesture_name[0] == "click":
            current_time = time.time()
            if current_time - last_click_time >= click_cooldown:
                pyautogui.click()
                last_click_time = current_time

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

```

Wenn man diesen Code allerdings unter Windows laufen lässt bekommt man den Fehler, dass der “model_asset_path” nicht gefunden werden kann - warum ist unerklärlich. 

### Windows

Um diesen Fehler zu umgehen, benutzt man also folgenden Code: 

```python
import mediapipe as mp
import cv2
import pyautogui
import time
import numpy as np

pyautogui.FAILSAFE = False

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

# Festlegen der Fensterposition
cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
cv2.moveWindow("Video Feed", 0, 0)

click_cooldown = 0.5
last_click_time = 0

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global last_click_time
    gestures = result.gestures
    for gesture in gestures:
        gesture_name = [category.category_name for category in gesture]
        if gesture_name[0] == "click":
            current_time = time.time()
            if current_time - last_click_time >= click_cooldown:
                pyautogui.click()
                last_click_time = current_time

f = open("./rock_paper_scissors.task", mode="rb")
 
# Reading file data with read() method
data = f.read()
# Closing the opened file
f.close()

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=data),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

timestamp = 0
frame_reduction = 100  # Frame reduction to manage sensitivity
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

```

Diese Version des Codes verwendet: 

```python
f = open("./rock_paper_scissors.task", mode="rb")
# Reading file data with read() method
data = f.read()
# Closing the opened file
f.close()
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=data),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)
```

Dieser Abschnitt macht nichts anders als den Pfad zu öffnen und die Bytes zu lesen, um diese dann in die Option “model_asset_buffer” statt “model_asset_path” zu geben. 

## Anmerkungen

Um die Funktion des Codes vollständig zu garantieren sollte man so gut wie möglich die 2 trainierten Gesten benutzen und versuchen eine Geste dazwischen zu vermeiden. 

Folgende 2 Gesten: 

![Click](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/0d65e3de-fba9-4fb4-b168-9daa61ea893d.png)

Click

![Drag](Gesture%20Recognition%20e17c42256bf147f1b1f9ded31386502a/c23e761b-8ce0-4524-86e7-02717fbc8416.png)

Drag