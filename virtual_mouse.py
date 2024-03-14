import cv2
import mediapipe as mp
import pyautogui

# Initialisieren von MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Bildschirmauflösung herausfinden für die Maussteuerung
screen_width, screen_height = pyautogui.size()

# Kameracapture starten
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoriere leeres Kamerabild.")
        continue

    # Bild für die Verarbeitung vorbereiten
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Ergebnis zeichnen
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Hier könnten Sie zusätzliche Logik implementieren, um spezifische Gesten zu erkennen
            # Beispiel: Erkennung, ob die Hand geschlossen ist, um einen Klick zu simulieren
            
            # Mausbewegung basierend auf der Position des Handgelenks (Landmark 0)
            wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            # Umwandlung in Bildschirmkoordinaten
            x = int(wrist_landmark.x * screen_width)
            y = int(wrist_landmark.y * screen_height)
            pyautogui.moveTo(x, y)

            # Zeichnen der Handlandmarks im Bild
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Bild anzeigen
    cv2.imshow('MediaPipe Hands', image)
    
    # Beenden mit ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

