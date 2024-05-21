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

# Funktion aufrufen
take_photos(directory, start_number)
