import cv2
import datetime
import pygame
pygame.mixer.init()
sound_path = 'C:\\Users\\ADMIN\\Desktop\\Automatic_selfie capture\\selfie_star-main\\camera_click.wav'
camera_capture_sound = pygame.mixer.Sound(sound_path)

# Initialize the camera (use camera index 0)
cap = cv2.VideoCapture(0)

# Load Haar cascades for face and smile detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

capture_enabled = False  # Flag to indicate whether to capture an image

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is successfully read
    if not ret or frame is None:
        print("Error: Unable to read frame from the camera.")
        break

    # Convert the frame to grayscale for face and smile detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for x, y, w, h in faces:
        # Extract the region of interest (ROI) for face and convert to grayscale
        face_roi = frame[y:y + h, x:x + w]
        gray_roi = gray[y:y + h, x:x + w]

        # Detect smiles in the face ROI
        smiles = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)

        # Iterate over detected smiles
        for x1, y1, w1, h1 in smiles:
            # If capture is enabled, play camera capture sound, capture a selfie, and disable capture
            if not capture_enabled:
                camera_capture_sound.play()
                time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                file_name = f'selfie-{time_stamp}.png'
                cv2.imwrite(file_name, frame)
                capture_enabled = True

    # Display the frame
    cv2.imshow('cam star', frame)

    # Wait for a key event (press 'q' to exit)
    if cv2.waitKey(10) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
