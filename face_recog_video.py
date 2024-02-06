# Import necessary libraries
import cv2
import face_recognition
import os
import datetime

# Set constants and directories
KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
ATTENDANCE_DIR = 'attendance'
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'

# Function to generate a color based on the first three characters of the name
def name_to_color(name):
    color = [(ord(c.lower()) - 97) * 8 for c in name[:3]]
    return color

# Create attendance directory if it doesn't exist
if not os.path.exists(ATTENDANCE_DIR):
    os.makedirs(ATTENDANCE_DIR)

# Create unknown faces directory if it doesn't exist
if not os.path.exists(UNKNOWN_FACES_DIR):
    os.makedirs(UNKNOWN_FACES_DIR)

# Load known faces and their respective encodings
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_faces.append(encoding)
            known_names.append(name)
        else:
            print(f"No faces found in the image {filename}.")

# Load OpenCV face detector (Haarcascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print('Starting video stream...')
video = cv2.VideoCapture(0)
# Adjust camera resolution based on your needs
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dictionary to track attendance status for each student
attendance_log = {}

while True:
    ret, frame = video.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use OpenCV face detector for face detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract face encodings using face_recognition library
    face_encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x) for (x, y, w, h) in faces])

    face_names = []
    for face_encoding in face_encodings:
        # Initialize name as Unknown for each face
        name = "Unknown"

        # Perform face recognition with dynamic tolerance based on the number of faces detected
        tolerance = min(0.6, 0.6 + 0.1 * len(faces))
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        best_match_index = matches.index(True) if any(matches) else None

        # Assign the name if a match is found
        if best_match_index is not None:
            name = known_names[best_match_index]

        face_names.append(name)

    for (x, y, w, h), name in zip(faces, face_names):
        # Draw rectangles around faces and add names with timestamps
        cv2.rectangle(frame, (x, y), (x + w, y + h), name_to_color(name), FRAME_THICKNESS)
        cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), name_to_color(name), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        text = f"{name} - {timestamp}"
        cv2.putText(frame, text, (x + 6, y + h - 6 - 20 * face_names.index(name)), font, 0.5, (255, 255, 255), FONT_THICKNESS)

        # Save attendance record to a file if not already logged
        if name not in attendance_log:
            attendance_log[name] = True
            class_folder = f"{ATTENDANCE_DIR}/{name}"
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)
            with open(f"{class_folder}/attendance.txt", "a") as f:
                f.write(f"{timestamp} - {name} - Present\n")

        # Save full face region for unknown faces with timestamp
        if name == "Unknown":
            unknown_folder = f"{UNKNOWN_FACES_DIR}/{timestamp}"
            if not os.path.exists(unknown_folder):
                os.makedirs(unknown_folder)

                unknown_filename = f"{unknown_folder}/unknown_face.jpg"
                cv2.imwrite(unknown_filename, frame[y:y+h, x:x+w])

    # Display the processed frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video.release()
cv2.destroyAllWindows()
