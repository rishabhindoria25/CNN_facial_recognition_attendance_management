# CNN_facial_recognition_attendance_management

## Overview

This project is a Facial Recognition Attendance Management System implemented in Python using Convolutional Neural Networks (CNN). The system utilizes the OpenCV and face_recognition libraries to detect and recognize faces in a real-time video stream. The primary goal is to automate the attendance tracking process by associating recognized faces with student names and logging their attendance status along with timestamps.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- face_recognition
- Haarcascades for face detection

## Project Structure

The project is organized into several components:

- **Directories:**
  - `known_faces`: Contains subdirectories for each known individual, each containing images for facial recognition training.
  - `unknown_faces`: Stores images of unknown faces with timestamps for record-keeping.
  - `attendance`: Captures attendance records for recognized individuals.

- **Constants:**
  - `KNOWN_FACES_DIR`: Directory for known faces.
  - `UNKNOWN_FACES_DIR`: Directory for unknown faces.
  - `ATTENDANCE_DIR`: Directory for storing attendance records.
  - `FRAME_THICKNESS`: Thickness of rectangles drawn around faces.
  - `FONT_THICKNESS`: Thickness of font used for displaying names and timestamps.
  - `MODEL`: The face recognition model used (set to 'cnn').

## Implementation

### 1. Loading Known Faces and Encodings

The script loads known faces from the `known_faces` directory, extracts facial encodings using the `face_recognition` library, and stores them for recognition.

### 2. Face Detection

The OpenCV Haarcascades face detector is employed to detect faces in the video stream. Detected faces are then passed to the `face_recognition` library to obtain face encodings.

### 3. Face Recognition

The script uses face recognition to compare the encodings of detected faces with the known faces' encodings. The recognition process is performed with a dynamic tolerance based on the number of faces detected, improving accuracy.

### 4. Attendance Logging

The system logs attendance by creating subdirectories for each recognized face in the `attendance` directory. Attendance records are stored in `attendance.txt` files with timestamps and attendance status.

### 5. Unknown Faces Handling

Images of unknown faces are saved in the `unknown_faces` directory for record-keeping, allowing for later analysis.

## Usage

1. Organize known faces in the `known_faces` directory.
2. Run the script to start the video stream and face recognition.
3. Press 'q' to exit the video stream and close the application.

## Notes

- Ensure the presence of Haarcascades XML file for face detection.
- Adjust camera resolution using the `video.set` statements based on your requirements.

## Conclusion

This Facial Recognition Attendance Management System provides a robust solution for automating attendance tracking using CNN-based face recognition.

## Author

Rishabh Indoria

## License

This project is licensed under the MIT License.
