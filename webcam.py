import pickle
import face_recognition
import cv2

# Load the face encodings and names from "face_enc" file created by training.py
with open("face_enc", "rb") as file:
    data = pickle.loads(file.read())
    kEncodings = data["encodings"]
    kNames = data["names"]

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/10 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Loop over the faces
    for (top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 10
        right *= 10
        bottom *= 10
        left *= 10

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Check if the face encoding matches any known faces
        matches = face_recognition.compare_faces(kEncodings, face_encoding)

        name = "Unknown"

            # Resize frame of video to 1/4 size for faster face detection processing
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        else:
            print("frame is empty")


        # If a match is found, get the index of the matched face and use it to get the name
        if True in matches:
            match_index = matches.index(True)
            name = kNames[match_index]

        # Put the name of the person under the rectangle
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
