import face_recognition

# Load the jpg files into numpy arrays
Yuepeng_image = face_recognition.load_image_file("./known_people/Yuepeng.JPG")
Louis_image = face_recognition.load_image_file("./known_people/Louis.JPG")
unknown_image = face_recognition.load_image_file("./known_people/Yuepeng2.jpg")
unknown_image2 = face_recognition.load_image_file("./known_people/Yuepeng3.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    Yuepeng_face_encoding = face_recognition.face_encodings(Yuepeng_image)[0]
    Louis_face_encoding = face_recognition.face_encodings(Louis_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    unknown_face_encoding2 = face_recognition.face_encodings(unknown_image2)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    Yuepeng_face_encoding,
    Louis_face_encoding,
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Yupeng? {}".format(results[0]))
print("Is the unknown face a picture of Louis? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))

results = face_recognition.compare_faces(known_faces, unknown_face_encoding2)
print("Is the unknown face a picture of Yupeng? {}".format(results[0]))
print("Is the unknown face a picture of Louis? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))

