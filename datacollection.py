import cv2
import os

id = input("enter the name as id: ")
print("Wait for few minutes.....")
# Create a folder
if not os.path.exists("collected_data"):
    os.mkdir("collected_data")
parent_dir = "collected_data"

# name of the subdirectory
subdir_name = id

#subdirectory by joining the parent directory path and subdirectory name
subdir_path = os.path.join(parent_dir, subdir_name)

# Checking if the subdirectory already exists
if not os.path.exists(subdir_path):
    # Create the subdirectory
    os.mkdir(subdir_path)

#  webcam
cap = cv2.VideoCapture(0)

#counter for the number of faces collected
counter = 1
while True:
    # frame from the webcam
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier('C:/Users/sethy/PycharmProjects/faceRecognition/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw a rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow("Webcam", frame)

    # collect 20  samples for data set
    key = cv2.waitKey(1) & 0xFF
    if counter<=20:
        # If any face is detected, crop and save it.
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (128, 128))
            cv2.imwrite(subdir_path+'/'+str(id)+"."+"{}.jpg".format(counter), face_roi)
            print("Faces saved!{}".format(counter))
            counter += 1
            
    # Exit the loop if the 'q' key is pressed
    if key == ord('q'):
        break
# Release the webcam and close the window
cap.release()

print("DATA COLLECTION done .... ")
print("MOVE FORWARD--------------->>>>>>>Pre-processing")
cv2.destroyAllWindows()
