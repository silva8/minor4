import cv2
import sys
import requests
import os
from IPython.display import HTML
from os.path import expanduser

subscription_key = "48715137ab314838989ae3066bb3dde1"
assert subscription_key
face_api_url = 'https://westus.api.cognitive.microsoft.com/face/v1.0/detect'
cwd = os.path.realpath(__file__)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if len(faces) != 0:
        cv2.imwrite("faces/frame%d.jpg" % count, frame)     # save frame as JPEG file
        img = open(expanduser("faces/frame%d.jpg" % count), 'rb')
        headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key
        }
        params = {
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
            'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
        }
        response = requests.post(
            face_api_url, params=params, headers=headers, data= img)
        faces = response.json()
        print(faces)
        count+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
