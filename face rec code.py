import time

import cv2
import numpy as np
import face_recognition
from matplotlib import pyplot as plt
from IPython.display import display
from PIL import Image


def capture():
    print("You have 5 seconds to position yourself")
    j = 5
    for i in range(5):
        print(j)
        time.sleep(1)
        j = j - 1

    print("SMILEEEEE")

    import cv2

    # Open the first webcam device
    cap = cv2.VideoCapture(0)

    # Capture a frame
    ret, frame = cap.read()

    # Release the webcam
    cap.release()

    # Save the frame as an image file
    cv2.imwrite("check.jpg", frame)
    # cv2.imshow(cap)


def test():
    imgUser = face_recognition.load_image_file('user.jpg')
    imgTest = face_recognition.load_image_file('check.jpg')
    faceLoc = face_recognition.face_locations(imgUser)[0]
    encodedOba = face_recognition.face_encodings(imgUser)[0]
    img1 = cv2.rectangle(imgUser, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 8)
    faceLocTest = face_recognition.face_locations(imgTest)[0]
    encodedTest = face_recognition.face_encodings(imgTest)[0]
    img2 = cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 8)
    results = face_recognition.compare_faces([encodedOba], encodedTest)
    faceDis = face_recognition.face_distance([encodedOba], encodedTest)
    print(results, faceDis)
    if results==True:
        print("BOX IS UNLOCKED")
    else:
        print("COULD NOT RECOGNIZE YOUR FACE")
    imgg1 = cv2.putText(img1, f'{results} {faceDis[0]}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    plt.imshow(imgg1)
    plt.xticks([]), plt.yticks([])
    plt.show()

    plt.imshow(img2)
    plt.xticks([]), plt.yticks([])
    plt.show()


capture()
# time.sleep(3)
test()
