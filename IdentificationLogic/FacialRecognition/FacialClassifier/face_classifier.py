''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2
import numpy as np
import os


def scan_faces(users_names: dict):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if os.path.isfile('Trainer/trainer.yml'):
        print("loading trained params for scanning faces")
        recognizer.read('Trainer/trainer.yml')
    else:
        print("shouldn't get here!!!")
        exit()

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        rectangle_color = (0, 255, 0)  # green in RGB

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), rectangle_color, 2)

            face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less then 100 ==> "0" is perfect match
            if confidence < 100 and (face_id == 1 or face_id == 0):
                face_name = users_names[face_id]
                confidence = "  {0}%".format(round(100 - confidence))
                flag_red_box = False
            else:
                face_name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                flag_red_box = True

            cv2.putText(img, str(face_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            if flag_red_box:        # isn't working but could be nice
                rectangle_color = (0, 0, 255)
            else:
                rectangle_color = (0, 255, 0)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
