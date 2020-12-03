''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2
import numpy as np
from PIL import Image
import os


class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.isfile('Trainer/trainer.yml'):
            print("reloading recognizer from trained params - from previous run")
            self.recognizer.read('Trainer/trainer.yml')
        else:
            print("initializing new training params")

    def _get_images_and_labels(self, path: str, user_id: int):
        """
        function to get the images and label data
        :param path: the path to the dataset
        :param user_id:
        :return: a list of faces labeled with the user's face_id
        """

        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for imagePath in image_paths:

            pil_image = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(pil_image, 'uint8')

            faces = self.detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(user_id)

        return face_samples, ids

    def train_for_new_user(self, new_user_name: str, new_user_id: int):
        # Path for face image database
        path = 'DataSet/Users/' + new_user_name

        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = self._get_images_and_labels(path, new_user_id)

        if os.path.isfile('Trainer/trainer.yml'):
            print("recognizer is reloading from an existing state")
            self.recognizer.read('Trainer/trainer.yml')

        self.recognizer.update(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        self.recognizer.write('Trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    def train_for_new_users(self, new_users_names, new_users_ids):
        # Path for face image database
        faces = []
        ids = []
        for new_user_name, new_user_id in zip(new_users_names, new_users_ids):
            path = 'DataSet/Users/' + new_user_name

            print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

            curr_faces, curr_ids = self._get_images_and_labels(path, new_user_id)
            faces += curr_faces
            ids += curr_ids

        if os.path.isfile('Trainer/trainer.yml'):
            self.recognizer.read('Trainer/trainer.yml')

        self.recognizer.update(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        self.recognizer.write('Trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    def scan_new_user(self, number_of_pics: int = 30):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        # For each person, enter one numeric face user_name
        user_name = input('\n enter new user name    ==>  ')
        imgages_path = "DataSet/Users/" + str(user_name)

        try:
            os.mkdir(imgages_path)
        except OSError:
            print("\nUser already exists in the system")
        else:
            print("\n [INFO] Initializing face capture. Look the camera and wait ...")

        # Initialize individual sampling face count
        count = 0

        while count < number_of_pics:
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite(imgages_path + '/' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

        return user_name


if __name__ == '__main__':
    fd = FaceDetector()
    fd.train_for_new_user("jawad", 2)

