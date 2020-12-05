from IdentificationLogic.FacialRecognition.Trainer.face_training import FaceDetector
from IdentificationLogic.FacialRecognition.FacialClassifier import face_classifier

import cv2
import os


class FaceRecognitionHandler:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.number_of_users = 0  # the number of users that are already in the system, supports starting users.
        self.user_names = dict()  # {0: "nader", 1: "asad"}
        self.number_of_pictures = 20

    def add_user(self):
        user_name = self.face_detector.scan_new_user(self.number_of_pictures)
        self.face_detector.train_for_new_user(user_name, self.number_of_users)
        self.user_names[self.number_of_users] = user_name
        self.number_of_users = self.number_of_users + 1

    def scan_camera(self):
        face_classifier.scan_faces(self.user_names)


if __name__ == '__main__':
    '''
    face_recognition_handler = FaceRecognitionHandler()
    face_recognition_handler.add_user()
    face_recognition_handler.scan_camera()
    '''
    face_recognition_handler = FaceRecognitionHandler()

    #face_recognition_handler.add_user()
    #face_recognition_handler.add_user()

    face_recognition_handler.scan_camera()

