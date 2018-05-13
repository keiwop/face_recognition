#! /usr/bin/env python3
# On Archlinux, I had to install opencv (3.4.1), gtkglext (1.2.0), python-matplotlib (2.2.2) and python-numpy (1.14.3) 

import config

import numpy as np
import time
import cv2
import os

from os.path import join



def imgToRGB(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imgToGray(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



class FaceRecognition:
	def __init__(self):
		self.training_faces = []
		self.training_ids = []
		self.training_names = []
		self.face_recognizer = None
		self.cam = None
		self.original_img = None
		self.predicted_img = None
		
		self.prepare_training_data()
		self.start_training("lbp")
	
	
	def start_camera(self):
		self.cam = cv2.VideoCapture(0)
	
	
	def predict_cam_img(self):
		cam_img = self.cam.read()[1]
		cam_img_resized = self.scale_img(cam_img)
		self.predicted_img = cam_img
		self.predict(cam_img_resized)
	
	
	def show_prediction(self):
		cv2.imshow("predicted", self.predicted_img)
		cv2.waitKey(1)
	
	def show_next_cam_img(self):
		cam_img = self.cam.read()[1]
		cv2.imshow("predicted", cam_img)
		cv2.waitKey(1)
	
	
	def detect_faces(self, img, classifier="lbp"):
		gray_img = imgToGray(img)
		
		if classifier == "lbp":
			cascade_classifier = cv2.CascadeClassifier(join(config.classifier_path, "lbpcascades/lbpcascade_frontalface_improved.xml"))
		elif classifier == "haar":
			cascade_classifier = cv2.CascadeClassifier(join(config.classifier_path, "haarcascades/haarcascade_frontalface_alt.xml"))
			
		faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.2)
		return faces


	def get_single_face(self, img, faces, idx=0):
		gray_img = imgToGray(img)
		rect = faces[idx]
		x, y, w, h = rect
		face = gray_img[y:y+h, x:x+w]
		if config.show_detected_face:
			cv2.imshow("FACE", face)
			cv2.waitKey(1)
		return face, rect


	def draw_rectangle(self, img, rect):
		x, y, w, h = rect
		cv2.rectangle(img, (x, y), (x+w, y+h), config.rect_color, config.rect_width)


	def draw_text(self, img, text, x, y):
		cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, config.font_size, config.rect_color, config.rect_width)


	def prepare_training_data(self):
		dirs = os.listdir(config.training_path)
		self.training_names = [""] * len(dirs)
		
		for dir_name in dirs:
			if not dir_name.startswith("p"):
				continue
			person_id, person_name = dir_name.split(".")
			person_images = os.listdir(join(config.training_path, dir_name))
			training_id = int(person_id[1:])
			
			for img_name in person_images:
				img_path = join(config.training_path, dir_name, img_name)
				img = cv2.imread(img_path)
				
				faces_list = self.detect_faces(img, "lbp")
				face, rect = self.get_single_face(img, faces_list)
				if face is not None:
					self.training_faces.append(face)
					self.training_ids.append(training_id)
					self.training_names[training_id] = person_name
	
	
	def start_training(self, recognizer="lbp"):
		print(dir(cv2.face))
		if recognizer == "lbp":
			self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
		elif recognizer == "eigen":
			self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
		elif recognizer == "fisher":
			self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
		self.face_recognizer.train(self.training_faces, np.array(self.training_ids))
	
	
	def scale_img(self, img):
		return cv2.resize(img, (0, 0), fx=config.scaling_factor, fy=config.scaling_factor)
		
	
	def scale_rect(self, rect):
		for i, val in enumerate(rect):
			rect[i] = val * int(1 / config.scaling_factor)
		return rect
	

	def predict(self, img):
		faces = self.detect_faces(img)
		print("FACES:", faces)
		for i in range(len(faces)):
			print("I:", i)
			face, rect = self.get_single_face(img, faces, idx=i)
			person_id = self.face_recognizer.predict(face)[0]
			print("RECOGNIZER:", self.face_recognizer.predict(face))
			print("ID:", person_id)
			name = self.training_names[person_id]
			rect = self.scale_rect(rect)
			x, y, w, h = rect
			self.draw_rectangle(self.predicted_img, rect)
			self.draw_text(self.predicted_img, name, x, y-5)
		self.working_img = img
	



if __name__ == "__main__":

	face_rec = FaceRecognition()
	face_rec.start_camera()

	while True:
		try:
			#img_test_path = join(config.test_img_path, "Maxime_0.jpg")
			#img_test = cv2.imread(img_test_path)
			#face_rec.predict(img_test)
			
			face_rec.predict_cam_img()
			face_rec.show_prediction()
			#face_rec.show_next_cam_img()
		except Exception as e:
			print("Exception:", e)
		time.sleep(0.01)

	face_rec.cam.release()
	print("End of program")
