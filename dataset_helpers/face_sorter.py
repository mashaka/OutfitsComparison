import face_recognition
import os
import sys
import numpy as np
import pickle
import csv

#
# Change this to your paths
#

PHOTOS_PATH = "./../Datasets/Fashion144k_v1/photos/"
INPUT_PATH = "./../Datasets/Fashion144k_v1/photos.txt"
OUTPUT_PATH = "./results/"
ENCODINGS_PATH = "./encodings.pickle"
TEST_IDS_PATH = "./../Datasets/Fashion144k_v1/testids.csv"

image_names = []
encodings = []
clusters = []
testids = set()


def readTestIds():
	with open(TEST_IDS_PATH) as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None)
		for row in reader:
			testids.add(int(row[0]))
	print("TOTAL NUMBER OF TEST IDS: " + str(len(testids)))
	print("_______________________________________________________")
			


def read_images():
	with open(INPUT_PATH) as f:
		for i, line in enumerate(f):
			name = line.strip('\n')
			if os.path.isfile(PHOTOS_PATH + name) and i in testids:
				image_names.append(name)

	print("TOTAL NUMBER OF IMAGES: " + str(len(image_names)))
	print("_______________________________________________________")


def prepare_encodings():
	read_images()
	for i, image_name in enumerate(image_names):
		image = face_recognition.api.load_image_file(PHOTOS_PATH + image_name)
		if i in testids:
			encodings.append(face_recognition.api.face_encodings(image, num_jitters=10))
		print("    ECODED: " + str(i + 1) + "/" + str(len(image_names)))
	with open(ENCODINGS_PATH, 'wb') as f:
		pickle.dump(encodings, f)
	print("_______________________________________________________")


def load_encodings():
	read_images()
	with open(ENCODINGS_PATH, 'rb') as f:
		for i, e in enumerate(pickle.load(f)):
			if i in testids:
				encodings.append(e)
	print("ENCODINGS LOADED")
	print("_______________________________________________________")


def create_clusters():
	isFirstSet = False
	for i, encoding in enumerate(encodings):
		if len(encoding) == 0:
			continue

		if isFirstSet == False:
			clusters.append([i])
			isFirstSet = True
			continue

		clusters_representative_ids = [cluster[0] for cluster in clusters]
		clusters_representative_encodings = [encodings[id][0] for id in clusters_representative_ids]

		try:
			results = face_recognition.api.compare_faces(np.array(clusters_representative_encodings), encoding, tolerance=0.4)
		except:
			continue

		for r, result in enumerate(results):
			if result == True:
				clusters[r].append(i)
				break
		clusters.append([i])
		print("    CLUSTER DONE: " + str(i + 1) + "/" + str(len(encodings)))
	print("_______________________________________________________")


def create_results_file():
	for i, cluster in enumerate(clusters):
		if len(cluster) < 2:
			continue
		file = open(OUTPUT_PATH + str(i) + '.txt', 'wt', encoding='utf8')
		images = [image_names[id] for id in cluster]
		file.write("\n".join(images))
		file.close()


readTestIds()
#prepare_encodings()
load_encodings()
create_clusters()
create_results_file()