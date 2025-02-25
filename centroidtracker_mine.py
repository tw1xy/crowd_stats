# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=20):
		self.nextObjectID = 0
		self.objects_all = OrderedDict()
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		self.object_person = OrderedDict()
		self.object_face = OrderedDict()
		self.object_age = OrderedDict()
		self.object_gender = OrderedDict()
		
		self.maxDisappeared = maxDisappeared


	def age(self,ID):
		return self.object_age.get(ID)

	def reg_age(self,ID, age_itself):
		self.object_age[ID] = age_itself
		
	def gender(self,ID):
		return self.object_gender.get(ID)

	def reg_gender(self,ID, gender_itself):
		self.object_gender[ID] = gender_itself

	def face(self,ID):
		return self.object_face.get(ID)

	def reg_face(self,ID, face_itself):
		self.object_face[ID] = face_itself

	def person(self,ID):
		return self.object_person.get(ID)

	def reg_person(self,ID, person_itself):
		self.object_person[ID] = person_itself



	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]
		try:		
			del self.object_person[objectID]
			del self.object_face[objectID]
			del self.object_age[objectID]
			del self.object_gender[objectID]
		except:
			pass

	def update(self, rects):

		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		rect = []

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
			

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			

			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		return self.objects