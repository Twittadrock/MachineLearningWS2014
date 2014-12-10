import numpy as np
import math as math
import pandas as pd
import matplotlib.pyplot as plt

#Perzeptron-Kriterium
#	IN: wrongClassifieds - List/Matrix with m as points and n as coordinates/attributes
#		weightVector as column vector
#	Return:
#	 		Positive and equals 0 if weightVector is a solution	
def errorFunction(weightVector, wrongClassifieds):
	return sum(np.dot(wrongClassifieds, -1 * weightVector))

def devErrorFunction(wrongClassifieds):
	return sum(-1*wrongClassifieds)

#Normalizing the trainingsSet
# 	Return: 
#		returns normalized trainingsSet
def normalize(trainingsSet):

	trainingsSetCopy = np.copy(trainingsSet)
	R = -1

	for i in range(trainingsSetCopy.shape[0]) :

		length = math.sqrt(math.pow(trainingsSetCopy[i, 0], 2) + math.pow(trainingsSetCopy[i, 1], 2))
		if R == -1:
			R = length

		if length > R:
			R = length

	trainingsSetCopy[:,0:2] = trainingsSetCopy[:,0:2]/length

	return trainingsSetCopy, length

#Batch processing function
#	Returns: 
def batchPerzeptron(trainingsSet):

	minX, maxX, minY, maxY = findMinMaxXFromTrainingsSet(trainingsSet)
	normalizedTrainingsSet, length = normalize(trainingsSet)
	
	w = np.array([0,0])
	b = 0
	n = 1
	l = normalizedTrainingsSet.shape[0]

	R_quad = 1
	m_nearest = float('inf')

	while True:
		wrongClassified = False

		#Folie 8
		for i in range(l):
			# test if 0::-1
			xi = normalizedTrainingsSet[i,0:2]
			yi = normalizedTrainingsSet[i, 2]
			m = (np.dot(w,xi) + b) * yi
			
			if m < m_nearest:
				m_nearest = m
			
			if m <= 0:
				#Folie 22
				w = w + n * yi * xi
				b = b + n * yi * R_quad
				wrongClassified = True

				plot(trainingsSet, w, b * length, minX, maxX,minY,maxY)
		
		if (wrongClassified == False):
			break
	#normalze w to functional distance
	#w = w/np.linalg.norm(w, ord=2) * m_nearest  
	return w, b * length #, np.linalg.norm(w, ord=2)

#Searches for min and max x in trainingsSet
#	returns minX, maxX 
def findMinMaxXFromTrainingsSet(trainingsSet): 
	minX = 0
	maxX = 0
	minY = 0
	maxY = 0
	for i in range(len(trainingsSet)):
		currentX = trainingsSet[i, 0]
		currentY = trainingsSet[i,1]
		if currentX < minX:
			minX = currentX
		if currentX > maxX:
			maxX = currentX
		if currentY < minY:
			minY = currentY
		if currentY > maxX:
			maxY = currentY
	return int(math.ceil(minX)), int(math.ceil(maxX)), int(math.ceil(minY)), int(math.ceil(maxY))


#Plot graphic depending on passed parameters
#	returns: none
def plot(trainingsSet, w, b, minX, maxX, minY, maxY):

	print w, b, minX, maxX

	fig, axs = plt.subplots(1)


	xl = np.linspace(minX, maxX, 50)
	yl = np.linspace(minY, maxY, 50)
	X, Y = np.meshgrid(xl,yl)
	ZX = X * w[0]
	ZY = Y * w[1]
	Zdata = ZX + ZY + b 

	levels = np.linspace(-200, 200, 20)
	cs = axs.contourf(X, Y, Zdata, levels=levels)
	fig.colorbar(cs, format="%.2f")


	x = np.array(range(minX, maxX))
	y = (-w[0]*x -b)/w[1]

	designMatrixDf = pd.DataFrame(trainingsSet)

	colors = np.where(designMatrixDf[2] == 1, 'r', 'b')
	designMatrixDf.plot(kind='scatter', x=0 , y=1, c=colors, ax=axs)
	axs.plot(x, y, 'k-', c='y')

	plt.show()