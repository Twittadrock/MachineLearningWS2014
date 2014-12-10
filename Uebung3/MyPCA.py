# -*- coding: utf-8 -*-
"""
Created on Thu Nov 06 14:42:22 2014

@author: Matthias
"""

import numpy as np
import math

# Input: 	Data matrix - rows are samples and columns are properties
			#components -1 --> All components
# Output: 	PCA - where 
			#1. U is U from SVD, 
			#2. S is D squared from SVD and 
			#3. UT is U transposed from SVD
# Relationship PCA and SVD
 		    #PCA Kovarianzmatrix: X(XT) = WD(WT)   
# Attention Samples as columns and properties as rows
    		#SVD X = UD(VT)
    		#Relationship: X(XT) = U(D2)(UT)
def calcPCA(matrix,components):
    #Normalize the data
    matrix = normalize(matrix)
    #Calculate SVD from transposed data matrix
    U, D, VT = np.linalg.svd(matrix)
    #print U.shape
    #print D.shape
    #print VT.shape
    #Construct PCA from SVD
    UT = np.transpose(U)
    #Reduce dimensions
    if components == -1:
    	components = D.shape[0]
    Sdiag = np.diag(D[0:components])
    SReduced = np.zeros(matrix.shape)
    SReduced[0:components,0:components] = Sdiag
    return U,SReduced,VT


def normalize(matrix):
    for x in range(matrix.shape[1]) :
        mean = sum(matrix[:,x])/matrix.shape[0]
        sstd = np.std(matrix[:,x])
        matrix[:,x] = (matrix[:,x] - mean)/sstd 
    return matrix

        