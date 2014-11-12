# -*- coding: utf-8 -*-
"""
Created on Thu Nov 06 17:42:50 2014

@author: Matthias
"""

import os
from skimage import io
import numpy as np
import math
import MyPCA as pca
import pandas as pd

class Eigengesichter(object):

    path = "lfw_funneled"
    
    file_image_data = "image_data.npy"
    file_image_data_test = "image_data_test.npy"
    file_names_data = "names_data.npy"
    file_names_data_test = "names_data_test.npy"
    
    min_img_count = 70
    scale = 8
    components = 150
    
    mdataImage = None
    mdataTestImage = None
    
    mdataNamesTest = []
    mdataNames = []
    U = None
    S = None
    VT = None



    # Aufgabe a
    def getFolders(self):
        dirs = os.listdir(self.path)
        names = []
        for folder in dirs:
            if os.path.isdir(self.path + '/' +folder):
                if len(os.listdir(self.path + '/' + folder)) >= self.min_img_count:
                    names.append(folder)
        return names                
      
    #Returns all Images as rows in a data matrix  
    def loadImages(self,name,training='True'):
        tmpimages = os.listdir(self.path + '/' + name)
        images = []
        if training == 'True':
            for i in range(len(tmpimages)-2):
                images.append(tmpimages[i])
        else:
            images.append(tmpimages[len(tmpimages)-1])
        mdata = []
        if len(images) > 0:
            #Calc image size
            self.img_path = images[0]
            img =  io.imread(self.path +'/' +  name + '/' +self.img_path)
            imagesize = img.shape[0]
            #allocate memory
            rows = len(images)
            toPow = math.ceil(float(imagesize*1)/self.scale)
            columns = math.floor(math.pow(toPow,2))
            mdata = np.zeros((rows,columns))
            #define range
            rangeCountImages = range(rows)
            
            for i in rangeCountImages:
                self.img_path = images[i]
                #print name + '/' +img_path
                img =  io.imread(self.path +'/' +  name + '/' +self.img_path, as_grey=True)
                im_shape = img.shape
                
                #Allocate memory for row
                imageAsVector = np.zeros(columns)
                counter = 0
                for m in xrange(0,im_shape[0],self.scale):
                    for n in xrange(0,im_shape[1],self.scale):
                        #grey = 1/3 * (img[m,n,0] + img[m,n,1] + img[m,n,2])
                        imageAsVector[counter] = img[m,n]
                        counter = counter + 1
                #Write row
                mdata[i,:] = imageAsVector
            return mdata
               
    def addNames(self,name,count):
        for i in range(count):
            self.mdataNames.append(name)
    
    def addNamesTest(self,name,count):
        for i in range(count):
            self.mdataNamesTest.append(name)
   
    # prepare training data
    def prepareData(self):
        names = self.getFolders()
        
        if not os.path.isfile(self.file_image_data):
            print "Found " + str(len(names)) + " persons"
            for name in names:
                
               personImages = self.loadImages(name)
               self.addNames(name,personImages.shape[0])
               if self.mdataImage is None:
                   self.mdataImage = personImages
               else:
                   self.mdataImage = np.concatenate((self.mdataImage, personImages), axis=0)
               print "Finished current person training data: " + name
               
            np.save(self.file_image_data,self.mdataImage)
            np.save(self.file_names_data, self.mdataNames)
        else:
            self.mdataImage = np.load(self.file_image_data)
            self.mdataNames = np.load(self.file_names_data)   
            
    # prepare test data
    def prepareTestData(self):        
        names = self.getFolders()
                
        if not os.path.isfile(self.file_image_data_test):
            print "Found " + str(len(names)) + " persons"
            for name in names:
               
               personImageTest = self.loadImages(name,'Test')
               self.addNamesTest(name,personImageTest.shape[0])
               
               if self.mdataTestImage is None:
                   self.mdataTestImage = personImageTest
               else:
                   self.mdataTestImage = np.concatenate((self.mdataTestImage, personImageTest), axis=0)
               print "Finished current person test data: " + name
               
               self.mdataTestImage = pca.normalize(self.mdataTestImage)
               
            np.save(self.file_image_data_test, self.mdataTestImage)
            np.save(self.file_names_data_test, self.mdataNamesTest)
        else:
            self.mdataTestImage = np.load(self.file_image_data_test)
            self.mdataNamesTest = np.load(self.file_names_data_test) 
            
        
    
    def doPCA(self):
        self.prepareData()
        self.prepareTestData()
        self.U,self.S,self.VT = pca.calcPCA(self.mdataImage,self.components)
        
    def doEigenwerte(self):
        SList = np.transpose(np.diag(self.S))
        c = 0
        for n in range(self.components):
            print str(c) + ' ' + str(SList[n])
            c += 1
  
    def doEigengesichter(self,pos):
        dim = int(math.sqrt(self.VT.shape[1]))
        mdata = np.zeros((dim,dim))
        imageVector = self.VT[pos,:]
        for i in range(0,len(imageVector),int(dim)):
            row = np.array(imageVector)[i:i+dim]
            mdata[int(i/dim),:] = row
        io.imshow(mdata)          
        
        #print pd.DataFrame(Components).head()
   
    def TestEigengesichter(self):
       sevenEigengesichter = self.VT[0:7,:]
       projectedTest = np.dot(self.mdataTestImage,np.transpose(sevenEigengesichter))
       projectedTraining = np.dot(self.mdataImage,np.transpose(sevenEigengesichter))
       print projectedTest.shape
       print projectedTraining.shape
       
       for test in range(projectedTest.shape[0]):
           testProps = projectedTest[test,:]
           lowestDistance = None
           lowestIndex = None
           for training in range(projectedTraining.shape[0]):
               trainingProps = projectedTraining[training,:]
               diff = testProps - trainingProps
               euklidDistance = math.sqrt(sum(diff * diff))
               if lowestDistance is None:
                    lowestDistance = euklidDistance
                    lowestIndex = training
               
               if euklidDistance < lowestDistance:
                    lowestDistance = euklidDistance
                    lowestIndex = training
                    
           if self.mdataNamesTest[test] == self.mdataNames[lowestIndex]:
               print 'I am correct classified: ' + self.mdataNamesTest[test]
           else:
               print 'I am NOT correct classified: ' + self.mdataNamesTest[test]
               print '>> >> I am NOT: ' + self.mdataNames[lowestIndex]
   
    
if __name__ == '__main__':
   obj = Eigengesichter()
   obj.doPCA()
   obj.TestEigengesichter()
   exit()    



        
