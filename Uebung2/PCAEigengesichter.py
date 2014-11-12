# -*- coding: utf-8 -*-
"""
Created on Thu Nov 06 17:42:50 2014

@author: Matthias
"""

import os
from skimage import io
import numpy as np
import math

path = "lfw_funneled/"
min_img_count = 70
scale = 8
mdataAll = np.array([])

# Aufgabe a
def getFolders():
    dirs = os.listdir(path)
    names = []
    for folder in dirs:
        if os.path.isdir(path + folder):
            if len(os.listdir(path + folder)) >= min_img_count:
                names.append(path + folder)
    return names                
  
#Returns all Images as rows in a data matrix  
def loadImages(name):
    images = os.listdir(name)
    images_to_process = len(images)-1
    mdata = []
    if len(images) > 0:
        #Calc image size
        img_path = images[0]
        img =  io.imread(name + '/' +img_path)
        imagesize = img.shape[0]
        #allocate memory
        rows = images_to_process
        toPow = math.ceil(float(imagesize*1)/scale)
        columns = math.floor(math.pow(toPow,2))
        mdata = np.zeros((rows,columns))
        #define range
        range_countImages = range(images_to_process)
        # DEBUG
        #range_countImages = range(1)
        # DEBUG
        for i in range_countImages:
            img_path = images[i]
            print name + '/' +img_path
            img =  io.imread(name + '/' +img_path, as_grey=True)
            im_shape = img.shape
            
            #Allocate memory for row
            imageAsVector = np.zeros((1,columns))
            counter = 0
            for m in xrange(0,im_shape[0],scale):
                for n in xrange(0,im_shape[1],scale):
                    #grey = 1/3 * (img[m,n,0] + img[m,n,1] + img[m,n,2])
                    imageAsVector[1:counter] = img[m,n]
                    counter = counter + 1
            #Write row        
            mdata[i,:] = imageAsVector
    return mdata



def buildDesignMatrix():
    return 1

    
if __name__ == '__main__':
    names = getFolders()
    print len(names)
    mdataPerson =  loadImages(names[0])
    print mdataPerson.shape
    mdataAll
    np.concatenate((mdataAll, mdataPerson), axis=0)
    exit()

        
