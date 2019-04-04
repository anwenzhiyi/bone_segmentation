'''
将中间区域全部涂成白色
'''

import os
from skimage import data, filters, io, color, morphology, exposure
import numpy as np
import skimage.morphology as sm
from skimage.morphology import disk
import matplotlib.pyplot as plt
import skimage.filters.rank as sfr
from skimage import img_as_bool,img_as_ubyte
from skimage import draw,transform
from PIL import Image
from pylab import plot
from sklearn import linear_model
from array import *


str1='/home/lucas/Lab/Project1/data/top2bottom'
files = os.listdir(str1)
for file in files:
    print(file)

    image = io.imread(str1+'/'+file)

    print(image.shape)
    print(image.max(),image.min())

    image[image > 180] = 255
    image[image <= 180] = 0

    '''
    #左右扫描
    for row in range(512):
        col1 = col2 = -1
        for col in range(512):
            if image[row][col]==255:
                col1 = col
                break
        for col in range(511,0,-1):
            if image[row][col]==255:
                col2 = col
                break
        if col1==-1 or col2==-1:
            continue
        for col in range(col1,col2):
            image[row][col]=255
    '''

    #上下扫描
    for col in range(512):
        row1 = row2 = -1
        for row in range(512):
            if image[row][col]==255:
                row1 = row
                break
        for row in range(511,0,-1):
            if image[row][col]==255:
                row2 = row
                break
        if row1==-1 or row2==-1:
            continue
        for row in range(row1,row2):
            image[row][col]=255

    print(image.max(),image.min())

    io.imsave('/home/lucas/Lab/Project1/data/top2bottom/' + file[0:-4]+'_.jpg',image)
