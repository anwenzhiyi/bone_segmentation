import os
from skimage import data, filters, io, color, morphology, exposure
import numpy as np
import skimage.morphology as sm
from skimage.morphology import disk
import matplotlib.pyplot as plt
import skimage.filters.rank as sfr
from skimage import img_as_bool
from skimage import draw,transform
from PIL import Image
from pylab import plot
from sklearn import linear_model
from array import *


def img_resize(f):
    img = io.imread(f)
    row = 720
    col = 960
    #print(row,col)
    img = transform.resize(img,(row,col))
    return img

str1='/home/lucas/Lab/Project1/之前分割数据/image_init/'
str2='/home/lucas/Lab/Project1/之前分割数据/label_init/'
files = os.listdir(str1)
for file in files:
    print(file)
    image = img_resize(str1+file)
    io.imsave('/home/lucas/Lab/Project1/之前分割数据/image_init/' + file[0:-4]+'.jpg',image)

files = os.listdir(str2)
for file in files:
    print(file)
    image = img_resize(str2 + file)
    io.imsave('/home/lucas/Lab/Project1/之前分割数据/label_init/' + file[0:-5] + '.jpg', image)



#coll=io.ImageCollection(str1,load_func=img_resize)
#for i in range(len(coll)):
    #io.imsave('/home/lucas/Lab/Project1/image1/'+np.str(i+1)+'.jpg', coll[i])