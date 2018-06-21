"""
该脚本实现图像裁剪,将超声图像的周围无用区域去掉
"""
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
import os

def image_cut(image):
    dst1 = color.rgb2gray(image)

    high=0  #上边界
    low=0   #下边界
    right=0
    left=0

    #首先找出上边界
    for i in range(dst1.shape[0]//10,dst1.shape[0]//2):
        k=0
        for j in range(dst1.shape[1]):
            if dst1[i,j]>0.2:
                k+=1
        if k > dst1.shape[1]//4:
            high = i
            break
    #找出下边界
    for i in range(dst1.shape[0]-60, 65, -1):
        k=0
        for j in range(dst1.shape[1]):
            if dst1[i,j]>0.1:
                k+=1
        if k > 30:
            low = i
            break
    #找出左边界
    for i in range(dst1.shape[1]//2,40,-1):
        flag=0
        for j in range(high,high+20):
            if dst1[j,i]>0.1:
                for k in range(high+40,low-20):
                    if dst1[k,i]>0.1:
                        flag=1
                        break
                if flag == 1:
                    break
        if flag==0:
            left = i+1
            break
    #找出右边界
    for i in range(dst1.shape[1]//2,dst1.shape[1]-30):
        flag=0
        for j in range(high,high+20):
            if dst1[j,i]>0.1:
                flag=1
        if flag==0:
            right = i-1
            break
    #print(low)
    #print(high)
    #print(left)
    #print(right)

    #裁剪
    roi = image[high:low , left:right , :]
    #roi = transform.resize(roi,(512,512))
    return roi

if __name__ == '__main__':
    str1='/home/lucas/Lab/Project1/之前分割数据/image_init'
    files = os.listdir(str1)
    for file in files:
        print(file)
        image = io.imread(str1+'/'+file)
        image = image_cut(image)
        io.imsave('/home/lucas/Lab/Project1/之前分割数据/image_cutted/' + file[0:-4]+'.jpg',image)




