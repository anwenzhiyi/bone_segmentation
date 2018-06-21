"""
12.28  created by cyt
骨骼分割预处理之截取ROI,统一图像格式为512*512
"""
import angle_calculation
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
import os
import time

#根据row和col剪一个512*512的框体，使之包纳整个分割区域
#用字典 cut_area[]来记录下裁剪部分的信息，后面可能会用到
def cut_segmentation_ROI(row,col,image,cut_area):
    left = col-60
    right = col+452
    top = row-200
    bottom = row+312

    #调整,使之避免超出边界;以及避免在太过边界上，无用信息太多
    if left < 30:
        left = 30
        right = 542
    elif right > image.shape[1]-30:
        right = image.shape[1]-30
        left = right-512
    if top < 50:
        top = 50
        bottom = 562
    elif bottom > image.shape[0]-20:
        bottom = image.shape[0]-20
        top = bottom-512

    cut_area['left'] = left
    cut_area['right'] = right
    cut_area['top'] = top
    cut_area['bottom'] = bottom

    #裁剪
    ROI = image[top:bottom,left:right,:]
    return ROI


#用来处理每张图片,根据横向骨骼定位的结果选取合适的ROI(512*512),所以rescaleRate2没用了
#返回值为一个字典,存放了之后处理所需的信息
#pre_segmentation = {'image_initial': image_initial,'image_rescaled':image, 'image_ROI': image_ROI,
#                        'dst_image': dst_image,
#                        'cut_area_dict':cut_area,
#                        'rescaleRate1': rescale_rate1, 'rescaleRate2': rescale_rate2}
def pre_seg(f,file):
    image = io.imread(f)
    image_initial = io.imread(f)

    label = io.imread('/home/lucas/Lab/Project1/之前分割数据/label_data_cutted/'+file)

    #dict = {'image':image,'scaleRate':1,'row':300,'col':300,'image_operated':image}     #初始化
    dict = angle_calculation.function(image)

    image = dict['image']              #经过缩放的图片
    rescale_rate1 = dict['scaleRate']
    row = dict['row']
    col = dict['col']
    image_operated = dict['image_operated']

    #根据原图缩放率对标签图进行缩放
    label = transform.rescale(label,rescale_rate1)

    #根据row和col剪一个512*512的框体，使之包纳整个分割区域
    #同时记录下分割区域在原图中的位置信息，存放在 cut_area 字典中
    cut_area={'left':0,'right':0,'top':0,'bottom':0,}
    image_ROI = cut_segmentation_ROI(row,col,image,cut_area)
    label_ROI = cut_segmentation_ROI(row,col,label,cut_area)

    # float 转 unit8类型转换
    image_ROI = img_as_ubyte(image_ROI)
    label_ROI = img_as_ubyte(label_ROI)

    #将图像缩放至统一的512*512，记录下总的缩放率    ,因为第二次改代码直接裁成512×512,因此不用再次缩放了
    dst_image = transform.resize(image_ROI,(512,512))
    dst_label = transform.resize(label_ROI,(512,512))
    rescale_rate2 = 512.0/500.0

    # float 转 unit8类型转换
    dst_image = img_as_ubyte(dst_image)

    # 字典pre_segmentation用来存储分割前的截取ROI区域并缩放的信息,image是原图
    # 同时记录下分割区域在原图的缩放图中的位置信息，存放在 cut_area_dict 字典中
    pre_segmentation = {'image_operated': image_operated,'image_rescaled':image, 'image_ROI': image_ROI,
                        'dst_image' : dst_image, 'dst_label':dst_label,
                        'cut_area_dict':cut_area,
                        'rescaleRate1': rescale_rate1, 'rescaleRate2': rescale_rate2}

    return pre_segmentation


if __name__ == '__main__':
    time1 = time.time()
    str1='/home/lucas/Lab/Project1/之前分割数据/image_cutted'
    files = os.listdir(str1)
    for file in files:
        print(file)
        dict = pre_seg(str1+'/'+file,file)
        image = dict['dst_image']
        label = dict['dst_label']
        io.imsave('/home/lucas/Lab/Project1/之前分割数据/train_data/' + file[0:-4] + '.jpg',image)
        io.imsave('/home/lucas/Lab/Project1/之前分割数据/label_data/' + file[0:-4] + '.jpg',label)
    print(time.time()-time1)



