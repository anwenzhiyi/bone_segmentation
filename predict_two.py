import angle_calculation
from skimage import io, color, morphology, exposure
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw,transform,img_as_bool,img_as_ubyte
from PIL import Image
from sklearn import linear_model
from array import *
import os
import time
import Fnet
import image_cut,image_rename,Fill_hole
import cv2


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


#用来处理每张图片,根据横向骨骼定位的结果选取合适的ROI(512*512)
#返回值为一个字典,存放了之后处理所需的信息
#pre_segmentation = {'image_initial': image_initial,'image_rescaled':image, 'image_ROI': image_ROI,
#                        'dst_image': dst_image,
#                        'cut_area_dict':cut_area,
#                        'rescaleRate1': rescale_rate1, 'rescaleRate2': rescale_rate2}
def pre_seg(image):

    dict = angle_calculation.function(image)

    image = dict['image']              #经过缩放的图片
    rescale_rate1 = dict['scaleRate']
    row = dict['row']
    col = dict['col']
    image_operated = dict['image_operated']
    angle = dict['angle']

    #根据row和col剪一个512*512的框体，使之包纳整个分割区域
    #同时记录下分割区域在原图中的位置信息，存放在 cut_area 字典中
    cut_area={'left':0,'right':0,'top':0,'bottom':0,}
    image_ROI = cut_segmentation_ROI(row,col,image,cut_area)

    # float 转 unit8类型转换
    image_ROI = img_as_ubyte(image_ROI)

    # 字典pre_segmentation用来存储分割前的截取ROI区域并缩放的信息,image是原图
    # 同时记录下分割区域在原图的缩放图中的位置信息，存放在 cut_area_dict 字典中
    pre_segmentation = {'image_operated': image_operated,'image_rescaled':image, 'dst_image' : image_ROI,
                        'cut_area_dict':cut_area,
                        'rescaleRate1': rescale_rate1,
                        'angle':angle}

    return pre_segmentation

def do_predict(init_image,file,model):
    #将原始医学图像中裁剪出有效区域
    image = image_cut.image_cut(init_image)

    #进行角度预测，并据此裁出ROI（512*512）
    dict = pre_seg(image)
    image_operated1 = dict['image_operated']     #划线后的图像,经过了缩放至短边 650
    angle = dict['angle']
    dst_image = dict['dst_image']          #裁剪出ROI的待分割图像
    cut_area = dict['cut_area_dict']

    #进行分割,分割结果为predict_image
    dst_image = color.rgb2gray(dst_image)        #先转成灰度图,float类型的
    #io.imsave('/home/lucas/Lab/Project1/predict/result1/' + str(k) + '__.jpg', dst_image)
    dst_image = img_as_ubyte(dst_image)
    myFnet = Fnet.myFnet()
    predict_image = myFnet.predict(dst_image,model)
    predict_image.save('/home/lucas/Lab/Project1/predict/result1/'+file+'_.jpg')


    #对分割结果进行后处理,因为用了两个包:cv2和skimage，所以就两次读入了
    seg = cv2.imread('/home/lucas/Lab/Project1/predict/result1/'+file+'_.jpg',0)
    seg = Fill_hole.fillHole(seg)
    cv2.imwrite('/home/lucas/Lab/Project1/predict/result1/' + file + '.jpg', seg)
    seg = io.imread('/home/lucas/Lab/Project1/predict/result1/'+file+'.jpg')
    seg = (seg>=127)
    seg = morphology.remove_small_objects(seg, min_size=1000, connectivity=1)
    seg = img_as_ubyte(seg)     #转化成255的
    io.imsave('/home/lucas/Lab/Project1/predict/result1/'+file+'.jpg',seg)

    #在image_operated1上画出分割区域,即覆盖cut_area
    left = cut_area['left']
    right = cut_area['right']
    top = cut_area['top']
    bottom = cut_area['bottom']
    for i in range(top,top+512):
        for j in range(left,left+512):
            if seg[i-top][j-left]==255:           #将分割区域在原图中涂为绿色
                image_operated1[i][j][0] = 0
                if image_operated1[i][j][1]<150:
                    image_operated1[i][j][1] += 100
                else:
                    image_operated1[i][j][1] = 150
                image_operated1[i][j][2] = 0

    return image_operated1,angle


str1='/home/lucas/Lab/Project1/predict/predict_data/'
if __name__ == '__main__':
    myFnet = Fnet.myFnet()
    model = myFnet.load_model()
    files = os.listdir(str1)
    for file in files:
        image = io.imread(str1+file)
        #print(file)
        image,angle = do_predict(image,file,model)
        print('angle',angle)
        io.imsave('/home/lucas/Lab/Project1/predict/results/'+file,image)