'''
填充孔洞
'''
import cv2
from skimage import io,img_as_bool,morphology
import numpy as np

path = "/home/lucas/Lab/finetune_segmentation/Fnet4_results/"
path1 = "/home/lucas/Lab/finetune_segmentation/Fnet4_results1/"

def fillHole(image):
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  # 二值化处理
    h = 512
    w = 512
    ret, thre = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  #二值化处理
    mask = np.zeros((h+2,w+2),np.uint8)                            #掩码图像
    #mask = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_CONSTANT,value=[0])
    #mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    cv2.floodFill(thre,mask,(0,0),(255))          #漫水填充法
    dst = image | (~thre)
    return dst

if __name__ == "__main__":
    for k in range(0,121):
        seg = cv2.imread(path+str(k)+'.jpg',0)
        seg = fillHole(seg)
        cv2.imwrite(path1 + str(k) + '.jpg', seg)
        seg = io.imread(path1+str(k)+'.jpg',seg)
        seg = (seg>=127)
        seg = morphology.remove_small_objects(seg, min_size=1000, connectivity=1)
        io.imsave(path1+str(k)+'.jpg',seg.astype('float32'))
        #cv2.imshow('image',seg)
        #cv2.waitKey(0)



