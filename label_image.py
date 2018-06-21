'''
将手工标注的图像处理成可以训练的label
'''
from PIL import Image
from skimage import io
import os
from skimage import data, filters, io, color, morphology, exposure


def function(f):
    img = io.imread(f)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] > 130 and img[i][j][1] > 130 and img[i][j][2] < 120:
                img[i][j][:] = 255
            else:
                img[i][j][:] = 0

    img = color.rgb2gray(img)
    img = img.astype('float32')

    #print(img.max())
    #print(img.min())

    img[img > 0.5] = 1
    img[img <= 0.5] = 0

    img = img.astype('bool')
    img = morphology.remove_small_objects(img, min_size=50, connectivity=1)

    return img

str1='/home/lucas/Lab/Project1/标注数据/label_data2'
files = os.listdir(str1)
for file in files:
    print(file)
    image = function(str1+'/'+file)
    io.imsave('/home/lucas/Lab/Project1/标注数据/label_data3/' + file[0:-4]+'.jpg',image.astype('float32'))
