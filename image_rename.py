"""
脚本1：处理图片，将其统一命名格式
"""

import re
import os
import time
from skimage import io

def change_name(path):
    global i
    if not os.path.isdir(path) and not os.path.isfile(path):
        return False
    if os.path.isfile(path):
        file_path = os.path.split(path)  #分割出目录与文件
        lists = file_path[1].split('.')  #分割出文件与文件扩展名
        image1 = io.imread(path)
        if lists[0][-1]=='m' or lists[0][-1]=='M':       #原图
            i += 1
            os.rename(path, file_path[0]+'/'+str(i)+ '.jpg')
            #根据前缀名字一样找出标注图
            for x in os.listdir(img_dir):
                if x[0] == lists[0][0] and x[1] == lists[0][1] and x[2] == lists[0][2] and x[len(lists[0])-1] == lists[0][-1]:
                    if x[-5] == '角':
                        image = io.imread(file_path[0] + '/' + x)
                        os.rename(img_dir+x, file_path[0]+'/'+str(i)+'_1'+ '.jpg')    #角度标注图
                        io.imsave('/media/lucas/新加卷/lab.data/角度数据/4月角度数据/角度数据/' + str(i) + '.jpg', image1)
                        io.imsave('/media/lucas/新加卷/lab.data/角度数据/4月角度数据/角度数据/' + str(i) + '_.jpg', image)
                    elif x[-5] == '头':
                        image = io.imread(file_path[0] + '/' + x)
                        os.rename(img_dir + x, file_path[0] + '/' + str(i) + '_2' + '.jpg')    #分割标注图
                        io.imsave('/media/lucas/新加卷/lab.data/角度数据/4月角度数据/分割数据/' + str(i) + '.jpg', image1)
                        io.imsave('/media/lucas/新加卷/lab.data/角度数据/4月角度数据/分割数据/' + str(i) + '_.jpg', image)

    elif os.path.isdir(path):
        for x in os.listdir(path):    #'连接符'.join(list) 将列表组成字符串
            change_name(os.path.join(path,x))


if __name__ == '__main__':
    img_dir = '/home/lucas/Lab/data_2018.4.2/all/'
    start = time.time()
    i=-1                  #序号从i+1开始命名
    change_name(img_dir)
    c = time.time()-start
    print('程序运行耗时：%0.2f' %(c))