"""
角度计算
11.21版本
最新版本
(当前只有第17张图偏差相对大一点)
批处理
横向骨骼取两个点
纵向骨骼取几十个点用最小二乘法拟合

原始图
纵向均值滤波
纵向对比度增强
二值化+去小连通域
纵向骨骼局部增强
纵向骨骼局部区域二值化
去小连通域
选取合适的多个点用linear_model训练线性模型
"""
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


#将原图按比例缩放至短边边长为650，返回一个字典
def rescale_image(dic):
    image = dic['image']
    if image.shape[0]<image.shape[1]:
        k = image.shape[0]
    else:
        k = image.shape[1]
    l = 650.0/k
    image = transform.rescale(image,l)
    dic['scaleRate'] = l
    dic['image'] = image
    return dic

#截取灰度图image的感兴趣区域(left,top)左上角坐标到(right,bottom)
def cut_ROI(image,l,t,r,b):
    #上方区域
    for i in range(t):
        for j in range(image.shape[1]):
            image[i, j] = 0
    #下部区域
    for i in range(b, image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = 0
    #左侧区域
    for i in range(image.shape[0]):
        for j in range(l):
            image[i, j] = 0
    #右侧区域
    for i in range(image.shape[0]):
        for j in range(r,image.shape[1]):
            image[i, j] = 0
    return image

#将灰度图dst1做纵向均值滤波，并且根据结果做纵向对比度增强
#其中a为纵向均值滤波窗口的大小
#    b为做对比度增强时比较的窗口大小; f为增强的对比度大小
def filter(dst1,dst2, a , b , f):
    for j in range(5, dst1.shape[1] - 10):
        for i in range(a, dst1.shape[0] - a):
            p = 0.0
            for k in range(a):
                p += dst1[i + k, j]
            p /= a * 1.0
            dst2[i, j] = p
    if b==0:                        #返回只做均值滤波的结果
        return dst2
    for j in range(10, dst1.shape[1] - 10):
        for i in range(2 * b, dst1.shape[0] - 2 * b):
            if dst2[i, j] > dst2[i + b, j] and dst2[i, j] > dst2[i - b, j]:
                for k in range(b // 2):
                    dst1[i + k, j] = dst2[i, j] + f
    return dst1

#二值化处理,将灰度图像转为二值图像,阈值为 t
def binary_image(dst1,t):
    for i in range(dst1.shape[0]):
        for j in range(dst1.shape[1]):
            if (dst1[i, j] <= t):
                dst1[i, j] = 0
            else:
                dst1[i, j] = 1
    dst1 = img_as_bool(dst1)
    return dst1

#第一次定位出横向骨骼的大体位置,即 row
#根据每行白色像素点所占的比例和是否与边界相邻找出横向骨骼的大体位置
#rate为白色像素所占的比例
def round_horizontal_row(dst1,rate,width):
    row = 300
    for i in range(dst1.shape[0] - 60, 30, -1):
        p = 0  # 该行白色像素点所占比例
        flag = 0
        k = width
        for j in range(k):
            if dst1[i, j] == 1:
                p += 1
        for j in range(15):
            if dst1[i + j, 11] == 1:
                flag = 1
        if p > int(k*rate) and flag == 1:  # 找到该行
            row = i
            break
    return row

#用 cols 记录下横向骨骼的多个可能的右边界,取十一个值的最大值
def round_vertical_col(dst1,row):
    cols = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    k = 0
    for i in range(row - 30, row + 1, 3):
        for j in range(dst1.shape[1] // 8, dst1.shape[1] - 200, 4):
            if dst1[i, j] == 0:
                cols[k] = j
                break

        # 2018.4.3添加
        for j in range(dst1.shape[1] // 4, dst1.shape[1] - 200, 4):
            if dst1[i, j] == 0:
                cols[k] = j
                break

        k = k + 1
    cols.sort()
    #col=(cols[8]+cols[9]+cols[10])//3
    col = cols[10]
    return col

#用最小二乘法训练线性模型,Xi为横坐标训练数据,Yi为纵坐标训练数据
def train_linear_model(Xi,Yi):
    Xi = np.array(Xi)
    Yi = np.array(Yi)
    Xi = Xi.reshape(-1, 1)  #转置
    Yi = Yi.reshape(-1, 1)

    # 设置模型
    model = linear_model.LinearRegression()

    # 训练模型
    model.fit(Xi, Yi)

    # 用训练得的模型预测数据
    y_plot = model.predict(Xi)
    # 打印线性方程的权重
    # print(model.coef_)
    return y_plot

#求出两条线的夹角度数
def get_angle(x11,x12,y11,y12,x1,x2,y1,y2):
    x = np.array([x12 - x11, y12 - y11])
    y = np.array([x2 - x1, y2 - y1])
    # 求向量长度
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    # 求cos
    cos_angle = x.dot(y) / (Lx * Ly)
    # 求角度
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    #print(angle2)
    return angle2


#处理主函数.
#返回值为字典dict={'image':image,'scaleRate':scaleRate,'row':row,'col':col,'image_operated':image}
# 用来存储缩放后的图像/图片的缩放率/横向骨骼的右端点坐标/处理后的图像
def function(image):
    dict = {'image':image,'scaleRate':1,'row': 300,'col':300, 'image_operated':image,'angle':-1}

    #首先按比例将原图缩放,scaleRate为缩放率
    dic = {'image':image,'scaleRate':1}         #熟悉字典的用法及python函数传参机制
    rescale_image(dic)
    scaleRate = dic['scaleRate']
    dict['image'] = dic['image']
    image = dic['image']

    # float 转 unit8类型转换，切记以后遇到类型转换不能自己手动函数转,太容易出错
    image = img_as_ubyte(image)

    dst1 = color.rgb2gray(image)
    dst2 = color.rgb2gray(image)
    dst3 = color.rgb2gray(image)

    #plt.figure("test", figsize=(8, 8))
    plt.subplot(421)
    plt.imshow(dst1, plt.cm.gray)

    # 按纵向均值滤波作对比度增强, dst2存储的是做纵向均值处理后的灰度图，dst1存储的是对比度增强后的图像
    dst2 = filter(dst1, dst3, 10, 0,0)
    dst1 = filter(dst1,dst3,10,20,0.42)

    # 只留下感兴趣区域,将边界区域去掉
    dst1 = cut_ROI(dst1, 1, 1, dst1.shape[1] - 100, dst1.shape[0] - 180)

    plt.subplot(422)
    plt.imshow(dst2, plt.cm.gray)
    plt.subplot(423)
    plt.imshow(dst1, plt.cm.gray)

    # 二值化处理,阈值设为0.75,横向骨骼会很明显的显示出来
    dst1 = binary_image(dst1,0.75)

    # 去掉dst1中较小的连通域
    dst1 = morphology.remove_small_objects(dst1, min_size=100, connectivity=1)

    plt.subplot(424)
    plt.imshow(dst1, plt.cm.gray)


    # 找出横向骨骼的大体位置row1
    row1 = round_horizontal_row(dst1,0.65,160)    #初始值0.65 , 160
    #print("row1: "+str(row1))

    # 用 col1 记录下横向骨骼的右边界
    col1 = round_vertical_col(dst1,row1)
    #print("col1: "+str(col1))


    #plt.show()

    # 定位出横向骨骼，在原图 image 中画出
    Xi = []
    Yi = []
    # 添加点到训练模型
    for j in range(dst1.shape[1]//12, col1 - 20, 2):
        flag = 0
        for i in range(row1 - 10, row1 + 20, 1):
            if dst1[i, j] == 1 and flag == 0:
                flag = 1
            elif dst1[i, j] == 0 and flag == 0:
                break
            elif dst1[i, j] == 0 and flag == 1:
                Yi.append(i)
                Xi.append(j)
                break
    #print(len(Xi))
    y_plot = train_linear_model(Xi,Yi)

    Xi = np.array(Xi)
    Yi = np.array(Yi)
    Xi = Xi.reshape(-1, 1)  # 转置
    Yi = Yi.reshape(-1, 1)

    # 绘图(取训练好的模型的其中两个点连线)
    x2 = y_plot[len(y_plot) - 2, 0]
    y2 = Xi[len(Xi) - 2, 0]
    x1 = y_plot[0, 0]
    y1 = Xi[0, 0]

    #微调,使其接近与水平
    if x2-x1>3:
        x2-=5
    elif x1-x2>3:
        x1-=5

    x1 = int((x1-21))
    x2 = int((x2-21))
    y1 = int(y1)
    y2 = int(y2)
    #后面计算角度要用
    x11=x1
    x12=x2
    y11=y1
    y12=y2
    
    #延长并在原图image上绘制直线
    x2=2*(x2-x1)+x2
    y2=2*(y2-y1)+y2

    rr, cc = draw.line(x1-1, y1, x2-1, y2)
    draw.set_color(image, [rr, cc], [0, 255, 0])
    rr, cc = draw.line(x1, y1, x2, y2)
    draw.set_color(image, [rr, cc], [0, 255, 0])
    rr, cc = draw.line(x1+1, y1, x2+1, y2)
    draw.set_color(image, [rr, cc], [0, 255, 0])

    # 将纵向骨骼的区域做局部增强
    for i in range(row1 - 20, row1 + 150):
        for j in range(col1 - 30, col1 + 90):
            if dst2[i, j] > 0.32:
                dst2[i, j] = 1

    plt.subplot(425)
    plt.imshow(dst2, plt.cm.gray)

    # 二值化处理，使纵向骨骼显示出来
    dst2 = binary_image(dst2,0.9)

    # 将纵向骨骼的无关区域置零
    dst2 = cut_ROI(dst2,col1-30,row1-10,col1 + 90,row1 + 150)

    # 去掉dst1中较小的连通域
    #dst2 = morphology.remove_small_objects(dst2, min_size=50, connectivity=1)

    #plt.subplot(426)
    #plt.imshow(dst2, plt.cm.gray)
    #plt.subplot(428)
    #plt.imshow(dst2, plt.cm.gray)
    #plt.show()


    # 定位出纵向骨骼，在原图 image 中画出
    # 用最小二乘法拟合直线,用了linear_model来训练模型
    Xi=[]
    Yi=[]
    # 添加点,从上部取一些点,取右边界
    for i in range(row1-10 , row1 + 50, 2):
        flag = 0
        for j in range(col1 - 20, col1 + 85):
            if dst2[i, j] == 1 and flag == 0:
                #y1=j
                flag=1
            elif dst2[i, j] == 0 and flag == 1:
                #y1=(y1+j)//2
                y1=j
                Yi.append(y1)
                Xi.append(i)
                Yi.append(y1+1)
                Xi.append(i)
                break

    # 从下面取一些点,取右边界
    for i in range(row1 + 50, row1 + 120, 3):
        flag = 0
        for j in range(col1 - 20, col1 + 85):
            if dst2[i, j] == 1 and flag == 0:
                # y1 = j
                flag += 1
            elif dst2[i, j] == 0 and flag >= 1 and flag < 30:
                # y1 = (y1 + j) // 2
                Yi.append(j)
                Xi.append(i)
                break
    #print(len(Xi))
    y_plot = train_linear_model(Xi, Yi)

    Xi = np.array(Xi)
    Yi = np.array(Yi)
    Xi = Xi.reshape(-1, 1)  # 转置
    Yi = Yi.reshape(-1, 1)

    # 绘图(取训练好的模型的其中两个点连线)
    y2 = y_plot[len(y_plot) - 1, 0]
    x2 = Xi[len(Xi) - 1, 0]

    y1 = y_plot[2, 0]
    x1 = Xi[2, 0]

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    #延长
    x1=x1-(x2-x1)
    y1=y1-(y2-y1)
    x2=(x2-x1)//2+x2
    y2=(y2-y1)//2+y2

    rr, cc = draw.line(x1, y1-1, x2, y2-1)
    draw.set_color(image, [rr, cc], [255, 255, 0])
    rr, cc = draw.line(x1, y1, x2, y2)
    draw.set_color(image, [rr, cc], [255, 255, 0])
    rr, cc = draw.line(x1, y1+1, x2, y2+1)
    draw.set_color(image, [rr, cc], [255, 255, 0])
    #plt.scatter(Xi, Yi, color='red', label="样本数据", linewidth=2)
    #plt.plot(Xi, y_plot, color='green', label="拟合直线", linewidth=2)
    #plt.legend(loc='lower right')
    #plt.show()

    #求角度
    angle = get_angle(x11,x12,y11,y12,x1,x2,y1,y2)

    dict['scaleRate'] = scaleRate
    dict['row'] = row1
    dict['col'] = col1
    dict['image_operated'] = image
    dict['angle'] = angle

    return dict


if __name__ == '__main__':
    str1='/home/lucas/Lab/Project1/分割数据/image_cutted'
    files = os.listdir(str1)
    for file in files:
        print(file)
        image = io.imread(str1+'/'+file)
        dict = function(image)
        io.imsave('/home/lucas/Lab/Project1/result_tmp/' + file[0:-4] + '.jpg',dict['image_operated'])

'''
str2='/home/lucas/Lab/Project1/分割数据/image_cutted/7.jpg'
dict = function(str2)
io.imsave('/home/lucas/Lab/Project1/result_tmp/21.jpg',dict['image_operated'])
'''

