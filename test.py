"""
处理图片，将其序号统一
"""

import re
import os
import time

def change_name(path):
    global i
    global j
    if not os.path.isdir(path) and not os.path.isfile(path):
        return False
    if os.path.isfile(path):
        file_path = os.path.split(path)  #分割出目录与文件
        lists = file_path[1].split('.')  #分割出文件与文件扩展名
        if lists[0][-1]=='1' or lists[0][-1]=='2' or lists[0][-1]=='3' or lists[0][-1]=='4' or lists[0][-1]=='5' or lists[0][-1]=='6' or lists[0][-1]=='7':       #原图
            i += 1
            os.rename(path, file_path[0]+'/'+str(i)+ '.jpg')
            #根据前缀名字一样找出标注图
            for x in os.listdir(img_dir):
                if x[0] == lists[0][0] and x[1] == lists[0][1] and x[2] == lists[0][2] and x[len(lists[0])-1] == lists[0][-1]:
                    if x[-5] == '角':
                        os.rename(img_dir+x, file_path[0]+'/'+str(i)+'_1'+ '.jpg')    #角度标注图
                    elif x[-5] == '头':
                        os.rename(img_dir + x, file_path[0] + '/' + str(i) + '_2' + '.jpg')    #分割标注图

    elif os.path.isdir(path):
        for x in os.listdir(path):    #'连接符'.join(list) 将列表组成字符串
            change_name(os.path.join(path,x))


if __name__ == '__main__':
    img_dir = '/home/lucas/桌面/6.21数据/all_data/'
    start = time.time()
    i=0
    j=0
    change_name(img_dir)
    c = time.time()-start
    print('程序运行耗时：%0.2f' %(c))
    print('总共处理了%s张图片' %(i+1))