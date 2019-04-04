import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from data import dataProcess
from data import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.normalization import BatchNormalization
import numpy as np
import os
import glob
import time
import cv2
from keras.models import *
from keras.layers import Convolution2D
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.optimizers import *
from keras.layers import add, concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from PIL import Image
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.layers.core import Activation, Reshape, Permute
from mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from keras.layers import Lambda

'''
下采样:2 2 2 2 4 2 (感受野大致为128)
'''


class myFnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = 2

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def load_test_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_test = mydata.load_test_data()
        return imgs_test

    # 卷积———BN层———Relu操作
    def Conv2D_new(self, filters, kernel, padding, kernel_initializer, input):
        X = Conv2D(filters=filters, kernel_size=kernel, padding=padding, kernel_initializer=kernel_initializer)(input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        return X

    # 残余连接
    def Residual_block(self, conv1, conv2):
        net = add([conv1, conv2])
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        return net

    def get_Fnet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))
        print(inputs)

        kernel = 3
        padding = 'same'
        kernel_initializer = 'glorot_normal'
        pool_size = (2, 2)
        pool_size_4 = (4, 4)
        strides_4 = (4, 4)

        # encoding layers
        conv1_0 = self.Conv2D_new(64, 1, padding, kernel_initializer, inputs)
        conv1_1 = self.Conv2D_new(64, kernel, padding, kernel_initializer, conv1_0)
        conv1_2 = self.Conv2D_new(64, kernel, padding, kernel_initializer, conv1_1)
        res1 = self.Residual_block(conv1_0, conv1_2)
        pool1, arg1 = MaxPoolingWithArgmax2D(pool_size)(res1)
        print('pool1: ', pool1)
        print('arg1: ', arg1)

        conv2_0 = self.Conv2D_new(128, 1, padding, kernel_initializer, pool1)
        conv2_1 = self.Conv2D_new(128, kernel, padding, kernel_initializer, conv2_0)
        conv2_2 = self.Conv2D_new(128, kernel, padding, kernel_initializer, conv2_1)
        res2 = self.Residual_block(conv2_0, conv2_2)
        pool2, arg2 = MaxPoolingWithArgmax2D(pool_size)(res2)
        print('pool2: ', pool2)
        print('arg2: ', arg2)

        conv3_0 = self.Conv2D_new(256, 1, padding, kernel_initializer, pool2)
        conv3_1 = self.Conv2D_new(256, kernel, padding, kernel_initializer, conv3_0)
        conv3_2 = self.Conv2D_new(256, kernel, padding, kernel_initializer, conv3_1)
        res3 = self.Residual_block(conv3_0, conv3_2)
        pool3, arg3 = MaxPoolingWithArgmax2D(pool_size)(res3)
        print('pool3: ', pool3)
        print('arg3: ', arg3)

        conv4_0 = self.Conv2D_new(256, 1, padding, kernel_initializer, pool3)
        conv4_1 = self.Conv2D_new(256, kernel, padding, kernel_initializer, conv4_0)
        conv4_2 = self.Conv2D_new(256, kernel, padding, kernel_initializer, conv4_1)
        res4 = self.Residual_block(conv4_0, conv4_2)
        pool4, arg4 = MaxPoolingWithArgmax2D(pool_size)(res4)
        print('pool4: ', pool4)
        print('arg4: ', arg4)

        conv5_0 = self.Conv2D_new(512, 1, padding, kernel_initializer, pool4)
        conv5_1 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv5_0)
        conv5_2 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv5_1)
        res5 = self.Residual_block(conv5_0, conv5_2)
        pool5, arg5 = MaxPoolingWithArgmax2D(pool_size)(res5)
        print('pool5: ', pool5)
        print('arg5: ', arg5)

        conv6_0 = self.Conv2D_new(512, 1, padding, kernel_initializer, pool5)
        conv6_1 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv6_0)
        conv6_2 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv6_1)
        conv6_3 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv6_2)
        res6 = self.Residual_block(conv6_0, conv6_3)
        pool6, arg6 = MaxPoolingWithArgmax2D(pool_size)(res6)
        print('res6: ', res6)
        print('pool6: ', pool6)
        print('arg6: ', arg6)

        conv7 = self.Conv2D_new(512, 3, padding, kernel_initializer, pool6)
        conv7 = self.Conv2D_new(512, 3, padding, kernel_initializer, conv7)

        # decoding layers,先上采样，然后卷积
        up11 = MaxUnpooling2D(pool_size)([conv7, arg6])
        up11 = add([up11, res6])
        conv11_1 = self.Conv2D_new(512, kernel, padding, kernel_initializer, up11)
        conv11_2 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv11_1)
        conv11_3 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv11_2)
        res11 = self.Residual_block(up11, conv11_3)
        print('conv11_3: ', conv11_2)
        print('res11: ', res11)

        up12 = MaxUnpooling2D(pool_size)([res11, arg5])
        up12 = add([up12, res5])
        conv12_1 = self.Conv2D_new(512, kernel, padding, kernel_initializer, up12)
        conv12_2 = self.Conv2D_new(512, kernel, padding, kernel_initializer, conv12_1)
        res12 = self.Residual_block(up12, conv12_2)
        res12 = self.Conv2D_new(256, 1, padding, kernel_initializer, res12)
        print('conv12_2: ', conv12_2)

        up13 = MaxUnpooling2D(pool_size)([res12, arg4])
        up13 = add([up13, res4])
        conv13_1 = self.Conv2D_new(256, kernel, padding, kernel_initializer, up13)
        conv13_2 = self.Conv2D_new(256, kernel, padding, kernel_initializer, conv13_1)
        res13 = self.Residual_block(up13, conv13_2)
        print('conv13_2: ', conv13_2)

        up14 = MaxUnpooling2D(pool_size)([res13, arg3])
        up14 = add([up14, res3])
        conv14_1 = self.Conv2D_new(256, kernel, padding, kernel_initializer, up14)
        conv14_2 = self.Conv2D_new(256, kernel, padding, kernel_initializer, conv14_1)
        res14 = self.Residual_block(up14, conv14_2)
        res14 = self.Conv2D_new(128, 1, padding, kernel_initializer, res14)
        print('conv14_2: ', conv14_2)

        up15 = MaxUnpooling2D(pool_size)([res14, arg2])
        up15 = add([up15, res2])
        conv15_1 = self.Conv2D_new(128, kernel, padding, kernel_initializer, up15)
        conv15_2 = self.Conv2D_new(128, kernel, padding, kernel_initializer, conv15_1)
        res15 = self.Residual_block(up15, conv15_2)
        res15 = self.Conv2D_new(64, 1, padding, kernel_initializer, res15)
        print('conv15_2: ', conv15_2)

        up16 = MaxUnpooling2D(pool_size)([res15, arg1])
        up16 = add([up16, res1])
        conv16_1 = self.Conv2D_new(64, kernel, padding, kernel_initializer, up16)
        conv16_2 = self.Conv2D_new(64, kernel, padding, kernel_initializer, conv16_1)
        res16 = self.Residual_block(up16, conv16_2)

        conv17 = Conv2D(1, 1, activation='sigmoid')(res16)
        print('conv17: ', conv17)

        model = Model(input=inputs, output=conv17)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_Fnet()
        print("got segnet")

        model_checkpoint = ModelCheckpoint('Fnet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=12, verbose=1, validation_split=0.195,
                  shuffle=True,
                  callbacks=[model_checkpoint])

        # print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    def predict(self,image,model):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_test = mydata.creat_and_load_single_test_image_data(image)
        print("loading data done")

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        imgs = imgs_mask_test
        img = imgs[0]    #因为只有一张图像
        img = array_to_img(img)
        return img

    def load_model(self):
        model = self.get_Fnet()
        # 加载保存的模型权重
        model.load_weights('Fnet.hdf5')
        return model


    def use_saved_model_train(self):
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_Fnet()
        # 加载保存的模型权重
        model.load_weights('Fnet.hdf5')
        model_checkpoint = ModelCheckpoint('Fnet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=4, verbose=1, validation_split=0.05,
                  shuffle=True,
                  callbacks=[model_checkpoint])

    def save_single_img(self, num):
        print("array to image")
        imgs = np.load('predict/results1/imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("predict/results1/%d.jpg" % (num))

    # model可视化
    def draw_model(self):
        model = self.get_Fnet()
        plot_model(model, to_file='model_Fnet.png', show_shapes=True, show_layer_names=False)


if __name__ == '__main__':
    time1 = time.time()
    mysegnet = myFnet()
    #mysegnet.draw_model()
    #mysegnet.train()
    #mysegnet.use_saved_model_train()
    #mysegnet.save_img()


    time2 = time.time() - time1
    print("训练模型用了 " + str(time2 / 3600.) + " 小时")

    # 单张测试并保存结果
    #mysegnet.load_model_and_predict(0, 121)
