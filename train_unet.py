import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from data import dataProcess
from data import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D , concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from PIL import Image
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
import time


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

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
    def Conv2D_new(self,filters,kernel,padding,kernel_initializer,input):
        X = Conv2D(filters=filters,kernel_size=kernel,padding=padding,kernel_initializer=kernel_initializer)(input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        return X

    def get_unet(self):

        kernel_initializer = 'glorot_normal'

        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=16, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

        #print('predict test data')
        #imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        #np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    def use_saved_model_train(self):
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        #加载保存的模型权重
        model.load_weights('unet.hdf5')
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1, validation_split=0.195,
                  shuffle=True,
                  callbacks=[model_checkpoint])

    def use_saved_model(self,num):
        mydata = dataProcess(self.img_rows, self.img_cols)
        mydata.create_single_test_data(num)
        #mydata.create_test_data()
        imgs_test = self.load_test_data()
        print("loading data done")
        model = self.get_unet()
        #加载保存的模型权重
        model.load_weights('unet.hdf5')
        # compile 编译
        #model.compile(metrics=['accuracy'])

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('../results/imgs_mask_test.npy', imgs_mask_test)

    def load_model_and_predict(self,st_num,end_num):
        model = self.get_unet()
        # 加载保存的模型权重
        model.load_weights('unet.hdf5')
        for i in range(st_num,end_num):
            mydata = dataProcess(self.img_rows, self.img_cols)
            mydata.create_single_test_data(i)
            # mydata.create_test_data()
            imgs_test = self.load_test_data()
            print("loading data done")

            print('predict test data')
            imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
            np.save('../results/imgs_mask_test.npy', imgs_mask_test)
            self.save_single_img(i)

    def save_img(self):
        print("array to image")
        imgs = np.load('../results/imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("../results/%d.jpg" % (i))

    def save_single_img(self,num):
        print("array to image")
        imgs = np.load('../results/imgs_mask_test.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save("../results/%d.tif" % (num))

    # model可视化
    def draw_model(self):
        model = self.get_unet()
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)



if __name__ == '__main__':
    time1 = time.time()
    myunet = myUnet()
    #myunet.train()
    #myunet.use_saved_model_train()
    #myunet.save_img()
    #myunet.draw_model()

    time2 = time.time() - time1
    print("训练模型用了 " + str(time2 / 3600.) + " 小时")

    # 单张测试并保存结果
    myunet.load_model_and_predict(0,121)
