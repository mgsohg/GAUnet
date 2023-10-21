import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import random
import os
from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import time
from keras.callbacks import CSVLogger
from model_net import *
import keras
import tensorflow as tf
import tensorflow.keras.backend as K

path = ''
#
# path = './BUSI/CV_all/new/4/'
# path = './Thyroid Dataset/DDTI dataset/CV_all2/2/'


def load_img(path, grayscale=False):

    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)/255.0
        #print(np.min(img),print(np.max(img)))
    else:
        img = cv2.imread(path)
        image = np.array(img, dtype=np.uint8)
        h, w, c = image.shape
        padding_h = h
        padding_w = w

        padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
        padding_img = padding_img.astype("float") / 255.0
        img = img_to_array(padding_img)
        #print(np.min(img), print(np.max(img)))
    return img

def get_train_data():
    train_url = []
    train_set = []
    for pic in os.listdir(path + '/train/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    print(total_num)
    for i in range(len(train_url)):
        train_set.append(train_url[i])

    return train_set

def get_val_data():
    train_url = []
    train_set = []
    for pic in os.listdir(path + '/val/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    print(total_num)
    for i in range(len(train_url)):
        train_set.append(train_url[i])

    return train_set

def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(path + '/train/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(path + '/train_annot/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                # yield ([train_data, train_label], train_label)
                train_data = []
                train_label = []
                batch = 0

def generateValidData(batch_size, data=[]):
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(path + '/val/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(path + '/val_annot/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                # yield ([valid_data, valid_label], valid_label)
                valid_data = []
                valid_label = []
                batch = 0

def iou(y_true, y_pred, smooth = 1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true)+K.sum(y_pred)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true,y_pred):
  return 1.0 - dice_coef(y_true,y_pred)

def BCE():
    def bce(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    return bce

def BCE_Dice():
    def bcedice(y_true, y_pred):
        return 0.5*tf.keras.metrics.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return bcedice

def Dice():
    def dice(y_true, y_pred):
        return dice_loss(y_true, y_pred)
    return dice




EPOCHS = 50
BS = 12
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# physical_devices = tf.config.experimental.list_physical_devices('CPU')

print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_set = get_train_data()
val_set = get_val_data()
train_numb = len(train_set)
valid_numb = len(val_set)

print("the number of train data is", train_numb)
print("the number of val data is", valid_numb)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

optim = keras.optimizers.Adam(0.001)

model = GLAN()
# model.load_weights('./epoch2/model_022.hdf5')
model.summary()

#tf.keras.utils.plot_model(model, './custom_plot.png', show_shapes=True,dpi=42)

# #GLAN

model.compile(loss={'Fusion':BCE(),'GB':BCE(),'SRF':BCE(),'LRF':BCE(),'Side2':BCE(),'Side3':BCE(),'Side4':BCE()},
              loss_weights={'Fusion':1.0,'GB':1.0,'SRF':1.0,'LRF':1.0,'Side2':1.0,'Side3':1.0,'Side4':1.0},optimizer=optim,metrics=[dice_coef,iou])
checkpointer = ModelCheckpoint(os.path.join('./epoch2/', 'model_{epoch:03d}.hdf5'), monitor='accuracy', save_best_only=False, mode='max')

since = time.time()

flops = try_count_flops(model)
print(flops/1000000000,"GFlops")

csv_logger = CSVLogger('log.csv', append=True, separator=';')

H = model.fit_generator(generator=(generateData(BS,train_set)),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,
                validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=[checkpointer,csv_logger])

time_elapsed = time.time() - since
print('Training fininshed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
