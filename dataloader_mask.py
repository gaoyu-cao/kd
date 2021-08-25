'''
Descripttion: 
version: 
Author: 
Date: 2021-08-11 03:06:53
LastEditors: Please set LastEditors
LastEditTime: 2021-08-16 03:16:32
'''
from hashlib import new
import os
import re
import sys
import cv2
from cv2 import data
from matplotlib.pyplot import angle_spectrum
import numpy as np
import random
from numpy.core.numeric import full
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataRecord(object):
    '''
    Processing txt files
    '''

    def __init__(self, row):
        self._data = row

    @property 
    def head_images_path(self):
        return self._data[0]

    @property
    def full_image_path(self):
        return self._data[1]

    @property
    def label(self):
        return int(self._data[2])

    @property
    def sc(self):
        return [int(i) for i in self._data[3:7]]

    @property
    def hc(self):
        return [int(i) for i in self._data[7:]]

def process_image(img):
    img = img.astype('float32')
    img = img[:, :, (2, 1, 0)]      # rgb -> bgr
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    '''
    @description: augement img hsv
    @para:   {*}
    @return: {*}
    '''
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1         # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype       # uint8
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, data_list, num_class, crop_h, crop_w, head_img_h, head_img_w, screen_img_h, screen_img_w,
                    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, model='train', shuffle=True):
                    self.batch_size = batch_size
                    self.data_list = data_list
                    self.num_class = num_class
                    self.crop_h = crop_h
                    self.crop_w = crop_w
                    self.head_img_h = head_img_h
                    self.head_img_w = head_img_w
                    self.screen_img_h = screen_img_h
                    self.screen_img_w = screen_img_w
                    self.hsv_h = hsv_h
                    self.hsv_s = hsv_s
                    self.hsv_v = hsv_v
                    self.model = model
                    self.shuffle = shuffle

                    self._parse_list()
                    self.on_epoch_end()
                    self.max = len(self.video_list) // self.batch_size

    def _load_image(self, full_image_path):
        fullimage = cv2.imread(full_image_path)
        fullimage = cv2.resize(fullimage, (720, 1280))
        
        if self.model == 'train':
            augment_hsv(fullimage, self.hsv_h, self.hsv_s, self.hsv_v)
            fullimage = fullimage.astype('uint8')
            fullimage = self._data_augmentation(fullimage)
        return fullimage

    def _data_augmentation(self, img):
        '''
        args:
            img: 3D numpy tensor, tensor dtype must uint8
        return:
            augmentation data
        '''
        img = tf.image.random_jpeg_quality(img, 30, 70)
        return img.numpy()

    def img_aug(self, new_crop, screen, head):
        data_gen = ImageDataGenerator()
        dic_parameter = {'flip_horizontal': random.choice([True, False]),
                        #  'flip_vertical': random.choice([True, False]),
                        #  'theta': random.choice([0, 0, 0, 90, 180, 270])
                        }

        new_crop = data_gen.apply_transform(new_crop, transform_parameters=dic_parameter)
        screen = data_gen.apply_transform(screen, transform_parameters=dic_parameter)
        head = data_gen.apply_transform(head, transform_parameters=dic_parameter)
        
        # cv2.imwrite('new_crop_aug.jpg', new_crop)
        # cv2.imwrite('head_aug.jpg', head)
        # cv2.imwrite('screen_aug.jpg', screen)

        return new_crop, screen, head
    
    def _rotate_images(self, image, angle):
        h, w = image.shape[:2]
        cX, cY = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cY, cX), -angle, 0.75)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        return cv2.warpAffine(image, M, (nW, nH), borderValue=(0, 0, 0))


    def __getitem__(self, index):
        left_index = index * self.batch_size
        right_index = (index + 1) * self.batch_size
        indexes = self.indexes[left_index:right_index]
        record_list = [self.video_list[k] for k in indexes]
        return self.data_generate(record_list)
    
    def data_generate(self, record_list):
        batch_head_images = np.empty((self.batch_size, self.head_img_h, self.head_img_w, 3), dtype=np.float32)
        batch_screen_images = np.empty((self.batch_size, self.screen_img_h, self.screen_img_w, 3), dtype=np.float32)
        batch_crop_images = np.empty((self.batch_size, self.crop_h, self.crop_w, 3), dtype=np.float32)
        batch_anno = np.empty((self.batch_size), dtype=np.int)
        batch_path = []

        for label_id, record in enumerate(record_list):
            new_crop, screen, head = self.get(record)
            batch_anno[label_id] = record.label
            batch_crop_images[label_id,] = new_crop
            batch_head_images[label_id,] = head
            batch_screen_images[label_id,] = screen
            batch_path.append(record.head_images_path)
        batch_anno = tf.keras.utils.to_categorical(batch_anno, num_classes=self.num_class)
        if self.model == 'train' or self.model =='eval':
            return [batch_crop_images, batch_screen_images, batch_head_images, batch_anno], batch_anno
        else:
            return batch_path, [batch_crop_images, batch_screen_images, batch_head_images], batch_anno
                
    def get(self, record):
        fullimage = self._load_image(record.full_image_path)

        sc_xmin = record.sc[0]
        sc_ymin = record.sc[1]
        sc_xmax = record.sc[2]
        sc_ymax = record.sc[3]

        hc_xmin = record.hc[0]
        hc_ymin = record.hc[1]
        hc_xmax = record.hc[2]
        hc_ymax = record.hc[3]

        new_xmin = min(sc_xmin, hc_xmin)
        new_ymin = min(sc_ymin, hc_ymin)
        new_xmax = max(sc_xmax, hc_xmax)
        new_ymax = max(sc_ymax, hc_ymax)
        
        img_copy = np.zeros(np.shape(fullimage)[:2], dtype=np.uint8)
        img_copy[hc_ymin:hc_ymax, hc_xmin:hc_xmax] = 255
        img_copy[sc_ymin:sc_ymax, sc_xmin:sc_xmax] = 255
        image_mask = cv2.add(fullimage, np.zeros(np.shape(fullimage), dtype=np.uint8), mask=img_copy)
        new_crop = image_mask[new_ymin:new_ymax, new_xmin:new_xmax]
        new_crop = process_image(new_crop)

        head = fullimage[hc_ymin:hc_ymax, hc_xmin:hc_xmax]
        head = process_image(head)

        screen = fullimage[sc_ymin:sc_ymax, sc_xmin:sc_xmax]
        screen = process_image(screen)

        new_crop = cv2.resize(new_crop, (self.crop_w, self.crop_h))
        head = cv2.resize(head, (self.head_img_w, self.head_img_h))
        screen = cv2.resize(screen, (self.screen_img_w, self.screen_img_h))

        if self.model == 'train':
            angle = random.randint(0,20)
            # new_crop, screen, head = self._horizontal_flip(new_crop, screen, head)
            new_crop, screen, head = self.img_aug(new_crop, screen, head)

            return  new_crop, screen, head
            
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _parse_list(self):
        self.video_list = [DataRecord(x.strip().split(',')) for x in open(self.data_list)]

    def __len__(self):
        return len(self.video_list) // self.batch_size


if __name__ == '__main__':
    # data_list='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list3/val_list.txt'
    data_list = '/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/train_list_new.txt'
    data_loader = Dataloader(batch_size=32, data_list=data_list, num_class=2, crop_h=224, crop_w=224,
                            head_img_h=96, head_img_w=96, screen_img_h=224, screen_img_w=128, shuffle=True, model='train')
    for i, (data, label) in enumerate(data_loader):
        print(i, data[0].shape, data[1].shape, data[2].shape, label.shape)