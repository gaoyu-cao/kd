'''
Descripttion: 
version: 
Author: 
Date: 2021-08-13 03:00:14
LastEditors: Please set LastEditors
LastEditTime: 2021-08-13 07:11:51
'''
import cv2
import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from math import *
import numpy as np
 
# a=random.randint(0,20)
# # 旋转angle角度，缺失背景白色（255, 255, 255）填充
# def rotate_bound_white_bg(image, angle):
#     # grab the dimensions of the image and then determine the
#     # center
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
 
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
 
#     # compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
 
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
 
#     # perform the actual rotation and return the image
#     # borderValue 缺失背景填充色彩，此处为白色，可自定义
#     return cv2.warpAffine(image, M, (nW, nH),borderValue=(0,0,0))
#     # borderValue 缺省，默认是黑色（0, 0 , 0）
#     # return cv2.warpAffine(image, M, (nW, nH))

# head = cv2.imread('/mnt/sda1/cgy/head.jpg')
# screen = cv2.imread('/mnt/sda1/cgy/screen.jpg')
# new_crop = cv2.imread('/mnt/sda1/cgy/new_crop.jpg')
# imgRotation = rotate_bound_white_bg(new_crop, a)
 
# cv2.imwrite("img.png",new_crop)
# cv2.imwrite("imgRotation.png",imgRotation)
# # cv2.waitKey(0)


# # head = cv2.imread('/mnt/sda1/cgy/head.jpg')

# data_gen = ImageDataGenerator()
# dic_parameter = {   'flip_horizontal': random.choice([True, False]),
#                     'flip_vertical': random.choice([True, False]),
#                     # 'theta': random.choice([90, 150, 270])
#                     'theta': a
#                 }

# new_crop = data_gen.apply_transform(new_crop, transform_parameters=dic_parameter)
# screen = data_gen.apply_transform(screen, transform_parameters=dic_parameter)
# head = data_gen.apply_transform(head, transform_parameters=dic_parameter)
# cv2.imwrite('new_crop1.jpg', new_crop)
# cv2.imwrite('head1.jpg', head)
# cv2.imwrite('screen1.jpg', screen)
# # 
# print(dic_parameter)


import numpy as np
import random
import cv2

def sp_noise(image,prob):
  '''
  添加椒盐噪声
  prob:噪声比例 
  '''
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output


def gasuss_noise(image, mean=0, var=0.001):
  ''' 
    添加高斯噪声
    mean : 均值 
    var : 方差
  '''
  image = np.array(image/255, dtype=float)
  noise = np.random.normal(mean, var ** 0.5, image.shape)
  out = image + noise
  if out.min() < 0:
    low_clip = -1.
  else:
    low_clip = 0.
  out = np.clip(out, low_clip, 1.0)
  out = np.uint8(out*255)
#   cv2.imwrite("gasuss.jpg", out)
  return out

head = cv2.imread('/mnt/sda1/cgy/head.jpg')
head = sp_noise(head, 0.01)
head = gasuss_noise(head)
cv2.imwrite('head2.jpg', head)
