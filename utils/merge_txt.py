'''
Descripttion: 
version: 
Author: 
Date: 2021-08-11 03:06:09
LastEditors: Please set LastEditors
LastEditTime: 2021-08-11 03:06:09
'''
from tqdm import tqdm 


# txt_path_1 = '/mnt/sda2/cj/headscreen_cls/data_list3/train_list3.txt'
# txt_path_2 = '/mnt/sda2/cj/headscreen_cls/data_list_20210319/train_list.txt'
txt_path_1 = '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/positive.txt'
txt_path_2 = '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/negative_2.txt'
saved_path = '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/train_KD.txt'

data_1 = data_2 = ''

with open(txt_path_1, 'r') as f1:
    data_1 = f1.read()


with open(txt_path_2, 'r') as f2:
    data_2 = f2.read()

data_1 += "\n"
data_1 += data_2 


# print('new data1 lentgh {}'.format(len(data_1)))
with open (saved_path, 'w') as fp: 
    fp.write(data_1) 
