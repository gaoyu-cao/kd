#! /bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: 
 # @Date: 2021-06-17 03:35:11
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2021-08-11 07:13:17
### 
python3 main.py --batch_size=64 --gpu_id '0' --gpu_memory_fraction 0.8 --train_list  /mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/train_list_new.txt \
--val_list /mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/val_list_new.txt --temperature 30  --alpha 0.1 