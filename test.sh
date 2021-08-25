#! /bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: 
 # @Date: 2021-07-15 07:23:03
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2021-08-11 07:12:42
### 

# --eval_list=/mnt/sda2/cj/headscreen_cls/data_list_20210222/test_list_new.txt
python3 test.py --gpu_id=0 --batch_size=128 --gpu_memory_fraction=0.8 \
--test_model_path=/mnt/sda2/cj/Keras_Knowledge_Distillation/checkpoints/train_model_TEMPERATURE_30_Teacher_train_model_2021-07-14-08-00-04/train_model_2021-07-16-07-12-58/keras_watch_screen_model_epoch_030_acc_0.8380_loss_0.0583.h5 \
--test_list=/mnt/sda2/Public_Data/Data_lookscreen/test_dataset/test_list0606.txt