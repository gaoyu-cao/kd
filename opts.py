'''
Author: cheng jie 
Date: 2021-01-08 13:41:59
LastEditTime: 2021-08-16 06:08:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /cj/Keras_Video_Recognization/opts.py
'''
import argparse

parser = argparse.ArgumentParser(description='Keras implementation of video action classification')

# ================================data config===============================

parser.add_argument('--train_list', type=str, default='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/train_list_new.txt')
parser.add_argument('--val_list', type=str, default='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/val_list_new.txt')
parser.add_argument('--test_list', type=str, default='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/test_list_new.txt')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--data_augmentation', action='store_true')

# ================================img config==============================
parser.add_argument('--crop_h', type=int, default=224)
parser.add_argument('--crop_w', type=int, default=224)
parser.add_argument('--head_img_h', type=int, default=128)
parser.add_argument('--head_img_w', type=int, default=128)
parser.add_argument('--screen_img_h', type=int, default=224)
parser.add_argument('--screen_img_w', type=int, default=128)

# ================================model config==============================
parser.add_argument('--use_summary', action='store_true')
parser.add_argument('--dataloader_mask', action='store_true')
parser.add_argument('--basemodel_name', type=str,default='mobilenetv2')
parser.add_argument('--screen_block_target_layer', type=str,default='block_5_depthwise_relu_2')
parser.add_argument('--head_block_target_layer', type=str,default='block_5_depthwise_relu_3')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dropout_ratdio', type=float, default=0.5)
parser.add_argument('--temperature', type=int, default=30)
parser.add_argument('--alpha', type=float, default=0.1)

#================================lr config=================================
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--base_lr', type=float, default=0.0001)
parser.add_argument('--use_lr_decay', action='store_true')
parser.add_argument('--lr_decay_epoch_period', type=int, default=10)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)

#===============================gpu config=================================
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--gpu_memory_fraction', type=float, default=0.9)

#===============================saving checkpoing and eventfile============
parser.add_argument('--save_model_root', type=str, default='/mnt/sda1/cgy/Keras_Knowledge_Distillation/checkpoints/Knowledge_Distillation')
parser.add_argument('--model_folder', type=str, default='train_model_v1')
parser.add_argument('--eventfiles', type=str, default='/mnt/sda1/cgy/Keras_Knowledge_Distillation/eventfiles/')
parser.add_argument('--log_path', type=str,default='/mnt/sda1/cgy/Keras_Knowledge_Distillation/log/')
parser.add_argument('--save_model_name', type=str, default='Keras_Screen_model_{:03d}.h5')

#===============================eval.py or test.py config============
parser.add_argument('--test_model_path', type=str, default='/mnt/sda1/cgy/Keras_Knowledge_Distillation/checkpoints/train_model_v1/epoch_020_acc_0.860303_loss_1.089444')
parser.add_argument('--saved_model_path', type=str, default='/mnt/sda1/cgy/Keras_Knowledge_Distillation/saved_model')
parser.add_argument('--version_num', type=str, default='5')
