'''
Descripttion: 
version: 
Author: 
Date: 2021-05-19 09:06:27
LastEditors: Please set LastEditors
LastEditTime: 2021-08-13 01:42:48
'''
# 测试模型
import pandas as pd
import os
import shutil
import json
import sys

# anno_label_path = '/mnt/sda2/Public_Data/Data_lookscreen/test_dataset/test_list0606.txt'   # 标注的test_label.txt
anno_label_path = '/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/test_list_new.txt'
score_txt_path = '/mnt/sda1/cgy/Keras_Knowledge_Distillation/pred_out/data_list_20210319/keras_watch_screen_model_epoch_030_acc_0.8313_loss_0.139.csv'
# score_txt_path = '/mnt/sda2/cj/Keras_Knowledge_Distillation/pred_out/test_dataset/keras_watch_screen_model_epoch_030_acc_0.8380_loss_0.058.csv'   # 测试结果*scores.txt


roc_root = '/mnt/sda1/cgy/Keras_Knowledge_Distillation/roc_result'
res_saved_path = score_txt_path.split('/')[-1][:-4] + '_roc.txt'      # 保存roc结果*res.txt
res_saved_full_path = os.path.join(roc_root, res_saved_path)
# os.makedirs(res_saved_full_path, exist_ok=True)
# 包含正负样本的label.txt文件:0/1
anno_data = pd.read_csv(anno_label_path, sep=',', names=['head_name','fullimage_path','label','sc_x1','sc_y1','sc_x2','sc_y2',
                                                        'hc_x1','hc_y1','hc_x2','hc_y2'])
# anno_data['npz_name'] = anno_data.pic_name.apply(lambda x:x[:-4]+'.npz')

pos = anno_data[anno_data.label==1].head_name.tolist()
neg = anno_data[anno_data.label==0].head_name.tolist()
data = pos + neg
print(len(data),len(pos),len(neg))
print(data[:5])

# 测试总样本/正样本/负样本:val_pics/val_pos/val_neg
# df = pd.read_csv(score_txt_path, sep=',', names=['head_name','negative_score','postive_score'])
df = pd.read_csv(score_txt_path)
df['head_name'] = df.path.apply(lambda x:x)
df = df[df.head_name.isin(data)]
# df['score_0'] = df['score_0'].apply(float)                                    
# df['score_1'] = df['score_0'].apply(lambda x:1-x)
val_pics = df['head_name'].tolist()
print(df.shape)
log = open(res_saved_full_path,'a')
sys.stdout = log
print('threshold accuracy precision recall FPR')
thresholds = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
for threshold in thresholds:
    val_pos = df.loc[df['postive_score'] >= threshold].head_name.tolist()
    # val_pos = df.loc[df['score_1'] >= threshold].head_name.tolist()
    val_neg = set(val_pics)-set(val_pos)
    pos = set(pos)&set(val_pics)            #保证pos为所有测试样本中的正样本
    neg = set(neg)&set(val_pics)
    TP = set(pos)&set(val_pos)
    FP = set(neg)&set(val_pos)
    TN = set(neg)&set(val_neg)

    accuracy = (len(TP)+len(TN))/len(val_pics)  #准确度
    precision = len(TP)/len(val_pos)        #精确度
    recall = len(TP)/len(set(pos))          #真阳率/召回率/查全率
    FPR = len(FP)/len(set(neg))         #假阳率
    print(threshold,accuracy,precision,recall,FPR)

log.close()
