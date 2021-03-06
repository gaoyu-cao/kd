'''
Descripttion: 
version: 
Author: 
Date: 2021-08-11 07:11:45
LastEditors: Please set LastEditors
LastEditTime: 2021-08-13 01:35:55
'''
'''
Descripttion: test.py, using savedmodel to inference 
version: v1
Author: cheng jie
Date: 2021-05-07 06:51:51
LastEditors: Please set LastEditors
LastEditTime: 2021-07-16 02:27:58
'''

import tensorflow as tf
from tensorflow.python.keras.saving.saved_model.load import load

from opts import parser
from dataloader_mask import Dataloader

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import os


class LayerFromSavedModel(tf.keras.layers.Layer):
    def __init__(self, loaded):
        '''
        @description: 
        @param {*}
        @return {*}
        '''
        self.loaded = loaded
        self.vars = loaded.variables

    def __call__(self, inputs):
        return self.loaded.signatures["serving_default"](inputs)


def test(args):
    num_classes = args.num_classes
    if args.dataloader_mask:
        print('dataloader type is mask')
        from dataloader_mask import Dataloader

    else:
        from dataloader import Dataloader
    test_loader = Dataloader(batch_size=args.batch_size, data_list=args.test_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w, shuffle=False, model='test')

    model_path = args.test_model_path
    model = tf.keras.models.load_model(model_path)
    # imported = tf.saved_model.load(model_path)
    
    print("load keras model from '{}' successful".format(model_path))
    # model.summary()
    # imported.summary()

    pred_class = []
    label_list = []
    sample_path_list = []
    postive_score = []
    negative_score = []

    for head_path, data, label in tqdm(test_loader):
        sample_path_list.extend(head_path)
        label = np.argmax(label, axis=-1).tolist()
        label_list.extend(label)
        result = model.predict(data)
        pred_class.extend(np.argmax(result, axis=-1).tolist())
        postive_score.extend(result[:,1].tolist())
        negative_score.extend(result[:,0].tolist())
    csv_data  = [[sample_path_list[i], label_list[i], pred_class[i], postive_score[i], negative_score[i]] for i in range(len(sample_path_list))]
    columns = ['path', 'label', 'pred_label', 'postive_score', 'negative_score']
    df = pd.DataFrame(data=csv_data, columns=columns)
    # '/mnt/sda2/cj/TF2_keras_watch_screen/pred_out/train_model_v5/test_result.csv'
    ## ======================= save pred csv =================
    saved_root = '/mnt/sda1/cgy/Keras_Knowledge_Distillation/pred_out/'
    
    test_data_path = args.test_list.split('/')[-2]
    saved_path = os.path.join(saved_root, test_data_path)
    os.makedirs(saved_path, exist_ok=True)
    save_csv_path = args.test_model_path.split('/')[-1][:-4]+'.csv'
    saved_full_path = os.path.join(saved_path, save_csv_path)
   
    df.to_csv(saved_full_path)
    pred_class = np.array(pred_class)
    label_true = np.array(label_list)
    postive_score = np.array(postive_score)
    negative_score = np.array(negative_score)
    target_names = ['Not_watch_screen','watch_screen']
    print(metrics.classification_report(label_true, pred_class, target_names=target_names))
    
        
    fpr, tpr, thresholds = metrics.roc_curve(label_true, postive_score, pos_label=1) 
    
    print('fpr: ' + str(fpr))
    print('tpr: ' + str(tpr))
    print('thresholds: ' + str(thresholds))

    AUC = metrics.auc(fpr, tpr) 
    print('AUC:{}'.format(AUC))
    plt.figure()
    plt.title('ROC Curve',fontweight='bold')
    plt.plot(fpr, tpr, color='darkorange',
                lw=2, label='ROC curve (area = %0.3f)' % AUC) ###??????????????????????????????????????????????????????
    plt.xlabel('False Positivve Rate',fontweight='bold')
    plt.ylabel('True Positivve Rate',fontweight='bold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    # save_csv = args.test_model_path.split('/')[-2]+'_epoch_{}'.format(epoch_id)+'_roc.png'
    # save_csv_full_path = os.path.join(saved_path, save_csv)
    # plt.savefig(save_csv_full_path)
    
    # plt.plot(fpr, tpr)  
    # plt.title('ROC_curve' + '(AUC: ' + str(AUC) + ')' )  
    # plt.ylabel('True Positive Rate')  
    # plt.xlabel('False Positive Rate') 
     
    # plt.savefig('/mnt/sda1/cj/Keras_Knowledge_Distillation/pred_out/test_auc.png') 


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    sess = tf.compat.v1.Session(config=config)
    print(("""
        GPU Congfig
        CUDA_VISIBLE_DEVICES :              {}
        PER_PROCESS_GPU_MEMORY_FRACTION     {}
    """).format(args.gpu_id, args.gpu_memory_fraction))
   
    test(args)


