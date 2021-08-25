'''
Descripttion: 
version: 1.0
Author: cheng jie 
Date: 2021-04-28 02:22:03
LastEditors: Please set LastEditors
LastEditTime: 2021-08-12 07:53:25
'''
import tensorflow as tf
# from dataloader_mask import Dataloader

# from dataloader import Dataloader
from opts import parser

import os
import time
import numpy as np
from tqdm import tqdm 

TEMPERATURE = 30
ALPHA = 0.1
NUM_CLASSES = 2

def get_teacher_model(teacher_new_weights=None):
    '''
    @description: KD loading trained teacher model which final layer is Activation
    @param 
        teacher_new_weights: keras model path 
    @return:
        keras model
    '''
    if teacher_new_weights.split('/')[-1].startswith('keras'):
        teacher_model = tf.keras.models.load_model(teacher_new_weights)
        print("==============================================================")
        print('load keras model successful :{}'.format(teacher_new_weights))
        print("==============================================================")
    return teacher_model


def knowledge_distillation_loss(input_distillation):
    '''
    @description: compute kd loss
    @param 
        param1:input_distillation ex：`y_pred, y_true, y_soft, y_pred_soft`
    @return {kd loss }
    '''
    alpha = ALPHA 
    temperature = TEMPERATURE
    y_pred, y_true, y_soft, y_pred_soft = input_distillation
    hard_target_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    soft_target_loss = tf.keras.losses.kullback_leibler_divergence(y_soft, y_pred_soft)
    total_loss =(1-alpha)*hard_target_loss + alpha * temperature * temperature * soft_target_loss
    # print('total_loss', total_loss.numpy())
    return total_loss


class BornAgainModel():
    '''
    @description: 
    @param 
        
    @return 

    '''
    def __init__(self, teacher_model, temperature, target_layer, num_classes):
       
    # input_layer : [crop, screen, head]
        self.train_model, self.born_again_model = None, None
        self.input_layer = teacher_model.input

        for i in range(len(teacher_model.layers)):
            teacher_model.layers[i].trainable = False
        
        self.target_layer = target_layer
        self.num_classes = num_classes
        self.temperature = temperature
        self.train_model, self.born_again_model = self.make_student_model()

    
    def make_student_model(self):
        '''
        @description: build student model
        @param {*}
        @return {*}
        '''
        teacher_logits = teacher_model.get_layer(index=-2).output
        teacher_logits_T = tf.keras.layers.Lambda(lambda x: x/self.temperature)(teacher_logits)
        teacher_probabilities_T = tf.keras.layers.Activation('softmax', name='teacher_model_softmax')(teacher_logits_T)
        
        base_model_crop = tf.keras.applications.MobileNetV2(input_tensor=self.input_layer[0],weights='imagenet', include_top=False)
        
        for layer in base_model_crop.layers:
            layer._name = layer.name + '_crop_layer'
        base_crop_out = base_model_crop.output
        
        # 控制crop图像的下采样'block_5_depthwise_relu'
        # base_crop_out = base_model_crop.get_layer('block_15_depthwise_relu_crop_layer').output
        crop_out = tf.keras.layers.GlobalAveragePooling2D(name='crop_global_pooling')(base_crop_out) 
        crop_out = tf.keras.layers.Dense(512, activation='relu',name='crop_dense_stu')(crop_out) 
        
        # build screen and head layer
        base_model_screen = tf.keras.applications.MobileNetV2(input_tensor=self.input_layer[1], weights='imagenet', include_top=False) 
        for layer in base_model_screen.layers:
            layer._name = layer.name + '_screen_layer'

        base_model_head = tf.keras.applications.MobileNetV2(input_tensor=self.input_layer[2], weights='imagenet', include_top=False) 
        for layer in base_model_head.layers:
            layer._name = layer.name + '_head_layer'

        screen_block_target_layer=self.target_layer + '_screen_layer'
        head_block_target_layer=self.target_layer + '_head_layer'
        
        screen_out = base_model_screen.get_layer(screen_block_target_layer).output
        screen_out = tf.keras.layers.GlobalAveragePooling2D(name='screen_global_pooling')(screen_out)

        head_out = base_model_head.get_layer(head_block_target_layer).output
        head_out = tf.keras.layers.GlobalAveragePooling2D(name='head_global_pooling')(head_out) 

        # concate the feature map of screen and head out
        concate_out = tf.keras.layers.Concatenate(name='screen_head')([screen_out, head_out])
        concate_out = tf.keras.layers.Dense(256, activation='relu',name='cat_dense_1_stu')(concate_out)
        
        final_out = tf.keras.layers.Concatenate(name='final_concate')([concate_out, crop_out])
        final_out = tf.keras.layers.Dense(256,  activation='relu',name='fc')(final_out)
        
        student_logits = tf.keras.layers.Dense(self.num_classes, name='classifier_dense')(final_out)
        student_probabilies = tf.keras.layers.Activation('softmax', name='student_softmax')(student_logits)
        student_logits_T = tf.keras.layers.Lambda(lambda x : x / self.temperature, name='student_logits_t')(student_logits)
        student_probabilies_T = tf.keras.layers.Activation('softmax', name='student_probabilies_T')(student_logits_T)
        born_again_model =tf.keras.Model(inputs=self.input_layer, outputs=student_probabilies)
        # born_again_model.summary()
        # # build 
        input_true = tf.keras.Input(name='input_true',shape=[None], dtype='float32')
        kd_loss = tf.keras.layers.Lambda(knowledge_distillation_loss, name='kd_loss', output_shape=(1,))([student_probabilies, input_true, teacher_probabilities_T, student_probabilies_T])
        # # print(kd_loss)
        train_model = tf.keras.Model(inputs=[self.input_layer[0],self.input_layer[1], self.input_layer[2], input_true], outputs=kd_loss)

        return train_model, born_again_model

    def evaluate(self, dataloader):
        '''
        @description: 
        @param :
            dataloader: data generator
            
        @return : metrics
        '''
        # metrics = self.born_again_model.evaluate(dataloader)
        self.born_again_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
        result = self.born_again_model.evaluate(dataloader, batch_size=64,workers=8, verbose=1)
        return result
            

class CustomCallBack(tf.keras.callbacks.Callback):
    def __init__(self, model, model_prefix, save_model_root, train_data, val_data):
        super(CustomCallBack, self).__init__()
        self.model = model
        self.model_prefix = model_prefix
        self.save_model_root = save_model_root
        self.train_data = train_data
        self.val_data = val_data
    def on_epoch_end(self, epoch, logs=None):  
        train_metrics = model.evaluate(self.train_data)
        val_metrics = model.evaluate(self.val_data)
        # print(train_acc, val_acc)
        print("born_again_model train_loss: {:.4f}\ttrain_acc: {:.4f}\tval_loss: {:.4f}\tval_acc: {:.4f}".format(train_metrics[0],train_metrics[1], val_metrics[0], val_metrics[1]))
     
        model_name = 'keras_watch_screen_model_epoch_{:0>3d}_acc_{:.4f}_loss_{:.4f}.h5'.format(epoch+1, val_metrics[1], logs['loss'])
        save_model_full_path = os.path.join(self.save_model_root, model_name)
        model.born_again_model.save(save_model_full_path)
     
        
if __name__ == "__main__":
    
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    sess = tf.compat.v1.Session(config=config)
    print(("""
        GPU Congfig
        CUDA_VISIBLE_DEVICES :              {}
        PER_PROCESS_GPU_MEMORY_FRACTION     {}
    """).format(args.gpu_id, args.gpu_memory_fraction))
    
    batch_size = args.batch_size
    # 
    num_classes = 2
    epochs = args.max_epochs
    if args.dataloader_mask:
        print('dataloader type is mask')
        from dataloader_mask import Dataloader

    else:
        from dataloader import Dataloader
    train_loader = Dataloader(batch_size=batch_size, data_list=args.train_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w, shuffle=True)
    val_loader = Dataloader(batch_size=batch_size, data_list=args.val_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w, shuffle=True, model='eval')
    # for i, (data, label) in enumerate(val_loader):
    #     print(i, data[0].shape, data[1].shape, data[1].shape, label.shape)
    teacher_model_path = '/mnt/sda1/cgy/Keras_Watch_Screen/checkpoints/data_list_20210319/train_model_2021-08-11-07-52-56/keras_watch_screen_model.h5'
    teacher_model = get_teacher_model(teacher_model_path)
    
    # teacher_model.summary()
    # alpha = 0.1
    # temperature = 20
    model = BornAgainModel(teacher_model, TEMPERATURE, 'block_2_depthwise_relu', 2)
    model.born_again_model.summary()
    
   # optimizer = tf.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5)
    train_model = model.train_model
    train_model.compile(
        optimizer=optimizer,    
        loss=lambda y_true, y_pred: y_pred)
    

    # keras callback
    teacher_name = teacher_model_path.split('/')[-2]
    save_model_root = '/mnt/sda1/cgy/Keras_Knowledge_Distillation/checkpoints/train_model_TEMPERATURE_{}_Teacher_{}/'.format(TEMPERATURE, teacher_name)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_model_folder = 'train_model_{}'.format(time_str)
    save_model_full_root =os.path.join(save_model_root, save_model_folder)
    os.makedirs(save_model_full_root, exist_ok=True)
    training_callback = CustomCallBack(model, 'kd', save_model_full_root, train_loader, val_loader)


    train_model.fit(train_loader, epochs=epochs, callbacks=[training_callback], workers=8)

    
    
    
    