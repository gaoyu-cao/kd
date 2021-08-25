import tensorflow as tf
import h5py
from model import KDNet
from opts import parser
import numpy as np



def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("      {}: {}".format(key, value))  

            print("    Dataset:")
            for name, d in g.items(): # 读取各层储存具体信息的Dataset类
                print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
                print("      {}: {}".format(name. d.value))
    finally:
        f.close()


if __name__ == '__main__':
    weight_file_path = '/mnt/sda2/cj/Keras_Knowledge_Distillation/checkpoints/model_loop_v5/Keras_Screen_model_015.h5'
    # print_keras_wegiths(weight_file_path)
    global args
    args = parser.parse_args()
    model = tf.keras.models.load_model(weight_file_path)
    print("load keras model from '{}' successful".format(weight_file_path))
    # model = KDNet(temperature=1,
    #       teacher_new_weights=None,
    #       use_droppout=args.dropout,
    #       dropout_ratdio=args.dropout_ratdio)
    # model.load_weights(weight_file_path)
    # print("load keras weights from '{}' successful".format(weight_file_path))
    for layer in model.layers:
        print("layer.name {}\nlayer's weights {}".format(layer.name, layer.get_weights()))
        # if np.isnan(layer.get_weights()).any():
        #     print(layer.name)
    #     else:
    # print('no nan')
        # if layer.name=='Conv1_3':
            
        #     print("layer.name {}\nlayer's weights {}".format(layer.name, layer.get_weights()))