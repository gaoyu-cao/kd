import os
import random

# ==================== check data =============================
#data_path = '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/train_new.txt'

# def show_data_distribuion(data_path):
#     positive_num, negative_num = 0, 0
#     with open(data_path, 'r') as f:
#         data_lists = f.readlines()

#     for data in data_lists:
#         if data.split(',')[2] == '1':

#             positive_num += 1
#         else:
#             negative_num += 1
#     print("""
#     'data distribution：{}'
#     all:{}\npositive_num :{}\nnegative_num :{}
#     """.format(data_path, len(data_lists),positive_num, negative_num ))

# data_Path_list = ['/mnt/sda2/cj/headscreen_cls/data_list_20210319/train_list.txt',
#                   '/mnt/sda2/cj/headscreen_cls/data_list_20210319/val_list.txt',
#                 '/mnt/sda2/cj/headscreen_cls/data_list_20210319/test_list.txt',
#                 '/mnt/sda2/cj/headscreen_cls/data_list_20210319/train_list.txt',
#                 '/mnt/sda2/cj/headscreen_cls/data_list_20210319/val_list.txt',
#                 '/mnt/sda2/cj/headscreen_cls/data_list_20210319/test_list.txt']

# for data_path in data_Path_list:
#     show_data_distribuion(data_path)

# ====================merge txt  ==============================
def write_txt(data_list, save_path):
    random.shuffle(data_list)
    with open(save_path, 'a', encoding='utf-8') as f:
        for info in data_list:
            info = info
            f.write(info)

data_path = '/mnt/sda2/cj/headscreen_cls/data_list_20210319/train_list.txt'
with open(data_path, 'r') as f:
    data_lists = f.readlines()

positive_list = []
negative_list = []

for data in data_lists:
    if data.split(',')[2] == '1':
        positive_list.append(data)
    else:
        negative_list.append(data)

print('positive data number is {}\nnegative data number is{}'
        .format(len(positive_list), len(negative_list)))

random.shuffle(negative_list)
# random.shuffle()
select_negat = negative_list[:30000]
no_select = negative_list[30000:]

# write_txt(positive_list, '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/positive.txt')
# write_txt(select_negat, '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/negative_1.txt')
# write_txt(no_select, '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/negative_2.txt')
positive_list.extend(select_negat)
print(len(positive_list))
write_txt(positive_list, '/mnt/sda2/cj/Keras_Knowledge_Distillation/data/train_kd.txt')
