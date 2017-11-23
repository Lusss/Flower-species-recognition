import os
import numpy as np
from shutil import copyfile

with open("train.txt") as f:
    train_list = f.readlines()
    train_list = [train_element[:-1].split() for train_element in train_list]
    train_list = np.array(train_list)
filename = np.zeros(6)
i = 0
for src_posit in train_list:
    filename[int(src_posit[1])] = filename[int(src_posit[1])]+1
    dst_posit = "./data/train/"+str(src_posit[1])+"/"+str(int(filename[int(src_posit[1])]))+".jpg"
    copyfile(src_posit[0], dst_posit)
    print("finished copy "+str(i))
    print(src_posit)
    i = i+1


# with open("test.txt") as f:
#     train_list = f.readlines()
#     train_list = [train_element[:-1].split() for train_element in train_list]
#     train_list = np.array(train_list)
# i = 0
# for src_posit in train_list:
#     dst_posit = "./test_data/"+str(i)+".jpg"
#     copyfile(src_posit[0], dst_posit)
#     print("finished copy "+str(i))
#     i = i+1


with open("val.txt") as f:
    train_list = f.readlines()
    train_list = [train_element[:-1].split() for train_element in train_list]
    train_list = np.array(train_list)
filename = np.zeros(6)
i = 0
for src_posit in train_list:
    filename[int(src_posit[1])] = filename[int(src_posit[1])]+1
    dst_posit = "./data/validation/"+str(src_posit[1])+"/"+str(int(filename[int(src_posit[1])]))+".jpg"
    copyfile(src_posit[0], dst_posit)
    print("finished copy "+str(i))
    i = i+1
