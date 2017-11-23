import os
import numpy as np
from shutil import copyfile
i = 0
# sort the test set by test.txt 's order'
with open("test.txt") as f:
    src_posit_list = f.readlines()
    for src_posit in src_posit_list:
        src_posit = src_posit[:-1]
        print(src_posit)
        dst_posit = "./test/"+str(i) + ".jpg"
        copyfile(src_posit, dst_posit)
        i=i+1
