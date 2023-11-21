import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

'''
Converts the downloaded Tiny-ImageNet validation folder to ImageNet style
'''

target_folder = './data/new_val/'
test_folder   = './data/test/'

val_dict = {}
with open('./data/new_val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]
        
paths = glob.glob('./data/new_val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        # os.mkdir(target_folder + str(folder) + '/images')
    if not os.path.exists(test_folder + str(folder)):
        os.mkdir(test_folder + str(folder))
        # os.mkdir(test_folder + str(folder) + '/images')
        
        
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/' + str(file)
    move(path, dest)

rmdir('./data/new_val/images')