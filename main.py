from __future__ import print_function
import argparse
import os
import random
import tensorflow as tf
import numpy as np
import time
import math
from PIL import Image, ImageOps
from model import *
import re
import sys

#input_arg
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='./data_stereo_flow/', 
                    help='datapath')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train')

args = parser.parse_args()
batch_size=2;
use_bn_flag=False;

def readPFM(file):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    color = None
    width = None
    height = None
    scale = None
    endian = None
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    A=str(file.readline().rstrip())[2:-1]
    A1,A2=A.split(' ')
    width=int(A1)
    height=int(A2)
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

print('Called with args:')
print(args)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_data_list(filepath):
    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []
    monkaa_path = filepath + '/monkaa/frames_cleanpass/'
    monkaa_disp = filepath + '/monkaa/disparity/'
    monkaa_dir  = os.listdir(monkaa_path)
    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
                all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')
        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    flying_path = filepath + '/flying/frames_cleanpass/'
    flying_disp = filepath + '/flying/disparity/'
    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A','B','C']
    for ss in subdir:
        flying = os.listdir(flying_dir+ss)
        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                    all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)
    flying_dir = flying_path+'/TEST/'#flying_test_dataset
    subdir = ['A','B','C']
    for ss in subdir:
        flying = os.listdir(flying_dir+ss)
        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
                    test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    driving_dir = filepath + 'Driving/frames_cleanpass/'
    driving_disp = filepath + 'Driving/disparity/'
    subdir1 = ['15mm_focallength']#you can add ,'35mm_focallength' in the bracket,but they are too similar
    subdir2 = ['scene_forwards']#you can add ,'scene_backwards' in the bracket,but they are too similar
    subdir3 = ['fast']#you can add ,'fast' in the bracket,but they are too similar
    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')    
                for im in imm_l:
                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                        all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)
                        all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')
                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                        all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp

#img_normalization
def scale_crop(input_img):
    input_img = input_img/255
    input_img[:,:,0] = (input_img[:,:,0] - 0.485)/0.229
    input_img[:,:,1] = (input_img[:,:,1] - 0.456)/0.224
    input_img[:,:,2] = (input_img[:,:,2] - 0.406)/0.225
    return input_img

def list_to_img(all_left_img,all_right_img,all_left_disp,index,batchsize,training=True,use_bn=False):
    list_left_img=[];
    list_right_img=[];
    list_left_disp=[];
    initial_index=index*batchsize
    for i in range(batchsize):
        left_img=Image.open(all_left_img[initial_index+i]).convert('RGB')
        right_img=Image.open(all_right_img[initial_index+i]).convert('RGB')
        left_disp,_=readPFM(all_left_disp[initial_index+i])
        left_disp=np.ascontiguousarray(left_disp,dtype=np.float32)
        if training:  
            w, h = left_img.size
            th, tw = 256, 512
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            left_disp = left_disp[y1:y1 + th, x1:x1 + tw]
        left_img=np.array(left_img,dtype=np.float32)
        right_img=np.array(right_img,dtype=np.float32)
        if use_bn:
            left_img   = scale_crop(left_img)
            right_img  = scale_crop(right_img)
        list_left_img.append(left_img)
        list_right_img.append(right_img)
        list_left_disp.append(left_disp)
    batch_left_img=np.array(list_left_img)
    batch_right_img=np.array(list_right_img)
    batch_left_disp=np.array(list_left_disp)
    return batch_left_img,batch_right_img,batch_left_disp

#read data_path
all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = load_data_list(args.datapath)

def main():
        with tf.Session() as sess:
            save_dir='E:/PSMNet-Tensorflow/snapshot/'
            start_full_time = time.time()
        #bulid model
            model = Model(sess, height=256, weight=512, batch_size=batch_size, max_disp=args.maxdisp,training=True,use_bn=use_bn_flag)
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)
        #reload model   
            ckpt = tf.train.get_checkpoint_state(save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            for epoch in range(1, args.epochs+1):
        # training
                batch_num = int(len(all_left_img)/ batch_size)
                for i in range(batch_num):
                    batch_left_img,batch_right_img,batch_left_disp=list_to_img(all_left_img,all_right_img,all_left_disp,i,batch_size, True,use_bn_flag)
                    start_time=time.time()
                    train_loss = model.train(batch_left_img,batch_right_img, batch_left_disp)
                    print('Iter %d training loss = %.3f , time = %.2f' %(i, train_loss, time.time() - start_time))
                print('epoch %d total training loss = %.3f' %(epoch, train_loss))
                if epoch % 1 == 0:
                    saver.save(sess, './snapshot/', global_step=epoch)
            writer = tf.summary.FileWriter(save_dir, tf.get_default_graph())
            writer.close()
            print('full_time=%.1f'%(time.time()-start_full_time))

if __name__ == '__main__':
   main()
