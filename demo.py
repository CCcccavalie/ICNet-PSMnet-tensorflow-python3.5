import os
import os.path
import random
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from model import *
import cv2
from scipy import misc

use_bn_flag=False

def scale_crop(input_img):
    input_img = input_img/255
    input_img[:,:,0] = (input_img[:,:,0] - 0.485)/0.229
    input_img[:,:,1] = (input_img[:,:,1] - 0.456)/0.224
    input_img[:,:,2] = (input_img[:,:,2] - 0.406)/0.225
    return input_img

def main():
    im_dir= 'E:/PSMNet-Tensorflow/data_stereo_flow/Driving/frames_cleanpass/15mm_focallength/scene_backwards/fast/'
    #gt_f_dir= 'E:/PSMNet-Tensorflow/data_stereo_flow/training/disp_noc'
    left_fold  = 'left/'
    right_fold = 'right/'
    im_name='0102.png'
    left_dir=os.path.join(im_dir,left_fold,im_name)
    right_dir=os.path.join(im_dir,right_fold,im_name)
    #gt_dir=os.path.join(gt_f_dir,im_name)

    left=cv2.imread(left_dir)
    m=256
    n=512
    right=cv2.imread(right_dir)
    #gt_im=cv2.imread(gt_dir)
    left_scale=cv2.resize(left,(n,m))#cv2.resize里长宽颠倒
    right_scale=cv2.resize(right,(n,m))
    #gt_im_scale=cv2.resize(gt_im,(n,m))
    if use_bn_flag:
        left   = [scale_crop(left_scale)]
        right   = [scale_crop(right_scale)]
    else:
        left   = [np.array(left_scale,dtype=np.float32)]#减均值除以方差处理
        right  = [np.array(right_scale,dtype=np.float32)]

    with tf.Session() as sess:
        
        model = Model(sess, height=m, weight=n, batch_size=1, max_disp=192,training=False,use_bn=use_bn_flag)
        loader = tf.train.Saver(var_list=tf.global_variables())
        check_dir = tf.train.get_checkpoint_state('E:/PSMNet-Tensorflow/snapshot/')
        model_dir=check_dir.model_checkpoint_path
        loader.restore(sess, model_dir)
        output_value=tf.cast(model.disps[2],dtype=tf.int32)
        output1 = sess.run(output_value,feed_dict={model.left: left, model.right: right})
        start_time = time.time()
        output = sess.run(output_value,feed_dict={model.left: left, model.right: right})
        print('time = %.3f' %(time.time() - start_time))
        print(output[0])
    misc.imsave('./output/output.jpg' , output[0])
    #misc.imsave('./output/gt.jpg' , gt_im_scale[:,:,0])


if __name__ == '__main__':
   main()