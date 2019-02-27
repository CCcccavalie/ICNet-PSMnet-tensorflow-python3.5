# ICNet-PSMnet-tensorflow-python3.5
This network uses ICNet as the farward network, uses PSMnet for depth estimation. The code was written in python 3.5/tensorflow
# Dataset Prepare
1.Creat a folder in the project directory, rename the folder as: 'data_stereo_flow'

2.Creat three sub folders in 'data_stereo_flow', rename the folders as: 'flying', 'Driving', 'monkaa'.

3.Download the Scene Flow dataset and uncompress them in folders respectively 

link: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

Notice:Three classes including 'FlyingThings3D','Driving','Monkaa'. Train_data：RGB images (cleanpass); Lable：Disparity

The dataset is very very large!!! If you don't want download all dataset, please modify 'load_data_list' in main.py
# Simple Start
1.Creat a folder in the project directory, rename the folder as: 'snapshot', for saved model

2.Creat a folder in the project directory, rename the folder as: 'output', for demo result

3.Train:run python main.py in cmd

if you run main.py first time,annotate these two codes in main.py

if you run main.py not first time, and want to train based on the saved model last time, unannotate these codes:

ckpt = tf.train.get_checkpoint_state(save_dir)

saver.restore(sess, ckpt.model_checkpoint_path)

4.Test:run python test.py

5.See image process result: run demo.py
# My Experience
Working on my device (1080TiGPU) five epochs costs 37 hours. I estimate that it will need 300 epochs to get good results,so it's very very time-consuming.

Using BN can shorten the training time obviously, but you need to increase batchsize(16 is suitable), I have wrote the BN layer

if your device can affort the large graphic memory,just modify 'use_bn_flag=True' in main.py,demo.py,test.py

notic:if you want to use BN, small batch_size will lead worse result

Unfortunately, 1080ti can only support batch size=2
