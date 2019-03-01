import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.keras as keras


class Model:

    def __init__(self, sess, height=512, weight=1024, batch_size=16,max_disp = 192,training = True,use_bn=False):
        self.reg = 1e-5  # TODO
        self.max_disp = max_disp  # TODO
        self.image_size_tf = None
        self.height = height
        self.weight = weight
        self.batch_size = batch_size
        self.training=training
        self.use_bn=use_bn
        self.sess = sess
        self.build_model()


    def build_model(self):
        self.left = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.right = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight, 3])
        self.dataL = tf.placeholder(tf.float32, shape=[self.batch_size, self.height, self.weight]) 
        self.image_size_tf = tf.shape(self.left)[1:3]
        if self.use_bn:
            self.conv4_left = self.CNN_BN(self.left,self.training)
            self.conv4_right = self.CNN_BN(self.right,self.training)
        else:
            self.conv4_left = self.CNN(self.left,self.training)
            self.conv4_right = self.CNN(self.right,self.training)

        self.cost = self.cost_vol(self.conv4_left, self.conv4_right, self.max_disp)
        self.outputs = self.CNN3D(self.cost,self.reg,use_bn=self.use_bn)
        self.disps = self.output(self.outputs)#size of (B, H, W),3个out
        self.loss = 0.5*self._smooth_l1_loss(self.disps[0], self.dataL)+0.7*self._smooth_l1_loss(self.disps[1], self.dataL)+self._smooth_l1_loss(self.disps[2], self.dataL)
        learning_rate = 1e-4
        if self.use_bn:
             learning_rate=learning_rate*10
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
        except:
            self.sess.run(tf.initialize_all_variables())

    def conv(self,bottom,k_h,k_w,c_o,s_h,s_w,name,padding='SAME',relu=True,training=True):
        c_i=bottom.get_shape()[-1]
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            biases =tf.get_variable('biases', [c_o],trainable=training)
            filter = tf.get_variable('weights',[k_h,k_w,c_i,c_o],trainable=training)
            output=tf.nn.conv2d(bottom, filter, [1, s_h, s_w, 1], padding=padding,data_format='NHWC')
            output = tf.nn.bias_add(output, biases)
            if relu :
                output=tf.nn.relu(output,name=scope.name)
            return output
    def atrous_conv(self,bottom,k_h,k_w,c_o,dilation,name,padding='SAME',relu=True,training=True):
        c_i=bottom.get_shape()[-1]
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            biases =tf.get_variable('biases', [c_o],trainable=training)
            filter = tf.get_variable('weights',[k_h,k_w,c_i,c_o],trainable=training)
            output=tf.nn.atrous_conv2d(bottom, filter, dilation, padding=padding)
            output = tf.nn.bias_add(output, biases)
            if relu :
                output=tf.nn.relu(output,name=scope.name)
            return output

    def hourglass(self,bottom,reg=1e-4,use_bn=True):
        conv_params = {
            'padding': 'SAME',
            'kernel_initializer': tfc.layers.xavier_initializer(),
            'kernel_regularizer': tfc.layers.l2_regularizer(reg),
            'bias_regularizer': tfc.layers.l2_regularizer(reg),
            'reuse':tf.AUTO_REUSE}
        
        output_sum=[]
        output = []
        origin_bottom=bottom
        bottom=tf.nn.relu(bottom,name='3Dconv_add/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=2, name='stack_1_1',**conv_params)#h/2*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_1_1_bn')
        bottom=tf.nn.relu(bottom,name='stack_1_1_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=1, name='stack_1_2',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_1_2_bn')
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_1_2_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=2, name='stack_2_1',**conv_params)#h/4*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_2_1_bn')
        bottom=tf.nn.relu(bottom,name='stack_2_1_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=1, name='stack_2_2',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_2_2_bn')
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_2_2_bn/relu')
        bottom = tf.layers.conv3d_transpose( bottom, 64, 3, strides=2,  padding='SAME',name='stack_3_1',reuse=tf.AUTO_REUSE)#h/2*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_3_1_bn')
        bottom = tf.add( bottom,output[-2],  name='stack_3_2') 
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_3_2/relu')
        bottom = tf.layers.conv3d_transpose( bottom, 32, 3, strides=2, padding='SAME', name='stack_4_1',reuse=tf.AUTO_REUSE)#h*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_4_1_bn')
        bottom = tf.add( bottom,origin_bottom,  name='stack_4_2') 
        output.append(bottom)
        output_sum.append(output)
        
        output = []
        origin_bottom=bottom
        bottom=tf.nn.relu(bottom,name='stack_4_2/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=2, name='stack_5_1',**conv_params)#h/2*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_5_1_bn')
        bottom=tf.nn.relu(bottom,name='stack_5_1_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=1, name='stack_5_2',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_5_2_bn')
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_5_2_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=2, name='stack_6_1',**conv_params)#h/4*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_6_1_bn')
        bottom=tf.nn.relu(bottom,name='stack_6_1_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=1, name='stack_6_2',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_6_2_bn')
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_6_2_bn/relu')
        bottom = tf.layers.conv3d_transpose( bottom, 64, 3, strides=2,  padding='SAME',name='stack_7_1',reuse=tf.AUTO_REUSE)#h/2*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_7_1_bn')
        bottom = tf.add( bottom,output[-2],  name='stack_7_2') 
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_7_2/relu')
        bottom = tf.layers.conv3d_transpose( bottom, 32, 3, strides=2, padding='SAME', name='stack_8_1',reuse=tf.AUTO_REUSE)#h*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_8_1_bn')
        bottom = tf.add( bottom,origin_bottom,  name='stack_8_2') 
        output.append(bottom)
        output_sum.append(output)
        
        output = []
        origin_bottom=bottom
        bottom=tf.nn.relu(bottom,name='stack_8_2/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=2, name='stack_9_1',**conv_params)#h/2*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_9_1_bn')
        bottom=tf.nn.relu(bottom,name='stack_9_1_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=1, name='stack_9_2',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_9_2_bn')
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_9_2_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=2, name='stack_10_1',**conv_params)#h/4*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_10_1_bn')
        bottom=tf.nn.relu(bottom,name='stack_10_1_bn/relu')
        bottom=tf.layers.conv3d(bottom, 64, 3, strides=1, name='stack_10_2',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_10_2_bn')
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_10_2_bn/relu')
        bottom = tf.layers.conv3d_transpose( bottom, 64, 3, strides=2,  padding='SAME',name='stack_11_1',reuse=tf.AUTO_REUSE)#h/2*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_11_1_bn')
        bottom = tf.add( bottom,output[-2],  name='stack_11_2') 
        output.append(bottom)
        bottom=tf.nn.relu(bottom,name='stack_11_2/relu')
        bottom = tf.layers.conv3d_transpose( bottom, 32, 3, strides=2, padding='SAME', name='stack_12_1',reuse=tf.AUTO_REUSE)#h*64
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='stack_12_1_bn')
        bottom = tf.add( bottom,origin_bottom,  name='stack_12_2') 
        output.append(bottom)
        output_sum.append(output)
        

        return output_sum
        
    def CNN3D(self, bottom,reg,use_bn=True):
        conv_params = {
                'padding': 'SAME',
                'kernel_initializer': tfc.layers.xavier_initializer(),
                'kernel_regularizer': tfc.layers.l2_regularizer(reg),
                'bias_regularizer': tfc.layers.l2_regularizer(reg),
                'reuse': tf.AUTO_REUSE
                    }
        bottom=tf.layers.conv3d(bottom,32,3,1,name='3Dconv0_1',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='3Dconv0_1_bn')
        bottom=tf.nn.relu(bottom,name='3Dconv0_1_bn/relu')
        bottom=tf.layers.conv3d(bottom,32,3,1,name='3Dconv0_2',**conv_params)
        short_cut = bottom
        bottom = tf.layers.conv3d(bottom,32,3,1,name='3Dconv0_3',**conv_params)
        if use_bn:
            bottom = tf.layers.batch_normalization(bottom,training=self.training,reuse=tf.AUTO_REUSE, name='3Dconv0_3_bn')
        bottom=tf.nn.relu(bottom,name='3Dconv0_3_bn/relu')
        bottom = tf.layers.conv3d(bottom,32,3, 1,name='3Dconv0_4', **conv_params)
        _3Dconv1 = tf.add(bottom, short_cut, '3Dconv_add')
        _3Dstack = self.hourglass( _3Dconv1,reg,use_bn=True)
        outputs=[]
        output_1=tf.nn.relu(_3Dstack[0][3],name='output_1_0/relu')
        output_1=tf.layers.conv3d(output_1,32,3, name='output_1_1',**conv_params)
        if use_bn:
            output_1 = tf.layers.batch_normalization(output_1,training=self.training,reuse=tf.AUTO_REUSE, name='output_1_1_bn')
        output_1=tf.nn.relu(output_1,name='output_1_1_bn/relu')
        output_1=tf.layers.conv3d(output_1,1,3, name='output_1',**conv_params)
        if use_bn:
            output_1 = tf.layers.batch_normalization(output_1,training=self.training,reuse=tf.AUTO_REUSE, name='output_1_bn')
        outputs.append(output_1)
        output_2=tf.nn.relu(_3Dstack[1][3],name='output_2_0/relu')
        output_2=tf.layers.conv3d(output_2,32,3, name='output_2_1',**conv_params)
        if use_bn:
            output_2 = tf.layers.batch_normalization(output_2,training=self.training,reuse=tf.AUTO_REUSE, name='output_2_1_bn')
        output_2=tf.nn.relu(output_2,name='output_2_1_bn/relu')
        output_2=tf.layers.conv3d(output_2,1,3, name='output_2_2',**conv_params)
        if use_bn:
            output_2 = tf.layers.batch_normalization(output_2,training=self.training,reuse=tf.AUTO_REUSE, name='output_2_2_bn')
        output_2=tf.add(output_2,output_1,name='output_2')
        outputs.append(output_2)
        output_3=tf.nn.relu(_3Dstack[2][3],name='output_3_0/relu')
        output_3=tf.layers.conv3d(output_3,32,3, name='output_3_1',**conv_params)
        if use_bn:
            output_3 = tf.layers.batch_normalization(output_3,training=self.training,reuse=tf.AUTO_REUSE, name='output_3_1_bn')
        output_3=tf.nn.relu(output_3,name='output_3_1_bn/relu')
        output_3=tf.layers.conv3d(output_3,1,3, name='output_3_2',**conv_params)
        if use_bn:
            output_3 = tf.layers.batch_normalization(output_3,training=self.training,reuse=tf.AUTO_REUSE, name='output_3_2_bn')
        output_3=tf.add(output_3,output_2,name='output_3')
        outputs.append(output_3)

        return outputs

    def CNN(self, bottom,training=True):
        new_h0,new_w0=bottom.get_shape().as_list()[1:3]
        bottom_interp=tf.image.resize_bilinear(bottom, size=[int(new_h0*0.5),int(new_w0*0.5)], align_corners=True,name='conv3_1_sub4')
        output1_1=self.conv(bottom_interp,3, 3, 32, 2, 2, padding='SAME', relu=True, name='conv1_1_3x3_s2',training=training)
        output1_2=self.conv(output1_1,3, 3, 32, 1, 1, padding='SAME', relu=True, name='conv1_2_3x3',training=training)
        output1_3=self.conv(output1_2,3, 3, 64, 1, 1, padding='SAME', relu=True, name='conv1_3_3x3',training=training)
        max_pool1_1=tf.nn.max_pool(output1_3,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1_3x3_s2',data_format='NHWC')
        output1_4=self.conv(max_pool1_1,1, 1, 128, 1, 1, padding='SAME',relu=False, name='conv2_1_1x1_proj',training=training)
        
        output2_1=self.conv(max_pool1_1,1, 1, 32, 1, 1, padding='SAME', relu=True, name='conv2_1_1x1_reduce',training=training)
        output2_2=self.conv(output2_1,3, 3, 32, 1, 1, padding='SAME', relu=True, name='conv2_1_3x3',training=training)
        output2_3=self.conv(output2_2,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv2_1_1x1_increase',training=training)
        sum1=tf.add(output1_4,output2_3,name='conv2_1')
        sum1_relu=tf.nn.relu(sum1,name='conv2_1/relu')
        
        output3_1=self.conv(sum1_relu,1, 1, 32, 1, 1, padding='SAME', relu=True, name='conv2_2_1x1_reduce',training=training)
        output3_2=self.conv(output3_1,3, 3, 32, 1, 1, padding='SAME', relu=True, name='conv2_2_3x3',training=training)
        output3_3=self.conv(output3_2,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv2_2_1x1_increase',training=training)
        sum2=tf.add(sum1_relu,output3_3,name='conv2_2')
        sum2_relu=tf.nn.relu(sum2,name='conv2_2/relu')
        
        output4_1=self.conv(sum2_relu,1, 1, 32, 1, 1, padding='SAME', relu=True, name='conv2_3_1x1_reduce',training=training)
        output4_2=self.conv(output4_1,3, 3, 32, 1, 1, padding='SAME', relu=True, name='conv2_3_3x3',training=training)
        output4_3=self.conv(output4_2,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv2_3_1x1_increase',training=training)
        sum3=tf.add(sum2_relu,output4_3,name='conv2_3')
        sum3_relu=tf.nn.relu(sum3,name='conv2_3/relu')
        output4_4=self.conv(sum3_relu,1, 1, 256, 2, 2, padding='SAME', relu=False, name='conv3_1_1x1_proj',training=training)
        
        output5_1=self.conv(sum3_relu,1, 1, 64, 2, 2, padding='SAME', relu=True, name='conv3_1_1x1_reduce',training=training)
        output5_2=self.conv(output5_1,3, 3, 64, 1, 1, padding='SAME', relu=True, name='conv3_1_3x3',training=training)
        output5_3=self.conv(output5_2,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_1_1x1_increase',training=training)
        sum4=tf.add(output4_4,output5_3,name='conv3_1')
        sum4_relu=tf.nn.relu(sum4,name='conv3_1/relu')
        output5_4=self.conv(sum4_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv3_1_sub2_proj',training=training)
        new_h,new_w=sum4_relu.get_shape().as_list()[1:3]
        sum4_relu_interp=tf.image.resize_bilinear(sum4_relu, size=[int(new_h*0.5),int(new_w*0.5)], align_corners=True,name='conv3_1_sub4')
        
        output6_1=self.conv(sum4_relu_interp,1, 1, 64, 1, 1, padding='SAME', relu=True, name='conv3_2_1x1_reduce',training=training)
        output6_2=self.conv(output6_1,3, 3, 64, 1, 1, padding='SAME', relu=True, name='conv3_2_3x3',training=training)
        output6_3=self.conv(output6_2,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_2_1x1_increase',training=training)
        sum5=tf.add(sum4_relu_interp,output6_3,name='conv3_2')
        sum5_relu=tf.nn.relu(sum5,name='conv3_2/relu')
        
        output7_1=self.conv(sum5_relu,1, 1, 64, 1, 1, padding='SAME', relu=True, name='conv3_3_1x1_reduce',training=training)
        output7_2=self.conv(output7_1,3, 3, 64, 1, 1, padding='SAME', relu=True, name='conv3_3_3x3',training=training)
        output7_3=self.conv(output7_2,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_3_1x1_increase',training=training)
        sum6=tf.add(sum5_relu,output7_3,name='conv3_3')
        sum6_relu=tf.nn.relu(sum6,name='conv3_3/relu')
        
        output8_1=self.conv(sum6_relu,1, 1, 64, 1, 1, padding='SAME', relu=True, name='conv3_4_1x1_reduce',training=training)
        output8_2=self.conv(output8_1,3, 3, 64, 1, 1, padding='SAME', relu=True, name='conv3_4_3x3',training=training)
        output8_3=self.conv(output8_2,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_4_1x1_increase',training=training)
        sum7=tf.add(sum6_relu,output8_3,name='conv3_4')
        sum7_relu=tf.nn.relu(sum7,name='conv3_4/relu')
        output8_4=self.conv(sum7_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_1_1x1_proj',training=training)
        
        output9_1=self.conv(sum7_relu,1, 1, 128, 1, 1, padding='SAME', relu=True, name='conv4_1_1x1_reduce',training=training)
        output9_2=self.atrous_conv(output9_1,3, 3, 128, 2, padding='SAME', relu=True, name='conv4_1_3x3',training=training)
        output9_3=self.conv(output9_2,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_1_1x1_increase',training=training)
        sum8=tf.add(output8_4,output9_3,name='conv4_1')
        sum8_relu=tf.nn.relu(sum8,name='conv4_1/relu')
        
        output10_1=self.conv(sum8_relu,1, 1, 128, 1, 1, padding='SAME', relu=True, name='conv4_2_1x1_reduce',training=training)
        output10_2=self.atrous_conv(output10_1,3, 3, 128, 2, padding='SAME', relu=True, name='conv4_2_3x3',training=training)
        output10_3=self.conv(output10_2,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_2_1x1_increase',training=training)
        sum9=tf.add(sum8_relu,output10_3,name='conv4_2')
        sum9_relu=tf.nn.relu(sum9,name='conv4_2/relu')
        
        output11_1=self.conv(sum9_relu,1, 1, 128, 1, 1, padding='SAME', relu=True, name='conv4_3_1x1_reduce',training=training)
        output11_2=self.atrous_conv(output11_1,3, 3, 128, 2, padding='SAME', relu=True, name='conv4_3_3x3',training=training)
        output11_3=self.conv(output11_2,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_3_1x1_increase',training=training)
        sum10=tf.add(sum9_relu,output11_3,name='conv4_3')
        sum10_relu=tf.nn.relu(sum10,name='conv4_3/relu')
        
        output12_1=self.conv(sum10_relu,1, 1, 128, 1, 1, padding='SAME', relu=True, name='conv4_4_1x1_reduce',training=training)
        output12_2=self.atrous_conv(output12_1,3, 3, 128, 2, padding='SAME', relu=True, name='conv4_4_3x3',training=training)
        output12_3=self.conv(output12_2,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_4_1x1_increase',training=training)
        sum11=tf.add(sum10_relu,output12_3,name='conv4_4')
        sum11_relu=tf.nn.relu(sum11,name='conv4_4/relu')
        
        output13_1=self.conv(sum11_relu,1, 1, 128, 1, 1, padding='SAME', relu=True, name='conv4_5_1x1_reduce',training=training)
        output13_2=self.atrous_conv(output13_1,3, 3, 128, 2, padding='SAME', relu=True, name='conv4_5_3x3',training=training)
        output13_3=self.conv(output13_2,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_5_1x1_increase',training=training)
        sum12=tf.add(sum11_relu,output13_3,name='conv4_5')
        sum12_relu=tf.nn.relu(sum12,name='conv4_5/relu')
        
        output14_1=self.conv(sum12_relu,1, 1, 128, 1, 1, padding='SAME', relu=True, name='conv4_6_1x1_reduce',training=training)
        output14_2=self.atrous_conv(output14_1,3, 3, 128, 2, padding='SAME', relu=True, name='conv4_6_3x3',training=training)
        output14_3=self.conv(output14_2,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_6_1x1_increase',training=training)
        sum13=tf.add(sum12_relu,output14_3,name='conv4_6')
        sum13_relu=tf.nn.relu(sum13,name='conv4_6/relu')
        output14_4=self.conv(sum13_relu,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_1_1x1_proj',training=training)
        
        output15_1=self.conv(sum13_relu,1, 1, 256, 1, 1, padding='SAME', relu=True, name='conv5_1_1x1_reduce',training=training)
        output15_2=self.atrous_conv(output15_1,3, 3, 256, 4, padding='SAME', relu=True, name='conv5_1_3x3',training=training)
        output15_3=self.conv(output15_2,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_1_1x1_increase',training=training)
        sum14=tf.add(output14_4,output15_3,name='conv5_1')
        sum14_relu=tf.nn.relu(sum14,name='conv5_1/relu')
        
        output16_1=self.conv(sum14_relu,1, 1, 256, 1, 1, padding='SAME', relu=True, name='conv5_2_1x1_reduce',training=training)
        output16_2=self.atrous_conv(output16_1,3, 3, 256, 4, padding='SAME', relu=True, name='conv5_2_3x3',training=training)
        output16_3=self.conv(output16_2,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_2_1x1_increase',training=training)
        sum15=tf.add(sum14_relu,output16_3,name='conv5_2')
        sum15_relu=tf.nn.relu(sum15,name='conv5_2/relu')
        
        output17_1=self.conv(sum15_relu,1, 1, 256, 1, 1, padding='SAME', relu=True, name='conv5_3_1x1_reduce',training=training)
        output17_2=self.atrous_conv(output17_1,3, 3, 256, 4, padding='SAME', relu=True, name='conv5_3_3x3',training=training)
        output17_3=self.conv(output17_2,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_3_1x1_increase',training=training)
        sum16=tf.add(sum15_relu,output17_3,name='conv5_3')
        sum16_relu=tf.nn.relu(sum16,name='conv5_3/relu')
        
        h,w=sum16_relu.get_shape().as_list()[1:3]
        avg_pool_1=tf.nn.avg_pool(sum16_relu,ksize=[1, h, w, 1],strides=[1, h, w, 1],padding='VALID',name='conv5_3_pool1',data_format='NHWC')
        avg_pool_1_resize=tf.image.resize_bilinear(avg_pool_1, size=[h,w], align_corners=True, name='conv5_3_pool1_interp')
        avg_pool_2=tf.nn.avg_pool(sum16_relu,ksize=[1, h/2, w/2, 1],strides=[1, h/2, w/2, 1],padding='VALID',name='conv5_3_pool2',data_format='NHWC')
        avg_pool_2_resize=tf.image.resize_bilinear(avg_pool_2, size=[h,w], align_corners=True, name='conv5_3_pool2_interp')
        avg_pool_3=tf.nn.avg_pool(sum16_relu,ksize=[1, h/3, w/3, 1],strides=[1, h/3, w/3, 1],padding='VALID',name='conv5_3_pool3',data_format='NHWC')
        avg_pool_3_resize=tf.image.resize_bilinear(avg_pool_3, size=[h,w], align_corners=True, name='conv5_3_pool3_interp')
        avg_pool_4=tf.nn.avg_pool(sum16_relu,ksize=[1, h/6, w/6, 1],strides=[1, h/6, w/6, 1],padding='VALID',name='conv5_3_pool4',data_format='NHWC')
        avg_pool_4_resize=tf.image.resize_bilinear(avg_pool_4, size=[h,w], align_corners=True, name='conv5_3_pool4_interp')
        sum17=tf.add_n([sum16_relu,avg_pool_1_resize,avg_pool_2_resize,avg_pool_3_resize,avg_pool_4_resize],name='conv5_3_sum')
        
        output17_1=self.conv(sum17,1, 1, 256, 1, 1, padding='SAME', relu=True, name='conv5_4_k1',training=training)
        output17_1_interp=tf.image.resize_bilinear(output17_1, size=[int(h*2),int(w*2)], align_corners=True,name='conv5_4_interp')
        output17_2=self.atrous_conv(output17_1_interp,3, 3, 128, 2, padding='SAME', relu=False, name='conv_sub4',training=training)
        sum18=tf.add(output5_4,output17_2,name='sub24_sum')
        sum18_relu=tf.nn.relu(sum18,name='sub24_sum/relu')
        
        output18_1_interp=tf.image.resize_bilinear(sum18_relu, size=[int(h*4),int(w*4)], align_corners=True,name='sub24_sum_interp')
        output18_2=self.atrous_conv(output18_1_interp,3, 3, 128, 2, padding='SAME', relu=False, name='conv_sub2',training=training)

        output19_1=self.conv(bottom,3, 3, 32, 2, 2, padding='SAME', relu=True, name='conv1_sub1',training=training)
        output19_2=self.conv(output19_1,3, 3, 32, 2, 2, padding='SAME', relu=True, name='conv2_sub1',training=training)
        output19_3=self.conv(output19_2,3, 3, 64, 2, 2, padding='SAME', relu=True, name='conv3_sub1',training=training)
        output19_4=self.conv(output19_3,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv3_sub1_proj',training=training)

        sum19=tf.add(output18_2,output19_4,name='sub12_sum')
        sum19_interp=tf.image.resize_bilinear(sum19, size=[int(h*8),int(w*8)], align_corners=True,name='sub12_sum_interp')
        output20_1=self.conv(sum19_interp,1, 1, 32, 1, 1, padding='SAME', relu=False, name='conv4_proj',training=training)
        #sum20_interp=tf.image.resize_bilinear(output20_1, size=[int(h*32),int(w*32)], align_corners=True,name='sub12_sum_interp')#与原图等大
        return output20_1

    def CNN_BN(self, bottom,training=True):
        new_h0,new_w0=bottom.get_shape().as_list()[1:3]
        bottom_interp=tf.image.resize_bilinear(bottom, size=[int(new_h0*0.5),int(new_w0*0.5)], align_corners=True,name='conv3_1_sub4')
        output1_1=self.conv(bottom_interp,3, 3, 32, 2, 2, padding='SAME', relu=False, name='conv1_1_3x3_s2',training=training)
        output1_1_bn=tf.layers.batch_normalization(output1_1,momentum=0.95,epsilon=1e-5,training=training,name='conv1_1_3x3_s2_bn',reuse=tf.AUTO_REUSE)
        output1_1_bn_relu=tf.nn.relu(output1_1_bn,name='conv1_1_3x3_s2_bn/relu')
        output1_2=self.conv(output1_1_bn_relu,3, 3, 32, 1, 1, padding='SAME', relu=False, name='conv1_2_3x3',training=training)
        output1_2_bn=tf.layers.batch_normalization(output1_2,momentum=0.95,epsilon=1e-5,training=training,name='conv1_2_3x3_bn',reuse=tf.AUTO_REUSE)
        output1_2_bn_relu=tf.nn.relu(output1_2_bn,name='conv1_2_3x3_bn/relu')
        output1_3=self.conv(output1_2_bn_relu,3, 3, 64, 1, 1, padding='SAME', relu=False, name='conv1_3_3x3',training=training)
        output1_3_bn=tf.layers.batch_normalization(output1_3,momentum=0.95,epsilon=1e-5,training=training,name='conv1_3_3x3_bn',reuse=tf.AUTO_REUSE)
        output1_3_bn_relu=tf.nn.relu(output1_3_bn,name='conv1_3_3x3/relu')
        max_pool1_1=tf.nn.max_pool(output1_3_bn_relu,ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1_3x3_s2',data_format='NHWC')
        output1_4=self.conv(max_pool1_1,1, 1, 128, 1, 1, padding='SAME',relu=False, name='conv2_1_1x1_proj',training=training)
        output1_4_bn=tf.layers.batch_normalization(output1_4,momentum=0.95,epsilon=1e-5,training=training,name='conv1_4_3x3_bn',reuse=tf.AUTO_REUSE)
        
        output2_1=self.conv(max_pool1_1,1, 1, 32, 1, 1, padding='SAME', relu=False, name='conv2_1_1x1_reduce',training=training)
        output2_1_bn=tf.layers.batch_normalization(output2_1,momentum=0.95,epsilon=1e-5,training=training,name='conv2_1_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output2_1_bn_relu=tf.nn.relu(output2_1_bn,name='conv2_1_1x1_reduce_bn/relu')
        output2_2=self.conv(output2_1_bn_relu,3, 3, 32, 1, 1, padding='SAME', relu=False, name='conv2_1_3x3',training=training)
        output2_2_bn=tf.layers.batch_normalization(output2_2,momentum=0.95,epsilon=1e-5,training=training,name='conv2_1_3x3_bn',reuse=tf.AUTO_REUSE)
        output2_2_bn_relu=tf.nn.relu(output2_2_bn,name='conv2_1_3x3_bn/relu')
        output2_3=self.conv(output2_2_bn_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv2_1_1x1_increase',training=training)
        output2_3_bn=tf.layers.batch_normalization(output2_3,momentum=0.95,epsilon=1e-5,training=training,name='conv2_1_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum1=tf.add(output1_4_bn,output2_3_bn,name='conv2_1')
        sum1_relu=tf.nn.relu(sum1,name='conv2_1/relu')
        
        output3_1=self.conv(sum1_relu,1, 1, 32, 1, 1, padding='SAME', relu=False, name='conv2_2_1x1_reduce',training=training)
        output3_1_bn=tf.layers.batch_normalization(output3_1,momentum=0.95,epsilon=1e-5,training=training,name='conv2_2_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output3_1_bn_relu=tf.nn.relu(output3_1_bn,name='conv2_2_1x1_reduce_bn/relu')
        output3_2=self.conv(output3_1_bn_relu,3, 3, 32, 1, 1, padding='SAME', relu=False, name='conv2_2_3x3',training=training)
        output3_2_bn=tf.layers.batch_normalization(output3_2,momentum=0.95,epsilon=1e-5,training=training,name='conv2_2_3x3_bn',reuse=tf.AUTO_REUSE)
        output3_2_bn_relu=tf.nn.relu(output3_2_bn,name='conv2_2_3x3_bn/relu')
        output3_3=self.conv(output3_2_bn_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv2_2_1x1_increase',training=training)
        output3_3_bn=tf.layers.batch_normalization(output3_3,momentum=0.95,epsilon=1e-5,training=training,name='conv2_2_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum2=tf.add(sum1_relu,output3_3_bn,name='conv2_2')
        sum2_relu=tf.nn.relu(sum2,name='conv2_2/relu')
        
        output4_1=self.conv(sum2_relu,1, 1, 32, 1, 1, padding='SAME', relu=False, name='conv2_3_1x1_reduce',training=training)
        output4_1_bn=tf.layers.batch_normalization(output4_1,momentum=0.95,epsilon=1e-5,training=training,name='conv2_3_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output4_1_bn_relu=tf.nn.relu(output4_1_bn,name='conv2_3_1x1_reduce_bn/relu')
        output4_2=self.conv(output4_1_bn_relu,3, 3, 32, 1, 1, padding='SAME', relu=False, name='conv2_3_3x3',training=training)
        output4_2_bn=tf.layers.batch_normalization(output4_2,momentum=0.95,epsilon=1e-5,training=training,name='conv2_3_3x3_bn',reuse=tf.AUTO_REUSE)
        output4_2_bn_relu=tf.nn.relu(output4_2_bn,name='conv2_3_3x3_bn/relu')
        output4_3=self.conv(output4_2_bn_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv2_3_1x1_increase',training=training)
        output4_3_bn=tf.layers.batch_normalization(output4_3,momentum=0.95,epsilon=1e-5,training=training,name='conv2_3_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum3=tf.add(sum2_relu,output4_3_bn,name='conv2_3')
        sum3_relu=tf.nn.relu(sum3,name='conv2_3/relu')
        output4_4=self.conv(sum3_relu,1, 1, 256, 2, 2, padding='SAME', relu=False, name='conv3_1_1x1_proj',training=training)
        output4_4_bn=tf.layers.batch_normalization(output4_4,momentum=0.95,epsilon=1e-5,training=training,name='conv3_1_1x1_proj_bn',reuse=tf.AUTO_REUSE)
        
        output5_1=self.conv(sum3_relu,1, 1, 64, 2, 2, padding='SAME', relu=False, name='conv3_1_1x1_reduce',training=training)
        output5_1_bn=tf.layers.batch_normalization(output5_1,momentum=0.95,epsilon=1e-5,training=training,name='conv3_1_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output5_1_bn_relu=tf.nn.relu(output5_1_bn,name='conv3_1_1x1_reduce_bn/relu')
        output5_2=self.conv(output5_1_bn_relu,3, 3, 64, 1, 1, padding='SAME', relu=False, name='conv3_1_3x3',training=training)
        output5_2_bn=tf.layers.batch_normalization(output5_2,momentum=0.95,epsilon=1e-5,training=training,name='conv3_1_3x3_bn',reuse=tf.AUTO_REUSE)
        output5_2_bn_relu=tf.nn.relu(output5_2_bn,name='conv3_1_3x3_bn/relu')
        output5_3=self.conv(output5_2_bn_relu,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_1_1x1_increase',training=training)
        output5_3_bn=tf.layers.batch_normalization(output5_3,momentum=0.95,epsilon=1e-5,training=training,name='conv3_1_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum4=tf.add(output4_4_bn,output5_3_bn,name='conv3_1')
        sum4_relu=tf.nn.relu(sum4,name='conv3_1/relu')
        output5_4=self.conv(sum4_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv3_1_sub2_proj',training=training)
        output5_4_bn=tf.layers.batch_normalization(output5_4,momentum=0.95,epsilon=1e-5,training=training,name='conv3_1_sub2_proj_bn',reuse=tf.AUTO_REUSE)
        
        new_h,new_w=sum4_relu.get_shape().as_list()[1:3]
        sum4_relu_interp=tf.image.resize_bilinear(sum4_relu, size=[int(new_h*0.5),int(new_w*0.5)], align_corners=True,name='conv3_1_sub4')
        
        output6_1=self.conv(sum4_relu_interp,1, 1, 64, 1, 1, padding='SAME', relu=False, name='conv3_2_1x1_reduce',training=training)
        output6_1_bn=tf.layers.batch_normalization(output6_1,momentum=0.95,epsilon=1e-5,training=training,name='conv3_2_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output6_1_bn_relu=tf.nn.relu(output6_1_bn,name='conv3_2_1x1_reduce_bn/relu')
        output6_2=self.conv(output6_1_bn_relu,3, 3, 64, 1, 1, padding='SAME', relu=False, name='conv3_2_3x3',training=training)
        output6_2_bn=tf.layers.batch_normalization(output6_2,momentum=0.95,epsilon=1e-5,training=training,name='conv3_2_3x3_bn',reuse=tf.AUTO_REUSE)
        output6_2_bn_relu=tf.nn.relu(output6_2_bn,name='conv3_2_3x3_bn/relu')
        output6_3=self.conv(output6_2_bn_relu,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_2_1x1_increase',training=training)
        output6_3_bn=tf.layers.batch_normalization(output6_3,momentum=0.95,epsilon=1e-5,training=training,name='conv3_2_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum5=tf.add(sum4_relu_interp,output6_3_bn,name='conv3_2')
        sum5_relu=tf.nn.relu(sum5,name='conv3_2/relu')
        
        output7_1=self.conv(sum5_relu,1, 1, 64, 1, 1, padding='SAME', relu=False, name='conv3_3_1x1_reduce',training=training)
        output7_1_bn=tf.layers.batch_normalization(output7_1,momentum=0.95,epsilon=1e-5,training=training,name='conv3_3_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output7_1_bn_relu=tf.nn.relu(output7_1_bn,name='conv3_3_1x1_reduce_bn/relu')
        output7_2=self.conv(output7_1_bn_relu,3, 3, 64, 1, 1, padding='SAME', relu=False, name='conv3_3_3x3',training=training)
        output7_2_bn=tf.layers.batch_normalization(output7_2,momentum=0.95,epsilon=1e-5,training=training,name='conv3_3_3x3_bn',reuse=tf.AUTO_REUSE)
        output7_2_bn_relu=tf.nn.relu(output7_2_bn,name='conv3_3_3x3_bn/relu')
        output7_3=self.conv(output7_2_bn_relu,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_3_1x1_increase',training=training)
        output7_3_bn=tf.layers.batch_normalization(output7_3,momentum=0.95,epsilon=1e-5,training=training,name='conv3_3_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum6=tf.add(sum5_relu,output7_3_bn,name='conv3_3')
        sum6_relu=tf.nn.relu(sum6,name='conv3_3/relu')
        
        output8_1=self.conv(sum6_relu,1, 1, 64, 1, 1, padding='SAME', relu=False, name='conv3_4_1x1_reduce',training=training)
        output8_1_bn=tf.layers.batch_normalization(output8_1,momentum=0.95,epsilon=1e-5,training=training,name='conv3_4_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output8_1_bn_relu=tf.nn.relu(output8_1_bn,name='conv3_4_1x1_reduce_bn')
        output8_2=self.conv(output8_1_bn_relu,3, 3, 64, 1, 1, padding='SAME', relu=False, name='conv3_4_3x3',training=training)
        output8_2_bn=tf.layers.batch_normalization(output8_2,momentum=0.95,epsilon=1e-5,training=training,name='conv3_4_3x3_bn',reuse=tf.AUTO_REUSE)
        output8_2_bn_relu=tf.nn.relu(output8_2_bn,name='conv3_4_3x3_bn/relu')
        output8_3=self.conv(output8_2_bn_relu,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv3_4_1x1_increase',training=training)
        output8_3_bn=tf.layers.batch_normalization(output8_3,momentum=0.95,epsilon=1e-5,training=training,name='conv3_4_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum7=tf.add(sum6_relu,output8_3_bn,name='conv3_4')
        sum7_relu=tf.nn.relu(sum7,name='conv3_4/relu')
        output8_4=self.conv(sum7_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_1_1x1_proj',training=training)
        output8_4_bn=tf.layers.batch_normalization(output8_4,momentum=0.95,epsilon=1e-5,training=training,name='conv4_1_1x1_proj_bn',reuse=tf.AUTO_REUSE)
        
        output9_1=self.conv(sum7_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv4_1_1x1_reduce',training=training)
        output9_1_bn=tf.layers.batch_normalization(output9_1,momentum=0.95,epsilon=1e-5,training=training,name='conv4_1_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output9_1_bn_relu=tf.nn.relu(output9_1_bn,name='conv4_1_1x1_reduce_bn/relu')
        output9_2=self.atrous_conv(output9_1_bn_relu,3, 3, 128, 2, padding='SAME', relu=False, name='conv4_1_3x3',training=training)
        output9_2_bn=tf.layers.batch_normalization(output9_2,momentum=0.95,epsilon=1e-5,training=training,name='conv4_1_3x3_bn',reuse=tf.AUTO_REUSE)
        output9_2_bn_relu=tf.nn.relu(output9_2_bn,name='conv4_1_3x3_bn/relu')
        output9_3=self.conv(output9_2_bn_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_1_1x1_increase',training=training)
        output9_3_bn=tf.layers.batch_normalization(output9_3,momentum=0.95,epsilon=1e-5,training=training,name='conv4_1_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum8=tf.add(output8_4_bn,output9_3_bn,name='conv4_1')
        sum8_relu=tf.nn.relu(sum8,name='conv4_1/relu')
        
        output10_1=self.conv(sum8_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv4_2_1x1_reduce',training=training)
        output10_1_bn=tf.layers.batch_normalization(output10_1,momentum=0.95,epsilon=1e-5,training=training,name='conv4_2_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output10_1_bn_relu=tf.nn.relu(output10_1_bn,name='conv4_2_1x1_reduce_bn/relu')
        output10_2=self.atrous_conv(output10_1_bn_relu,3, 3, 128, 2, padding='SAME', relu=False, name='conv4_2_3x3',training=training)
        output10_2_bn=tf.layers.batch_normalization(output10_2,momentum=0.95,epsilon=1e-5,training=training,name='conv4_2_3x3_bn',reuse=tf.AUTO_REUSE)
        output10_2_bn_relu=tf.nn.relu(output10_2_bn,name='conv4_2_3x3_bn/relu')
        output10_3=self.conv(output10_2_bn_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_2_1x1_increase',training=training)
        output10_3_bn=tf.layers.batch_normalization(output10_3,momentum=0.95,epsilon=1e-5,training=training,name='conv4_2_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum9=tf.add(sum8_relu,output10_3_bn,name='conv4_2')
        sum9_relu=tf.nn.relu(sum9,name='conv4_2/relu')
        
        output11_1=self.conv(sum9_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv4_3_1x1_reduce',training=training)
        output11_1_bn=tf.layers.batch_normalization(output11_1,momentum=0.95,epsilon=1e-5,training=training,name='conv4_3_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output11_1_bn_relu=tf.nn.relu(output11_1_bn,name='conv4_3_1x1_reduce_bn/relu')
        output11_2=self.atrous_conv(output11_1_bn_relu,3, 3, 128, 2, padding='SAME', relu=False, name='conv4_3_3x3',training=training)
        output11_2_bn=tf.layers.batch_normalization(output11_2,momentum=0.95,epsilon=1e-5,training=training,name='conv4_3_3x3_bn',reuse=tf.AUTO_REUSE)
        output11_2_bn_relu=tf.nn.relu(output11_2_bn,name='conv4_3_3x3_bn/relu')
        output11_3=self.conv(output11_2_bn_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_3_1x1_increase',training=training)
        output11_3_bn=tf.layers.batch_normalization(output11_3,momentum=0.95,epsilon=1e-5,training=training,name='conv4_3_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum10=tf.add(sum9_relu,output11_3_bn,name='conv4_3')
        sum10_relu=tf.nn.relu(sum10,name='conv4_3/relu')
        
        output12_1=self.conv(sum10_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv4_4_1x1_reduce',training=training)
        output12_1_bn=tf.layers.batch_normalization(output12_1,momentum=0.95,epsilon=1e-5,training=training,name='conv4_4_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output1_1_bn_relu=tf.nn.relu(output1_1_bn,name='conv4_4_1x1_reduce_bn/relu')
        output12_2=self.atrous_conv(output12_1_bn,3, 3, 128, 2, padding='SAME', relu=False, name='conv4_4_3x3',training=training)
        output12_2_bn=tf.layers.batch_normalization(output12_2,momentum=0.95,epsilon=1e-5,training=training,name='conv4_4_3x3_bn',reuse=tf.AUTO_REUSE)
        output12_2_bn_relu=tf.nn.relu(output12_2_bn,name='conv4_4_3x3_bn/relu')
        output12_3=self.conv(output12_2_bn_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_4_1x1_increase',training=training)
        output12_3_bn=tf.layers.batch_normalization(output12_3,momentum=0.95,epsilon=1e-5,training=training,name='conv4_4_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum11=tf.add(sum10_relu,output12_3_bn,name='conv4_4')
        sum11_relu=tf.nn.relu(sum11,name='conv4_4/relu')
        
        output13_1=self.conv(sum11_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv4_5_1x1_reduce',training=training)
        output13_1_bn=tf.layers.batch_normalization(output13_1,momentum=0.95,epsilon=1e-5,training=training,name='conv4_5_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output13_1_bn_relu=tf.nn.relu(output13_1_bn,name='conv4_5_1x1_reduce_bn/relu')
        output13_2=self.atrous_conv(output13_1_bn_relu,3, 3, 128, 2, padding='SAME', relu=False, name='conv4_5_3x3',training=training)
        output13_2_bn=tf.layers.batch_normalization(output13_2,momentum=0.95,epsilon=1e-5,training=training,name='conv4_5_3x3_bn',reuse=tf.AUTO_REUSE)
        output13_2_bn_relu=tf.nn.relu(output13_2_bn,name='conv4_5_3x3_bn/relu')
        output13_3=self.conv(output13_2_bn_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_5_1x1_increase',training=training)
        output13_3_bn=tf.layers.batch_normalization(output13_3,momentum=0.95,epsilon=1e-5,training=training,name='conv4_5_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum12=tf.add(sum11_relu,output13_3_bn,name='conv4_5')
        sum12_relu=tf.nn.relu(sum12,name='conv4_5/relu')
        
        output14_1=self.conv(sum12_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv4_6_1x1_reduce',training=training)
        output14_1_bn=tf.layers.batch_normalization(output14_1,momentum=0.95,epsilon=1e-5,training=training,name='conv4_6_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output14_1_bn_relu=tf.nn.relu(output14_1_bn,name='conv4_6_1x1_reduce_bn/relu')
        output14_2=self.atrous_conv(output14_1_bn_relu,3, 3, 128, 2, padding='SAME', relu=False, name='conv4_6_3x3',training=training)
        output14_2_bn=tf.layers.batch_normalization(output14_2,momentum=0.95,epsilon=1e-5,training=training,name='conv4_6_3x3_bn',reuse=tf.AUTO_REUSE)
        output14_2_bn_relu=tf.nn.relu(output14_2_bn,name='conv4_6_3x3_bn/relu')
        output14_3=self.conv(output14_2_bn_relu,1, 1, 512, 1, 1, padding='SAME', relu=False, name='conv4_6_1x1_increase',training=training)
        output14_3_bn=tf.layers.batch_normalization(output14_3,momentum=0.95,epsilon=1e-5,training=training,name='conv4_6_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum13=tf.add(sum12_relu,output14_3_bn,name='conv4_6')
        sum13_relu=tf.nn.relu(sum13,name='conv4_6/relu')
        output14_4=self.conv(sum13_relu,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_1_1x1_proj',training=training)
        output14_4_bn=tf.layers.batch_normalization(output14_4,momentum=0.95,epsilon=1e-5,training=training,name='conv5_1_1x1_proj_bn',reuse=tf.AUTO_REUSE)
        
        output15_1=self.conv(sum13_relu,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv5_1_1x1_reduce',training=training)
        output15_1_bn=tf.layers.batch_normalization(output15_1,momentum=0.95,epsilon=1e-5,training=training,name='conv5_1_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output15_1_bn_bn_relu=tf.nn.relu(output15_1_bn,name='conv5_1_1x1_reduce_bn/relu')
        output15_2=self.atrous_conv(output15_1_bn_bn_relu,3, 3, 256, 4, padding='SAME', relu=False, name='conv5_1_3x3',training=training)
        output15_2_bn=tf.layers.batch_normalization(output15_2,momentum=0.95,epsilon=1e-5,training=training,name='conv5_1_3x3_bn',reuse=tf.AUTO_REUSE)
        output15_2_bn_relu=tf.nn.relu(output15_2_bn,name='conv5_1_3x3_bn/relu')
        output15_3=self.conv(output15_2_bn_relu,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_1_1x1_increase',training=training)
        output15_3_bn=tf.layers.batch_normalization(output15_3,momentum=0.95,epsilon=1e-5,training=training,name='conv5_1_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum14=tf.add(output14_4_bn,output15_3_bn,name='conv5_1')
        sum14_relu=tf.nn.relu(sum14,name='conv5_1/relu')
        
        output16_1=self.conv(sum14_relu,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv5_2_1x1_reduce',training=training)
        output16_1_bn=tf.layers.batch_normalization(output16_1,momentum=0.95,epsilon=1e-5,training=training,name='conv5_2_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output16_1_bn_relu=tf.nn.relu(output16_1_bn,name='conv5_2_1x1_reduce_bn/relu')
        output16_2=self.atrous_conv(output16_1_bn,3, 3, 256, 4, padding='SAME', relu=False, name='conv5_2_3x3',training=training)
        output16_2_bn=tf.layers.batch_normalization(output16_2,momentum=0.95,epsilon=1e-5,training=training,name='conv5_2_3x3_bn',reuse=tf.AUTO_REUSE)
        output16_2_bn_relu=tf.nn.relu(output16_2_bn,name='conv5_2_3x3_bn/relu')
        output16_3=self.conv(output16_2_bn,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_2_1x1_increase',training=training)
        output16_3_bn=tf.layers.batch_normalization(output16_3,momentum=0.95,epsilon=1e-5,training=training,name='conv5_2_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum15=tf.add(sum14_relu,output16_3_bn,name='conv5_2')
        sum15_relu=tf.nn.relu(sum15,name='conv5_2/relu')
        
        output17_1=self.conv(sum15_relu,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv5_3_1x1_reduce',training=training)
        output17_1_bn=tf.layers.batch_normalization(output17_1,momentum=0.95,epsilon=1e-5,training=training,name='conv5_3_1x1_reduce_bn',reuse=tf.AUTO_REUSE)
        output17_1_bn_bn_relu=tf.nn.relu(output17_1_bn,name='conv5_3_1x1_reduce_bn/relu')
        output17_2=self.atrous_conv(output17_1_bn_bn_relu,3, 3, 256, 4, padding='SAME', relu=False, name='conv5_3_3x3',training=training)
        output17_2_bn=tf.layers.batch_normalization(output17_2,momentum=0.95,epsilon=1e-5,training=training,name='conv5_3_3x3_bn',reuse=tf.AUTO_REUSE)
        output17_2_bn_relu=tf.nn.relu(output17_2_bn,name='conv5_3_3x3_bn/relu')
        output17_3=self.conv(output17_2_bn_relu,1, 1, 1024, 1, 1, padding='SAME', relu=False, name='conv5_3_1x1_increase',training=training)
        output17_3_bn=tf.layers.batch_normalization(output17_3,momentum=0.95,epsilon=1e-5,training=training,name='conv5_3_1x1_increase_bn',reuse=tf.AUTO_REUSE)
        sum16=tf.add(sum15_relu,output17_3_bn,name='conv5_3')
        sum16_relu=tf.nn.relu(sum16,name='conv5_3/relu')
        
        h,w=sum16_relu.get_shape().as_list()[1:3]
        avg_pool_1=tf.nn.avg_pool(sum16_relu,ksize=[1, h, w, 1],strides=[1, h, w, 1],padding='VALID',name='conv5_3_pool1',data_format='NHWC')
        avg_pool_1_resize=tf.image.resize_bilinear(avg_pool_1, size=[h,w], align_corners=True, name='conv5_3_pool1_interp')
        avg_pool_2=tf.nn.avg_pool(sum16_relu,ksize=[1, h/2, w/2, 1],strides=[1, h/2, w/2, 1],padding='VALID',name='conv5_3_pool2',data_format='NHWC')
        avg_pool_2_resize=tf.image.resize_bilinear(avg_pool_2, size=[h,w], align_corners=True, name='conv5_3_pool2_interp')
        avg_pool_3=tf.nn.avg_pool(sum16_relu,ksize=[1, h/3, w/3, 1],strides=[1, h/3, w/3, 1],padding='VALID',name='conv5_3_pool3',data_format='NHWC')
        avg_pool_3_resize=tf.image.resize_bilinear(avg_pool_3, size=[h,w], align_corners=True, name='conv5_3_pool3_interp')
        avg_pool_4=tf.nn.avg_pool(sum16_relu,ksize=[1, h/6, w/6, 1],strides=[1, h/6, w/6, 1],padding='VALID',name='conv5_3_pool4',data_format='NHWC')
        avg_pool_4_resize=tf.image.resize_bilinear(avg_pool_4, size=[h,w], align_corners=True, name='conv5_3_pool4_interp')
        sum17=tf.add_n([sum16_relu,avg_pool_1_resize,avg_pool_2_resize,avg_pool_3_resize,avg_pool_4_resize],name='conv5_3_sum')
        
        output17_1=self.conv(sum17,1, 1, 256, 1, 1, padding='SAME', relu=False, name='conv5_4_k1',training=training)
        output17_1_bn=tf.layers.batch_normalization(output17_1,momentum=0.95,epsilon=1e-5,training=training,name='conv5_4_k1_bn',reuse=tf.AUTO_REUSE)
        output17_1_bn_relu=tf.nn.relu(output17_1_bn,name='conv5_4_k1_bn/relu')
        output17_1_interp=tf.image.resize_bilinear(output17_1_bn_relu, size=[int(h*2),int(w*2)], align_corners=True,name='conv5_4_interp')
        output17_2=self.atrous_conv(output17_1_interp,3, 3, 128, 2, padding='SAME', relu=False, name='conv_sub4',training=training)
        output17_2_bn=tf.layers.batch_normalization(output17_2,momentum=0.95,epsilon=1e-5,training=training,name='conv_sub4_bn',reuse=tf.AUTO_REUSE)
        sum18=tf.add(output5_4_bn,output17_2_bn,name='sub24_sum')
        sum18_relu=tf.nn.relu(sum18,name='sub24_sum/relu')
        
        output18_1_interp=tf.image.resize_bilinear(sum18_relu, size=[int(h*4),int(w*4)], align_corners=True,name='sub24_sum_interp')
        output18_2=self.atrous_conv(output18_1_interp,3, 3, 128, 2, padding='SAME', relu=False, name='conv_sub2',training=training)
        output18_2_bn=tf.layers.batch_normalization(output18_2,momentum=0.95,epsilon=1e-5,training=training,name='conv_sub2_bn',reuse=tf.AUTO_REUSE)

        output19_1=self.conv(bottom,3, 3, 32, 2, 2, padding='SAME', relu=False, name='conv1_sub1',training=training)
        output19_1_bn=tf.layers.batch_normalization(output19_1,momentum=0.95,epsilon=1e-5,training=training,name='conv1_sub1_bn',reuse=tf.AUTO_REUSE)
        output19_1_bn_relu=tf.nn.relu(output19_1_bn,name='conv1_sub1_bn/relu')
        output19_2=self.conv(output19_1_bn,3, 3, 32, 2, 2, padding='SAME', relu=False, name='conv2_sub1',training=training)
        output19_2_bn=tf.layers.batch_normalization(output19_2,momentum=0.95,epsilon=1e-5,training=training,name='conv2_sub1_bn',reuse=tf.AUTO_REUSE)
        output19_2_bn_relu=tf.nn.relu(output19_2_bn,name='conv2_sub1_bn/relu')
        output19_3=self.conv(output19_2_bn_relu,3, 3, 64, 2, 2, padding='SAME', relu=False, name='conv3_sub1',training=training)
        output19_3_bn=tf.layers.batch_normalization(output19_3,momentum=0.95,epsilon=1e-5,training=training,name='conv3_sub1_bn',reuse=tf.AUTO_REUSE)
        output19_3_bn_relu=tf.nn.relu(output19_3_bn,name='conv3_sub1_bn/relu')
        output19_4=self.conv(output19_3_bn_relu,1, 1, 128, 1, 1, padding='SAME', relu=False, name='conv3_sub1_proj',training=training)
        output19_4_bn=tf.layers.batch_normalization(output19_4,momentum=0.95,epsilon=1e-5,training=training,name='conv3_sub1_proj_bn',reuse=tf.AUTO_REUSE)
        
        sum19=tf.add(output18_2_bn,output19_4_bn,name='sub12_sum')
        sum19_interp=tf.image.resize_bilinear(sum19, size=[int(h*8),int(w*8)], align_corners=True,name='sub12_sum_interp')
        output20_1=self.conv(sum19_interp,1, 1, 32, 1, 1, padding='SAME', relu=False, name='conv4_proj',training=training)
        #sum20_interp=tf.image.resize_bilinear(output20_1, size=[int(h*32),int(w*32)], align_corners=True,name='sub12_sum_interp')#与原图等大
        return output20_1
    
    def cost_vol(self, left, right, max_disp=192):
        with tf.variable_scope('cost_vol',reuse=tf.AUTO_REUSE):
            shape = tf.shape(right)
            right_tensor = keras.backend.spatial_2d_padding(right, padding=((0, 0), (max_disp // 2, 0)))#第三维度补96个零
            disparity_costs = []
            for d in reversed(range(max_disp // 2)):#range倒过来
                left_tensor_slice = left
                right_tensor_slice = tf.slice(right_tensor, begin=[0, 0, d, 0], size=shape)
                right_tensor_slice.set_shape(tf.TensorShape([None, None, None, 32]))#似乎没必要，难道是降维？
                cost = tf.concat([left_tensor_slice, right_tensor_slice], axis=3)
                disparity_costs.append(cost)
            cost_vol = tf.stack(disparity_costs, axis=1)
        return cost_vol#B*96*H*W*64
    

    def output(self, outputs):
        disps = []
        for i, output in enumerate(outputs):
            squeeze = tf.squeeze(output, [4])#移除第五维度,B*96*H*W
            transpose = tf.transpose(squeeze, [0, 2, 3, 1])#B*H*W*96
            upsample = tf.transpose(tf.image.resize_images(transpose, self.image_size_tf), [0, 3, 1, 2])#(batch, depth, H, W)
            disps.append(self.soft_arg_min(upsample, 'soft_arg_min_%d' % (i+1)))
        return disps #B*H*W*3个
    
    def _smooth_l1_loss(self, disps_pred, disps_targets, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = disps_pred - disps_targets
        abs_in_box_diff = tf.clip_by_value(tf.abs(box_diff),1e-15,1e10)
        smoothL1_sign = tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2))
        in_loss_box = tf.pow(abs_in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box))
        return loss_box

    def soft_arg_min(self,filtered_cost_volume, name):
        with tf.variable_scope(name):
            #input.shape (batch, depth, H, W)
            # softargmin to disp image, outsize of (B, H, W)

            #print('filtered_cost_volume:',filtered_cost_volume.shape)
            probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                               axis=1, name='prob_volume')#B*H*W*Depth
                #print('probability_volume:',probability_volume.shape)
            volume_shape = tf.shape(probability_volume)
            soft_1d = tf.cast(tf.range(0, volume_shape[1], dtype=tf.int32),tf.float32)
            soft_4d = tf.tile(soft_1d, tf.stack([volume_shape[0] * volume_shape[2] * volume_shape[3]]))#复制
            soft_4d = tf.reshape(soft_4d, [volume_shape[0], volume_shape[2], volume_shape[3], volume_shape[1]])
            soft_4d = tf.transpose(soft_4d, [0, 3, 1, 2])
            estimated_disp_image = tf.reduce_sum(soft_4d * probability_volume, axis=1)#B*H*W
            print(estimated_disp_image.shape)
            #estimated_disp_image = tf.expand_dims(estimated_disp_image, axis=3)
            return estimated_disp_image

    def train(self, left, right, dataL):
        _,loss = self.sess.run([self.train_op, self.loss],
            feed_dict={self.left: left, self.right: right, self.dataL:dataL})
        return loss

    def test(self, left, right, dataL):
        loss = self.sess.run(self.loss,
            feed_dict={self.left: left, self.right: right, self.dataL:dataL})
        return loss

if __name__ == '__main__':
    with tf.Session() as sess:
        model = Model(sess)
