'''
Created on 2017. 1. 10.

@author: bi
'''

import tensorflow as tf
import os

class Model:
    
    def __init__(self):
        self.current = None
        self.is_sess_init = False
        self.target = None
        self.output = None
        self.feed = {}
        self.loop_cnt = 1200
        self.sess     = tf.Session()
        self.saver    = _Saver()
        self.weights  = []
        self.loss_threshold = .0005 
        self.is_stop = False
        self.learnning_rate = .00017
        self.is_check_loss_threshold = False
        self.layers = []
    
    def add_conv2d(self
                , filter_size
                , output_channel_size
                , width              = -1 
                , height             = -1 
                , input_channel_size = -1
                , act_func                    = tf.nn.relu
                , weights_init_func           = tf.truncated_normal
                , dict_weights_init_func_args = { 'stddev':.1 }
                , dict_conv_args              = { 'strides':[1,1,1,1], 'padding':'SAME'} 
                ):
        
        if width == -1 and height == -1 and input_channel_size == -1 :
            width, height, input_channel_size = self.get_width_height_input_channel_size()
            
        w = self.make_tf_variable(  [filter_size,filter_size,input_channel_size, output_channel_size] 
                                  , weights_init_func
                                  , dict_weights_init_func_args)
        
        b = self.make_tf_variable(  [output_channel_size] 
                                  , weights_init_func
                                  , dict_weights_init_func_args)
        
        if self.current is None :
            self.add_layer( width * height * input_channel_size)
            
        reshape = tf.reshape (self.current,[ -1, width, height , input_channel_size ])
        conv2d = tf.nn.conv2d ( reshape, w, **dict_conv_args )
        self.current = act_func ( conv2d + b )
        self.layers.append(self.current)
        
    def make_tf_variable (self, shape, init_func, args, name=None):
        init_val = init_func ( shape, **args )
        return tf.Variable ( init_val , name= name)
    
    def get_width_height_input_channel_size(self):
        shape = self.current.get_shape().as_list()
        # [None, width, height, out_channel] convolution layer 
        if shape.__len__() == 4 :
            return shape[1], shape[2], shape[3]  
    
    def add_pool(self, size=2):
        self.current = tf.nn.max_pool ( self.current, ksize =[1,size,size,1]
            ,strides=[1,size,size,1]
            ,padding='SAME' )
    
    def add_layer(self, neuron_num=-1, name  = None, shape=None
                  ,   act_func               = tf.nn.softplus
                  ,   weights_init_func      = tf.random_normal
                  , **weights_init_func_args  ):
        if self.current is None :
            self.init_input_layer(neuron_num, shape, name)
            return
        in_num = self.get_input_num ( self.current )
        W = tf.Variable ( weights_init_func( [in_num, neuron_num ], **weights_init_func_args ) , name=name)
        b = tf.Variable ( weights_init_func( [neuron_num], **weights_init_func_args )  )

        self.reshape_when_input_layer_have_two_more_ranks()
        current = tf.matmul (self.current, W) + b
        self.current = act_func ( current)
        
        self.weights.append(W)
        self.layers.append(self.current)
        
    def reshape_when_input_layer_have_two_more_ranks(self):
        if self.current.get_shape().__len__() != 2 :
            shape = self.current.get_shape().as_list()
            gob = 1
            for i in range ( 1, shape.__len__() ) :
                gob *= shape[i]
            self.current = tf.reshape ( self.current , [-1, gob] )
        
    def init_input_layer(self, neuron_num=-1, shape=None, name=None):
        if self.current is not None : return
        if shape is None : shape = [None,neuron_num]
        self.current = tf.placeholder ( tf.float32, shape, name=name )
        self.input = self.current
        self.layers.append(self.current)
    
    def get_input_num (self, tensor):
        list = tensor.get_shape().as_list()
        if list.__len__() == 4 : # conv 
            return list[1]*list[2]*list[3]
        return list[1]

    def init (self, input, target, save_dir=None):    
        self.output = self.current
        if input  is not None : self.set_input(input)
        if target is not None : self.set_target(target)
        self.init_loss()
        self.optimize()
        self.init_session()
        self.saver.init(self.sess, save_dir)
        
    def fit(self):
        self.loop ( self.sess, self.feed)
        
    def set_input(self, input):
        self.feed[self.input] = input
    
    def set_target(self, target):
        if self.output is None :
            self.output = self.current 
        if self.target is None :
            self.target = tf.placeholder ( tf.float32, [None , self.output.get_shape().as_list()[1]] , name = 'target')
        self.feed[self.target] = target
    
    def init_loss(self):
        # self.loss = tf.reduce_sum( .5 *  tf.pow(self.target - self.output , 2) )
        
        # entropy 
        self.loss = tf.reduce_mean (tf.reduce_sum ( self.target * -tf.log(self.output), reduction_indices=[1] ) )
        
    def optimize(self):
        #self.train_step = tf.train.GradientDescentOptimizer(0.01). minimize ( self.loss )
        self.train_step = tf.train.AdamOptimizer(self.learnning_rate).minimize(self.loss)

    def init_session(self):
        if not self.is_sess_init :
            if is_old_tensorflow_versioin () :
                self.sess.run(tf.initialize_all_variables())
            else :
                self.sess.run(tf.global_variables_initializer())
            self.is_sess_init = True
    
    def loop (self, s, feed):
        for i in range(self.loop_cnt) :
            self.one_step( i, s, feed)
            if self.is_stop : break
            self.check_loss_threshold()
    
    def check_loss_threshold(self):
        if self.is_check_loss_threshold == False : return    
        loss = self.run ( self.loss)
        if loss < self.loss_threshold : self.is_stop = True 

    def one_step(self, loop_index, sess, feed):
        sess.run ( self.train_step, feed_dict= feed )
        
    def eval(self,x ):
        if self.output is None :
            print ' after doing init(), call eval()'
            return
        feed = {self.input:x}
        ret =  self.sess.run(self.output , feed_dict = feed)
        return ret
    
    def get_acc (self, out=None, target=None) :
        if out    is None : out    = self.output
        if target is None : target = self.target
        out_max    = tf.argmax(out,1)
        target_max = tf.argmax(target,1)
        correct_prediction = tf.equal(out_max, target_max ) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.run ( accuracy )
    
    def eval_list(self, x):
        return self.eval(x)[0].tolist()
    
    def eval_list_round(self, x):
        return self.round(self.eval_list(x))
    
    def round(self, vec ):
        rounded = []
        for x in vec :
            if x > .5 :
                rounded.append ( 1. )
            else : 
                rounded.append ( 0)
        return rounded
    
    def run (self, tensor):
        return self.sess.run ( tensor, feed_dict = self.feed)

    def close(self):
        self.sess.close()
        tf.reset_default_graph()
        
    
    
class _Saver:
    def init(self,sess, name = 'save/save'):
        self.saver = tf.train.Saver()
        self.sess = sess
        self.init_dir(name)
    
    def init_dir (self, name):
        i = name.rfind(os.sep)
        if i < 0 : 
            self.dir  = name
            self.path = name + os.sep + name
        else :
            self.dir  = name [:i]
            self.path = name
        self.makedirs ( self.dir ) 
        
    def makedirs(self, dir):        
        if os.path.exists(dir) : return
        try :
            os.makedirs(dir)
        except OSError as e:
            if e.errno != 17 : # file exist
                raise e
            
    def save (self):
        self.saver.save(self.sess, self.path)
    def load(self):
        self.saver.restore(self.sess, self.path)
        

def is_old_tensorflow_versioin ( old_major_version = 0, old_minor_version = 6 ):
    ver_list = tf.__version__.split ( '.' )
    major = ver_list[0]
    minor = ver_list[1]
    if major <= old_major_version and minor <= old_minor_version :
        return True
    return False
