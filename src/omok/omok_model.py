'''
Created on 2017. 1. 21.

@author: bi
'''

import tensorflow as tf

from omok.model import Model


class OmokModel(Model):
    
    def __init__(self):
        Model.__init__(self)
        self.width = 15
        self.feature_size = 3
        
    def create(self):
        self.add_layer(self.width * self.width * self.feature_size
                       , act_func=tf.nn.relu, stddev=.02)
        self.add_layer(50
                       , act_func=tf.nn.relu, stddev=.02)
        self.add_layer(self.width * self.width,act_func=tf.nn.softmax )
        
    def init_loss(self):
        self.loss = tf.reduce_sum( .5 *  tf.pow(self.target - self.output , 2) )
    
    def one_step(self, loop_index, sess, feed):
        Model.one_step(self, loop_index, sess, feed)
        if self.get_acc() > .95 :
            self.is_stop = True
            print 'skip acc > .95'
    
    
    
