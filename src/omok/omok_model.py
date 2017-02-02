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
        self.add_layer(self.width * self.width * self.feature_size)
        self.add_layer(50)
        self.add_layer(self.width * self.width,act_func=tf.nn.softmax )
    
    
