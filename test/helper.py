'''
Created on 2017. 1. 11.

@author: bi
'''

import numpy as np
import unittest
from omok.model import Model

def make_random_one_hot_vector (size):
    v = np.zeros(size)
    i = np.random.randint(0,size-1)
    v[i] = 1.
    return v
    
def get_index_one_hot ( v):
    for x, i in zip (v, range(v.__len__() ) ) :
        if x == 1 : return i
    return -1
 
def make_random_feature_and_target(width, size):
    features = np.random.rand(width, width, size)
    features = features.reshape ( [ width*width * size])
    target   = make_random_one_hot_vector(width*width)
    return features, target

def make_random_features_set_and_targets_set ( width, feature_size, set_size):
    features_set = []
    targets_set  = []
    for _ in range ( set_size ) :
        f , t = make_random_feature_and_target ( width, feature_size)
        features_set .append( f)
        targets_set  .append( t)
    return features_set, targets_set

class Deco ( Model) :
    def one_step(self, loop_index, sess, feed):
        Model.one_step(self, loop_index, sess, feed)
        if loop_index % 100 == 0 :
            loss = self.run ( self.loss)
            print (' %5d %.3f  '% ( loop_index, loss )   )


class HelerTest ( unittest.TestCase):
    def test_make_random_features(self):
        width = 12
        size  = 40
        features = np.random.rand(width, width, size)
        self.assertEqual((12,12, 40), features.shape )
    
    def test_numpy_zeros(self):
        width = 12
        vector = np.zeros ( width * width)
        for x in vector:
            assert 0 == x
    
    def test_make_random_one_hot_vector(self):
        size = np.random.randint(1, 100)
        v = make_random_one_hot_vector(size)
        cnt = 0
        for x in v :
            if x == 1 : cnt += 1
        assert cnt == 1
        
    



