'''
Created on 2017. 1. 10.

@author: bi
'''
import unittest

import numpy as np

import tensorflow as tf

from omok.model import Model, _Saver
import os
import helper


class Test(unittest.TestCase):

    def setUp(self):
        self.mock = Model()

    def tearDown(self):
        self.mock.close()

    def test_add_layer(self):
        mock = self.mock
        mock.add_layer(3)
        
    def test_add_layer_shape(self):
        mock = self.mock
        mock.add_layer( shape= [None,12*12, 5])
    
    def test_for_features_net(self):
        mock = self.mock
        width = 12
        size  = 3
        mock.add_layer(shape=[None, width*width* size])
        mock.add_layer(100)
        mock.add_layer(width * width)
        
    def test_reshape(self):
        width = 12
        size  = 40
        features = np.random.rand(width, width, size)
        features = features.reshape ( [ width*width * size])
        
    def test_tf_reduce_mean_of_boolean_list (self):
        x = [True, True, True, False]
        try :
            with tf.Session() as s :
                acc = tf.reduce_mean ( tf.cast(x, tf.float32 ) )
                self.assertEquals (.75, s.run(acc))
        finally :
            s.close()
    
    def test_acc(self):
        out    = np.zeros([ 5, 3 ])
        target = np.zeros([ 5, 3 ])
        out    [:5,0] = 1
        target [:4,0] = 1
        target [4,1] = 1
        try :
            with tf.Session() as s :
                acc = self.mock.get_acc(out, target) 
                self.assertAlmostEquals ( .8, acc  )
        finally : s.close()

# class Slow ( ):   
class Slow ( unittest.TestCase ):   
    
    def setUp(self):
        self.mock = Model()

    def tearDown(self):
        self.mock.close()
    
    def test_xor(self):
        mock = self.mock
        mock.add_layer(2 );
        mock.add_layer(2 ,weights_init_func=tf.random_uniform );
        mock.add_layer(1);
        
        mock.loop_cnt = 60000
        mock.init([ [0,0],[0,1],[1,0],[1,1]] 
                 , [[1],[0],[0],[1]]
                 , save_dir= 'save/xor' 
                )
        mock.fit()
        self.assertGreater ( mock.get_acc(), .99 )


    def test_feature_input(self):
        width = 12
        size  = 40
        features , target = helper.make_random_feature_and_target ( width, size)
        
        mock = self.mock
        mock.add_layer(shape=[None, width*width* size])
        mock.add_layer(100)
        mock.add_layer(width * width)
        
        mock.init([features], [target], save_dir='save/f')
        mock.loop_cnt = 10    
        mock.fit()
    
    def test_one_input(self):
        width = 12
        size  = 40
        features , target = helper.make_random_feature_and_target ( width, size)
        
        mock = self.mock #= helper.Deco()
        mock.add_layer(shape=[None, width*width* size])
        mock.add_layer(100, act_func=tf.nn.sigmoid, stddev = .03 )
        mock.add_layer(50 , act_func=tf.nn.sigmoid, stddev = .03 )
        mock.add_layer(width * width, act_func=tf.nn.softmax)
        mock.init([features], [target], save_dir='save/f')
        
        mock.loop_cnt = 5000
        mock.is_check_loss_threshold = True
        mock.fit()
        self.assertEqual(1. , mock.get_acc() )
    
    def test_multi_input(self):
        width = 12
        size  = 40
        features_set = []
        targets_set  = []
        for _ in range (50) :
            features , target = helper.make_random_feature_and_target ( width, size)
            features_set.append(features)
            targets_set .append(target)
        
        mock = self.mock #  = helper.Deco()
        mock.add_layer(shape=[None, width*width* size] )
        mock.add_layer(100, act_func=tf.nn.relu, stddev=.02)
        mock.add_layer(40 , act_func=tf.nn.relu, stddev=.02)
        mock.add_layer(width * width, act_func=tf.nn.softmax)
        mock.init( features_set, targets_set, save_dir='save/f')
        
        mock.loop_cnt = 15000
        mock.learnning_rate = .003
        mock.is_check_loss_threshold = True
        mock.loss_threshold = .1
        mock.fit()
        
        self.assertGreater(mock.get_acc(), .99)



class _SaverTest(unittest.TestCase):
    def test_dir(self):
        self.assertEquals ( 'dir', os.path.dirname('dir/file'))    
        self.assertEquals ( 'dir2/dir', os.path.dirname('dir2/dir/file'))
    
    def test_python_os_sep (self):
        self.assertEquals ( '/', os.sep)
        
    def test_python_rfind(self ):
        assert 1  == 'abc'.rfind('b')
        assert -1 == 'abc'.rfind('x')
        
    def test_init_dir(self):
        mock = _Saver()
        name = 'tmp_name'
        mock.init_dir( name)
        self.assertEqual(name , mock.dir )
        self.assertEqual(name +'/' + name, mock.path )
        assert os.path.exists(mock.dir)
        os.removedirs(mock.dir)
        
    def test_init_dir_with_dir(self):
        mock = _Saver()
        name = 'dir/tmp_name'
        mock.init_dir( name)
        self.assertEqual('dir' , mock.dir )
        self.assertEqual(name, mock.path )
        assert os.path.exists(mock.dir)
        os.removedirs(mock.dir)
    
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()