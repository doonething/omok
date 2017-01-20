import unittest as ut

import tensorflow as tf

from omok.model import Model

class Test(ut.TestCase):
# class Test():
    
    def setUp(self):
        ut.TestCase.setUp(self)
        self.mock = Model()
    
    def tearDown(self):
        ut.TestCase.tearDown(self)
        self.mock.close()
    
    # width == height
    def test_add_conv2d_square(self):
        mock = self.mock
        width = 10
        height = width
        input_channel_size = 3
        # 10 * 10 * 3 
        # width = 10 = height 
        # input channel size = 3
        mock.add_layer ( width  * height * input_channel_size )
        
        filter_size = 5
        output_channel_size = 64
        mock.add_conv2d(filter_size, output_channel_size, width,height, input_channel_size)
        
    def test_make_auto_normal_layer(self):
        mock = self.mock
        width = 28
        height = width
        input_channel_size = 5
        output_channel_size = 7
        filter_size = 5
        mock.add_conv2d(filter_size, output_channel_size, width,height, input_channel_size)
        self.assertEquals ( [None,width*height*input_channel_size]
                            , mock.layers[0].get_shape().as_list())
    
        
    def test_add_conv2d_on_conv2d (self):
        mock = self.mock
        width = 28
        height = width
        input_channel_size = 5
        output_channel_size = 7
        filter_size = 5
        mock.add_conv2d(filter_size, output_channel_size, width,height, input_channel_size)
        mock.add_conv2d(filter_size, output_channel_size)
        mock.add_layer(5)
    
    def test_pool(self):
        mock = self.mock
        width = 28
        height = width
        input_channel_size = 5
        output_channel_size = 7
        filter_size = 5
        mock.add_conv2d(filter_size, output_channel_size, width,height, input_channel_size)
        mock.add_pool()
        mock.add_conv2d(filter_size, output_channel_size)
        mock.add_pool()
        mock.add_layer(7)
        
    def test_get_width_height_for_convolution_layer(self):
        mock = self.mock
        width = 28
        height = width
        input_channel_size = 5
        output_channel_size = 7
        filter_size = 5
        mock.add_conv2d(filter_size, output_channel_size, width,height, input_channel_size)
        width_, height_, output_channel_size_ = mock.get_width_height_input_channel_size()
        self.assertEquals(  [width , height , output_channel_size]
                          , [width_, height_, output_channel_size_])
    
        
    def test_add_second_conv2d_layer_without_width_height_input_channel_size(self):
        mock = self.mock
        width = 28
        height = width
        input_channel_size = 5
        output_channel_size = 7
        filter_size = 5
        mock.add_conv2d(filter_size, output_channel_size, width,height, input_channel_size)
        mock.add_conv2d( filter_size, output_channel_size )
        
class Slow ( ut.TestCase):
# class Slow ( ):
    def setUp(self):
        ut.TestCase.setUp(self)
        self.mock = Model()
    
    def tearDown(self):
        ut.TestCase.tearDown(self)
        self.mock.close()
            
    def test_mnist(self):
        mock = self.mock #= Deco()
        width = 28
        height = width
        input_channel_size  = 1
        
        output_channel_size = 7
        filter_size = 5
        mock.add_conv2d(filter_size, output_channel_size, width,height, input_channel_size)
        mock.add_pool()
        mock.add_conv2d(filter_size, output_channel_size)
        mock.add_pool()
        mock.add_layer(1024, stddev=.01)
        mock.add_layer(10, act_func= tf.nn.softmax, stddev=.01)
        
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("data/", one_hot=True)
        b = mnist.train.next_batch(100)
        mock.init(b[0], b[1], 'save/mnist')
        
        mock.is_check_loss_threshold = True
        mock.loss_threshold = .285
        mock.fit()
        self.assertGreater(mock.get_acc(), .94)
            
        b = mnist.test.next_batch(200)
        out = mock.eval(b[0])
        self.assertGreater(mock.get_acc(out, b[1]), .65)
        

if __name__ == '__main__' :
    ut.main()