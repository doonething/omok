'''
Created on 2017. 1. 21.

@author: bi
'''
import unittest
from omok.omok_model import OmokModel
import helper


class Test(unittest.TestCase):


    def setUp(self):
        self.mock = OmokModel()


    def tearDown(self):
        self.mock.close()


    def test_create(self):
        self.mock.create()
        
    def test_fit(self):
        mock = self.mock
        features, targets = helper.make_random_features_set_and_targets_set(mock.width, mock.feature_size, 50)
        mock.create()
        mock.init(features, targets, 'save/save')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()