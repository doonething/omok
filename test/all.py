'''
Created on 2017. 1. 20.

@author: bi
'''

import unittest as u
import sys

from conv import conv_test 
import model


# creating a new test suite
suite = u.TestSuite()
 

# adding a test case
suite.addTest(u.makeSuite(conv_test.Test))
suite.addTest(u.makeSuite(conv_test.Slow))
suite.addTest(u.makeSuite(model.Test))
suite.addTest(u.makeSuite(model.Slow))


u.TextTestRunner().run(suite)