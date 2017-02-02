'''
Created on 2017. 2. 1.

@author: bi
'''
import unittest
import os

from yixin.reader import YixinSave

class Test(unittest.TestCase):


    def setUp(self):
        self.mock = YixinSave()

    def test_temp_file_write(self):
        path = 'tmp.txt'
        with open ( path, 'w') as f :
            f.write( 'line1\n')
            f.write( 'line2\n')
        with open ( path ) as f :
            lines = f.readlines()
            self.assertEquals ( 'line1', lines[0][:-1])
            self.assertEquals ( 'line2', lines[1][:-1])
        os.remove(path)
        
    def test_read_width_num (self):
        path = 'tmp.txt'
        with open ( path, 'w') as f :
            f.write( '3\n')
            f.write( '5\n')
            f.write( '2\n')
        self.mock.read_yixin(path)
        self.assertEquals ( 3, self.mock.width)
        self.assertEquals ( 5, self.mock.height)
        self.assertEquals ( 2, self.mock.num)
        
    def test_string_split(self):
        s = '7 7'
        splited = s.split(' ')
        self.assertEquals ( '7', splited[0])
        self.assertEquals ( '7', splited[1])
        
    def test_cordinate_to_index(self):
        save = self.mock
        save.width = 5 
        self.assertEquals ( 3, save.get_index_from_coordinate(0, 3))
        self.assertEquals ( 8, save.get_index_from_coordinate(1, 3))
        save.width = 15 
        # 15 * 7 = 105 , 105 + 7 = 112
        self.assertEquals ( 112, save.get_index_from_coordinate(7,7))
        self.assertEquals ( 111, save.get_index_from_coordinate(7,6))
         
    def test_read_first_black_and_white(self):
        path = 'tmp.txt'
        with open ( path, 'w') as f :
            f.write( '10\n10\n2\n7 7\n8 8')
        self.mock.read_yixin(path)
        self.assertEqual([77], self.mock.blacks)
        self.assertEqual([88], self.mock.whites)
    
    def test_last_element(self):
        l = [ 1,2,3 ]
        self.assertEqual(3, l[-1])
        
    def test_make_features (self):
        save = YixinSave()
        save.width = 3
        save.height = 3
        save.blacks = [0,1,2]
        save.whites = [3,4,5]
        features = save.make_features ()
        self.assertEqual([1,1,1,1,1,1,1,1,1], features[0][0] ) # none
        self.assertEqual([0,0,0,0,0,0,0,0,0], features[0][1] ) # black
        self.assertEqual([0,0,0,0,0,0,0,0,0], features[0][2] ) # white
        
        self.assertEqual([0,1,1,1,1,1,1,1,1], features[1][0] ) # none
        self.assertEqual([1,0,0,0,0,0,0,0,0], features[1][1] ) # black
        self.assertEqual([0,0,0,0,0,0,0,0,0], features[1][2] ) # white
        
        self.assertEqual([0,1,1,0,1,1,1,1,1], features[2][0] ) # none
        self.assertEqual([1,0,0,0,0,0,0,0,0], features[2][1] ) # black
        self.assertEqual([0,0,0,1,0,0,0,0,0], features[2][2] ) # white
        
        self.assertEqual([0,0,1,0,1,1,1,1,1], features[3][0] ) # none
        self.assertEqual([1,1,0,0,0,0,0,0,0], features[3][1] ) # black
        self.assertEqual([0,0,0,1,0,0,0,0,0], features[3][2] ) # white
        
        self.assertEqual([0,0,1,0,0,1,1,1,1], features[4][0] ) # none
        self.assertEqual([1,1,0,0,0,0,0,0,0], features[4][1] ) # black
        self.assertEqual([0,0,0,1,1,0,0,0,0], features[4][2] ) # white
        
        self.assertEqual([0,0,0,0,0,1,1,1,1], features[5][0] ) # none
        self.assertEqual([1,1,1,0,0,0,0,0,0], features[5][1] ) # black
        self.assertEqual([0,0,0,1,1,0,0,0,0], features[5][2] ) # white
        
        self.assertEqual([0,0,0,0,0,0,1,1,1], features[6][0] ) # none
        self.assertEqual([1,1,1,0,0,0,0,0,0], features[6][1] ) # black
        self.assertEqual([0,0,0,1,1,1,0,0,0], features[6][2] ) # white
        
    def test_make_input_target_for_white_win(self):
        save = YixinSave()
        save.width = 3
        save.height = 3
        save.blacks = [0,1,2]
        save.whites = [3,4,5]
        input_and_targets  = save.make_input_targets_for_white_win()
        input = input_and_targets[0].input
        self.assertEqual([0,1,1,1,1,1,1,1,1], input[0] ) # none
        self.assertEqual([1,0,0,0,0,0,0,0,0], input[1] ) # black
        self.assertEqual([0,0,0,0,0,0,0,0,0], input[2] ) # white
        self.assertEqual([0,0,0,1,0,0,0,0,0], input_and_targets[0].target)
        
        input = input_and_targets[1].input
        self.assertEqual([0,0,1,0,1,1,1,1,1], input[0] ) # none
        self.assertEqual([1,1,0,0,0,0,0,0,0], input[1] ) # black
        self.assertEqual([0,0,0,1,0,0,0,0,0], input[2] ) # white
        self.assertEqual([0,0,0,0,1,0,0,0,0], input_and_targets[1].target)
    
    def test_make_input_target_for_black_win(self):
        save = self.mock
        save.width = 3
        save.height = 3
        save.blacks = [0,1,2]
        save.whites = [3,4,5]
        input_and_targets  = save.make_input_targets_for_black_win()
        input = input_and_targets[0].input
        self.assertEqual([1,1,1,1,1,1,1,1,1], input[0] ) # none
        self.assertEqual([0,0,0,0,0,0,0,0,0], input[1] ) # black
        self.assertEqual([0,0,0,0,0,0,0,0,0], input[2] ) # white
        self.assertEqual([1,0,0,0,0,0,0,0,0], input_and_targets[0].target)
        
        input = input_and_targets[1].input
        self.assertEqual([0,1,1,0,1,1,1,1,1], input[0] ) # none
        self.assertEqual([1,0,0,0,0,0,0,0,0], input[1] ) # black
        self.assertEqual([0,0,0,1,0,0,0,0,0], input[2] ) # white
        self.assertEqual([0,1,0,0,0,0,0,0,0], input_and_targets[1].target)
    
    
    def test_read_white_win_on_many_files(self):
        inputTargetSet = []
        dir = 'yixin'
        for f in os.listdir(dir) :
            path = dir + os.sep + f
            if f[0] == 'w' :
                inputTargetSet.append( self.mock.read_white_win(path) )


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_file_read']
    unittest.main()