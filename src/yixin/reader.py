'''
Created on 2017. 2. 2.

@author: bi
'''

import copy
import os
import traceback


class YixinSave :
    def __init__(self):
        self.width = 0
        self.height = 0
        self.num = 0
        self.blacks = []
        self.whites = []
    
    def read_dir(self, dir):
        inputTargetSet = []
        for f in os.listdir(dir) :
            path = dir + os.sep + f
            if f.startswith('.') : continue
            try :
                inputTargetSet.append( self.read_file(path) )
            except Exception as e :
                traceback.print_exc()
                raise Exception('failed file : ' + path)
        return inputTargetSet 
    
    def read_file(self, path):
        self.read_yixin(path)
        self.make_features()
        return self.make_input_targets()
    
    def make_input_targets(self):
        raise Exception('YixinSave.make_input_targets() is abstract method , use class YixinSaveWhiteWin or YixinSaveBlackWin')
            
    def read_yixin(self, path):
        with open (path) as f :
            lines = f.readlines()
            self.width  = int ( lines[0])
            self.height = int ( lines[1])
            self.num    = int ( lines[2])
            current = self.blacks
            for line in lines [3:]:
                splited = line.split(' ')
                y_index = int ( splited[0])
                x_index = int ( splited[1])
                index   = self.get_index_from_coordinate(y_index, x_index)
                current.append ( index)
                if current == self.blacks :
                    current = self.whites
                else : current = self.blacks
        
    def make_features(self):
        size = self.width * self.height
        none_feature  = self.make_list(size,1)
        black_feature = self.make_list(size,0)
        white_feature = self.make_list(size,0)
        features = []
        features.append([none_feature, black_feature, white_feature])
        for b,w in zip (self.blacks, self.whites) :
            n_ = copy.deepcopy( features[-1][0])
            b_ = copy.deepcopy( features[-1][1])
            w_ = copy.deepcopy( features[-1][2])
            b_[b] = 1
            n_[b] = 0
            features.append ( [n_,b_,w_])
            
            n_ = copy.deepcopy( features[-1][0])
            b_ = copy.deepcopy( features[-1][1])
            w_ = copy.deepcopy( features[-1][2])
            w_[w] = 1
            n_[w] = 0
            features.append ( [n_,b_,w_])
            
        return features 

    
    
    ##########################################
    
    def get_index_from_coordinate(self, y, x):
        return y * self.width + x
    
    def make_list(self, size, value):
        l = []
        for _ in range(size):
            l.append (value)
        return l

    def make_target(self, one_hot_index):
        l = self.make_list( self.width * self.height , 0)
        l [ one_hot_index] = 1
        return l
    
    class InputTarget:
        def __init__(self):
            self.input = []
            self.target = []


class YixinSaveWhiteWin(YixinSave):
    def make_input_targets(self):
        inputTargets = [] #YixinSave.InputTargets()
        features = self.make_features()
        for i, white in zip ( range(1,features.__len__(),2 ) 
                             ,self.whites ):
            inputTarget = YixinSave.InputTarget()
            inputTarget.input  = features[i]
            inputTarget.target = self.make_target( white)
            inputTargets.append( inputTarget )
        return inputTargets
    

class YixinSaveBlackWin(YixinSave):
    def make_input_targets(self):
        inputTargets = [] #YixinSave.InputTargets()
        features = self.make_features()
        for i, black in zip ( range(0,features.__len__(),2 ) 
                             ,self.blacks ):
            inputTarget = YixinSave.InputTarget()
            inputTarget.input  = features[i]
            inputTarget.target = self.make_target( black)
            inputTargets.append( inputTarget )
        return inputTargets
