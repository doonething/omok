
간단한 tensorflow wrapper 

test 디렉토리에서 실행 가능


```python

import tensorflow as tf

import sys
sys.path.append('../src')

from omok.model import Model
import helper

#  테스트용 무작위 features 생성
width = 12
size  = 40
features_set = []
targets_set  = []
for _ in range (50) :
		features , target = helper.make_random_feature_and_target ( width, size)
		features_set.append(features)
		targets_set .append(target)

#  모델 생성 
#  add_layer 함수로 층을 추가
mock = Model()
mock.add_layer(shape=[None, width*width* size] )
mock.add_layer(100, act_func=tf.nn.relu, stddev=.02)
mock.add_layer(40 , act_func=tf.nn.relu, stddev=.02)
mock.add_layer(width * width, act_func=tf.nn.softmax)
mock.init( features_set, targets_set, save_dir='save/f')


# 학습
mock.loop_cnt = 15000
mock.learnning_rate = .003
mock.is_check_loss_threshold = True
mock.loss_threshold = .1
mock.fit()

# 목표값과 비교
print mock.get_acc()
```
