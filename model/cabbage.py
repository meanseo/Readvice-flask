
from pickletools import optimize
import pandas as pd
import tensorflow.compat.v1 as tf
from icecream import ic
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir


class Solution:
    def __init__(self) -> None:
        self.basedir = os.path.join(basedir, 'model')
        self.df = None
        self.x_data = None 
        self.y_data = None 
    
    def preprocess(self):
        self.df = pd.read_csv('./model/data/price_data.csv', encoding='UTF-8')
        ic(self.df)
        xy = np.array(self.df, dtype=np.float32)
        # year,avgTemp,minTemp,maxTemp,rainFall,avgPrice 기후 요소가 4개
        self.x_data = xy[:, 1:-1]
        self.y_data = xy[:, [-1]]
        # ic(self.x_data)
        # ic(self.y_data)                                      

    def create_model(self): # 모델생성
        #텐서모델 초기화(모델템플릿 생성)
        model = tf.global_variables_initializer()
        #확률변수 데이터
        self.preprocess()
        #선형식제작 y = Wx+b
        X = tf.placeholder(tf.float32, shape = [None, 4])
        Y = tf.placeholder(tf.float32, shape = [None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name = "weight")
        b = tf.Variable(tf.random_normal([1]), name = "bias")
        hypothesis = tf.matmul(X, W) + b # WX + b
        #손실함수
        cost = tf.reduce_mean(tf.square(hypothesis -Y))
        #최적화알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        #세션생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #트레이닝 
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                        feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0:
                print('# %d 손실비용: %d'%(step, cost_))
                print('- 배추가격: %d'%(hypo_[0]))
        #모델저장
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt'), global_step=1000)
        print('저장완료')

    def load_model(self, avgTemp, minTemp, maxTemp, rainFall):
        tf.disable_v2_behavior()
        X = tf.placeholder(tf.float32, shape = [None, 4])
        Y = tf.placeholder(tf.float32, shape = [None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name = "weight")
        b = tf.Variable(tf.random_normal([1]), name = "bias")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(self.basedir, 'cabbage', 'cabbage.ckpt-1000'))
            data = [[avgTemp,minTemp,maxTemp,rainFall],]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print(dict)
        return int(dict[0])


if __name__=='__main__':
    tf.disable_v2_behavior()
    s = Solution()
    # s.create_model()
