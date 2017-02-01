# -*- coding: utf-8 -*-

import random
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


allowed_chars = ":q“;b%—8–kezlf.n x…vsaw0t’'3@6-ch&_u1/2,7$g4)ypj?#!i”5(d9o\"mr"


def transform_text(text):
    """Apply all transformations you want here ... :)))))))"""
    return list(filter(lambda x: x > -1, map(allowed_chars.find, text))) + [len(allowed_chars)]

dataset = [
    transform_text(t['text']) for i in (2015, 2016, 2017)
                              for t in json.load(open('data/%d.json' % i))
]


def get_batch(batch_size):
    return random.sample(dataset, batch_size)


class Trumpinator():
    def __init__(self, batch_size, seqlen, nchars):
        self.input = tf.placeholder(tf.int32, [batch_size, seqlen])
        embedded_inputs = tf.one_hot(self.input, nchars)
        
        cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        self.init_state = cell.zero_state(batch_size, tf.float32)

        output, self.output_state = tf.nn.dynamic_rnn(
                cell,
                embedded_inputs,
                initial_state=self.init_state)

        self.output = tf.nn.softmax(slim.fully_connected(output, nchars))

        self.target = tf.placeholder(tf.int32, [batch_size, seqlen])
        embedded_targets = tf.one_hot(self.target, nchars)
        self.loss = tf.reduce_sum(tf.square(embedded_targets-self.output))
        self.train_step = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

def train():

    batch_size = 32
    seqlen = 10
    great_nn = Trumpinator(batch_size, seqlen, len(allowed_chars))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in range(1000):

            X = np.ones((batch_size, seqlen))
            y = np.ones((batch_size, seqlen))

            loss, _ = sess.run([great_nn.loss, great_nn.train_step], feed_dict={
                great_nn.input : X,
                great_nn.target : y
                })
            
            if step % 10 == 0:
                print(loss)



if __name__ == "__main__":
    train()
