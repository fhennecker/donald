# -*- coding: utf-8 -*-

import random
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


allowed_chars = ":q“;b%—8–kezlf.n x…vsaw0t’'3@6-ch&_u1/2,7$g4)ypj?#!i”5(d9o\"mr"


def transform_text(text):
    """Apply all transformations you want here ... :)))))))"""
    return (list(filter(lambda x: x > -1, map(allowed_chars.find, text))) + [len(allowed_chars)])[:140]

dataset = [
    transform_text(t['text']) for i in (2015, 2016, 2017)
                              for t in json.load(open('data/%d.json' % i))
]


def get_batch(batch_size):
    return random.sample(dataset, batch_size)


class Trumpinator():
    def __init__(self, batch_size, seqlen, nchars):
        self.input = tf.placeholder(tf.int32, [batch_size, seqlen], name='input')
        self.lengths = tf.placeholder(tf.int32, [batch_size], name='lengths')
        embedded_inputs = tf.one_hot(self.input, nchars)
        
        cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        #  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        self.init_state = cell.zero_state(batch_size, tf.float32)

        output, self.output_state = tf.nn.dynamic_rnn(
                cell,
                embedded_inputs,
                sequence_length=self.lengths,
                initial_state=self.init_state)

        self.output = tf.nn.softmax(slim.fully_connected(output, nchars))

        self.mask = tf.placeholder(tf.float32, [batch_size, seqlen], name='mask')
        self.target = tf.placeholder(tf.int32, [batch_size, seqlen], name='target')
        embedded_targets = tf.one_hot(self.target, nchars)
        self.loss = tf.reduce_sum(tf.reduce_sum(tf.square(embedded_targets-self.output), axis=2)*self.mask)
        self.train_step = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)

def train():
    batch_size = 32
    seqlen = 141
    great_nn = Trumpinator(batch_size, seqlen, len(allowed_chars))

    X = np.zeros((batch_size, seqlen), dtype=np.int32)
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for step in range(int(1e10)):
            X[:,:] = 0
            Y[:,:] = 0
            Z[:,:] = 0
            batch = get_batch(batch_size)
            lens = np.array([len(x) for x in batch], dtype=np.int32) - 1
            for i, b in enumerate(batch):
                X[i,:lens[i]] = b[:-1]
                Y[i,:lens[i]] = b[1:]
                Z[i,:lens[i]] = 1

            loss, _ = sess.run([great_nn.loss, great_nn.train_step], feed_dict={
                great_nn.input : X,
                great_nn.target: Y,
                great_nn.mask: Z,
                great_nn.lengths: lens,
            })
            
            if step % 10 == 0:
                print(loss)

def freerun():
    great_nn = Trumpinator(1, 1, len(allowed_chars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore()

        hidden_state = None
        done = False
        previous_char = 'i'
        while not done:
            feed_dict = {great_nn.input: [[allowed_chars.index(previous_char)]], 
                    great_nn.lengths:[1]}
            if hidden_state is not None:
                feed_dict[great_nn.init_state] = hidden_state
            out, hidden_state = sess.run([great_nn.output, great_nn.output_state],
                    feed_dict=feed_dict)
            previous_char = allowed_chars[np.argmax(out[0][0])]
            print(previous_char)
            done = previous_char == len(allowed_chars)
        



if __name__ == "__main__":
    #  train()
    freerun()
