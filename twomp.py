# -*- coding: utf-8 -*-

import random
import json
import tensorflow as tf


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
    def __init__(self, batch_size):
        self.input = tf.placeholder(tf.int32, [batch_size, 140])
        self.lengths = tf.placeholder(tf.int32, [batch_size])
        cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        self.init_state = cell.zero_state(batch_size, tf.float32)


if __name__ == "__main__":
    t = Trumpinator(32)
    print(dataset[1:10])
