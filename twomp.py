import random
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

dataset = list(open('data/cleaned.txt', 'r').readlines())
def get_batch(batch_size):
    return random.sample(dataset, batch_size)


class Trumpinator():
    def __init__(self, batch_size, seqlen, nchars):
        
        self.input = tf.placeholder(tf.int32, [batch_size, 140])
        self.lengths = tf.placeholder(tf.int32, [batch_size])
        embedded_inputs = tf.one_hot(self.input, nchars)
        
        cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)

        self.init_state = cell.zero_state(batch_size, tf.float32)

        output, self.output_state = tf.nn.dynamic_rnn(
                cell,
                embedded_inputs,
                initial_state=self.init_state)

        self.output = tf.nn.softmax(slim.fully_connected(output, nchars))


if __name__ == "__main__":
    t = Trumpinator(32, 20, 61)

