import random
import json
import tensorflow as tf
import tensorflow.contrib.slim as slim


def transform_text(text):
    """Apply all transformations you want here ... :)))))))"""
    return text

dataset = [
    transform_text(t['text']) for i in (2015, 2016, 2017)
                              for t in json.load(open('data/%d.json' % i))
]


def get_batch(batch_size):
    return random.sample(dataset, batch_size)


class Trumpinator():
    def __init__(self, batch_size, seqlen, nchars):
        self.input = tf.placeholder(tf.int32, [batch_size, seqlen])
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

        self.target = tf.placeholder(tf.int32, [batch_size, seqlen])
        embedded_targets = tf.one_hot(self.target, nchars)
        self.loss = tf.reduce_sum(tf.square(embedded_targets-embedded_inputs))
        self.train_step = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)



if __name__ == "__main__":
    t = Trumpinator(32, 20, 61)
