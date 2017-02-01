import random
import sys
import tensorflow as tf

dataset = list(open('data/cleaned.txt', 'r').readlines())
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

