## CNN1D.py
##
## A simple 1D Convolution Neural Network classifier model.

import tensorflow as tf
import numpy as np
from parts import *


DEEPMIND_LAYERS = ([Convolutional((8,8), 32, name='conv1', stride=4),
                   ReLU(name='relu1'),
                   Convolutional((4,4), 64, name='conv2', stride=2),
                   ReLU(name='relu2'),
                   Convolutional((3,3), 64, name='conv3', stride=1),
                   ReLU(name='relu3'),
                   Flatten(name='flatten'),
                   FullConnection(512, name='full1'),
                   ReLU(name='relu4'),
                   FullConnection(4, name='full2'),
                   Linear(name='linear')
            ],
                  [Convolutional((8,8), 32, name='conv1_tgt', stride=4),
                   ReLU(name='relu1_tgt'),
                   Convolutional((4,4), 64, name='conv2_tgt', stride=2),
                   ReLU(name='relu2_tgt'),
                   Convolutional((3,3), 64, name='conv3_tgt', stride=1),
                   ReLU(name='relu3_tgt'),
                   Flatten(name='flatten_tgt'),
                   FullConnection(512, name='full1_tgt'),
                   ReLU(name='relu4_tgt'),
                   FullConnection(4, name='full2_tgt'),
                   Linear(name='linear_tgt')
            ])            

class DeepQNetwork(object):

   def __init__(self, frame_shape, hidden_layers, num_actions, **kwargs):
      """
      Build a deep convolutional neural network network

      frame_shape    - shape of the input frame: (width x height x num_channels)
      hidden_layers  - A list of hidden layers, which take the form of a tuple, which depends
                       on the type (first element of the tuple)
      num_targets    - the number of actions which can be performed
      """

      # Input and target placeholders
      self.frame_shape = tuple(frame_shape)
      self.num_actions = num_actions

      self.input = tf.placeholder(tf.float32, shape=(None,) + tuple(frame_shape))
      self.target = tf.placeholder(tf.float32, shape=(None, num_actions))

      self._weight_decay = kwargs.get('weight_decay', 0.0)

      # Build up the hidden layers for the network
      # Start by reshaping the input to work with the 2D convolutional tensors
      current_layer = self.input

      for layer in hidden_layers:
         current_layer = layer.build(current_layer)

      self.output = current_layer

      # Set the objective to the L2-norm of the residual
      self._objective = tf.reduce_sum(tf.square(self.output - self.target))

      # Enable saving and loading of variables
      self.saver = tf.train.Saver()
      

   def get_Qs(self, states, sess):
      """
      Do a forward pass with the provided states
      """

      single_state = False

      if len(states.shape) == 3:
        _input = np.reshape(states, (1,) + self.frame_shape)
        single_state = True
      else:
        _input = states

      Q = sess.run(self.output, feed_dict={self.input: _input})

      if single_state:
        Q = np.reshape(Q, (self.num_actions,))

      return Q


   def objective(self):
      """
      Return the objective tensor of this network
      """

      return self._objective


   def get_feed_dict(self, data):
      """
      Create a feed dictionary for this model
      """

      return {self.input: data['input'], self.target: data['target']}


   def train(self, train_step, data):
      """
      Train on the input data (x) and the target (y).  The train step is some optimizer
      """

      train_step.run(feed_dict=self.get_feed_dict(data))


   def save(self, path, sess):
      """
      Save the variables
      """

      # Not working...
      self.saver.save(sess, path)


   def restore(self, path, sess):
      """
      """

      self.saver.restore(sess, path)