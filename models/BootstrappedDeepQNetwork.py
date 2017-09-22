## BootstrappedDeepQNetwork.py
##
## Bootstrapped Deep Q Network.
##

import tensorflow as tf
import numpy as np
import os

from models.bootstrapped.optimizer import ClippedRMSPropOptimizer


class BootstrappedDeepQNetwork(object):

   def __init__(self, frame_shape, shared_hidden_layers, head_layers, num_actions, num_heads, sess, **kwargs):
      """
      Build a deep convolutional neural network network

      frame_shape    - shape of the input frame: (width x height x num_channels)
      hidden_layers  - A list of hidden layers, which take the form of a tuple, which depends
                       on the type (first element of the tuple)
      num_targets    - the number of actions which can be performed
      """

      # List of weights and biases
      self.params = {}

      network_name = kwargs.get('network_name', 'DQN')

      # Input and target placeholders
      self._trainable = kwargs.get('trainable', True)

      self.sess = sess

      self.frame_shape = tuple(frame_shape)
      self.num_actions = num_actions
      self.num_heads = num_heads

      with tf.variable_scope(network_name):
        self.input = tf.placeholder(tf.float32, shape=(None,) + tuple(frame_shape), name='input')

        # Build up the hidden layers for the network
        # Start by reshaping the input to work with the 2D convolutional tensors
        current_layer = self.input

        # Build up the shared portion of the network
        for layer in shared_hidden_layers:
          current_layer, w, b = layer.build(current_layer, trainable=self._trainable)
          if w:
            self.params['w_'+layer.name] = w
          if b:
            self.params['b_'+layer.name] = b

        heads = [current_layer] * num_heads
        # Build up each head layer
        for i in range(num_heads):
          with tf.variable_scope(network_name + '_head%d'%i):
            for layer in head_layers:
              current_layer, w, b = layer.build(heads[i], trainable=self._trainable)
              heads[i] = current_layer
              if w:
                self.params['w_'+layer.name+'_head%d' % i] = w
              if b:
                self.params['b_'+layer.name+'_head%d' % i] = b


        # Merge the outputs to a single tensor
        self.Q = tf.stack(heads, axis=1)
        
      # Set the objective to the L2-norm of the residual
      if self._trainable:
        self.optimizer = ClippedRMSPropOptimizer(self)
      else:
        self.optimizer = None

      self.saver = tf.train.Saver()
        

   def get_Qs(self, states):
      """
      Do a forward pass with the provided states
      """

      single_state = False

      if len(states.shape) == 3:
        _input = np.reshape(states, (1,) + self.frame_shape)
        single_state = True
      else:
        _input = states

      Q = self.sess.run(self.Q, feed_dict={self.input: _input})

      if single_state:
        Q = np.reshape(Q, (self.num_heads, self.num_actions))

      return Q


   def train(self, data):
      """
      Train on the input data (x) and the target (y).
      """

      loss = 0.0
      fd = {self.input: data['input'], self.optimizer.target_q: data['target'],
            self.optimizer.action: data['action'], self.optimizer.weights: data['weights'],
            self.optimizer.masks: data['masks']}

      if self._trainable:
         _, loss = self.sess.run([self.optimizer.train_step, self.optimizer.loss], feed_dict=fd)

      return loss


   def save(self, directory, step=0):
      """
      Save the current values of the parameters as numpy arrays
      """

      # Make the save directory if it doesn't exist
      if not os.path.exists(directory):
        os.makedirs(directory)

      # Get the current state of the parameters
      for name, param in self.params.items():
         param_value = self.sess.run(param.value())

         np.save(directory + '/' + name, param_value)

      self.saver.save(self.sess, directory + '/dqn_model', global_step=step)


   def restore(self, directory):
      """
      """

      # Perform some assertions / checks to make sure that the directory exists...

      for name, param in self.params.items():
         values = np.load(directory + '/' + name + '.npy')

         self.sess.run(param.assign(values))

#      print path

#      self.saver.restore(self.sess, path)



