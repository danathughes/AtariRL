## DeepQNetwork.py
##
## A Deep Q Network, as implemented in the Nature paper.
##

import tensorflow as tf
import numpy as np
from parts import *
import os

class UpdateOperation(object):
  """
  Operations to perform an update step between a DQN and a target DQN
  """

  def __init__(self, dqn, target_dqn, sess):
    """
    """

    self.dqn = dqn
    self.target_dqn = target_dqn
    self.sess = sess

    # Create an operator to update the target weights from the current DQN
    self.update_operations = []

    with tf.variable_scope('update_operation'):
      for name in self.dqn.params:
        op = self.target_dqn.params[name].assign(self.dqn.params[name].value())
        self.update_operations.append(op)


  def run(self):
    """
    Perform the update operations
    """

    self.sess.run(self.update_operations)

    
class Optimizer(object):
  """
  Optimizer for the DQN.  Somewhat pilfered from the devsisters implementation...
  """

  def __init__(self, dqn_output, num_actions, sess, **kwargs):
    """
    """

    self._name = kwargs.get('name', 'optimizer')

    # Discount factor, learning rate, momentum, etc.
    self.learning_rate = kwargs.get('learning_rate', 0.00025)
    self.momentum = kwargs.get('momentum', 0.0)
    self.epsilon = kwargs.get('epsilon', 1e-6)
    self.decay = kwargs.get('decay', 0.99)

    # Alternative has RMS Params as: Learning Rate = 0.00025, Decay = 0.99, Momentum = 0.0, Epsilon=1e-6

    with tf.variable_scope(self._name):
      # Input to the optimizer is the DQN output, the action performed and the Q value of the target DQN
      self.q_values = dqn_output

      self.action = tf.placeholder(tf.uint8, [None], name='action')
      self.target_q = tf.placeholder(tf.float32, [None], name='target_Q_value')


      action_one_hot = tf.one_hot(self.action, num_actions, 1.0, 0.0)
      action_q = tf.reduce_sum(self.q_values*action_one_hot, reduction_indices=1, name='action_q')

      delta = self.target_q - action_q
      self.delta = delta

      # Create the loss function (squared difference)
      self.squared_loss = 0.5*tf.square(delta)

      self.huber_loss = tf.where(tf.abs(delta) < 1.0, self.squared_loss, tf.abs(delta) - 0.5)
      self.loss = tf.reduce_mean(self.huber_loss, name='loss')

      # Create the optimization operation
      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay, momentum=self.momentum, epsilon=self.epsilon)
#      self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
      # Need to clip gradients between -1 and 1 to stabilize learning
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      capped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) if grad is not None else (None, var) for grad, var in grads_and_vars]
      self.train_step = self.optimizer.apply_gradients(capped_grads_and_vars)

      # Make a training step to minimize the loss function
#      self.train_step = self.optimizer.minimize(self.loss)


class DeepQNetwork(object):

   def __init__(self, frame_shape, hidden_layers, num_actions, sess, **kwargs):
      """
      Build a deep convolutional neural network network

      frame_shape    - shape of the input frame: (width x height x num_channels)
      hidden_layers  - A list of hidden layers, which take the form of a tuple, which depends
                       on the type (first element of the tuple)
      num_targets    - the number of actions which can be performed
      """

      # List of weights and biases
      self.params = {}

      self._network_name = kwargs.get('network_name', 'DQN')

      # Input and target placeholders
      self._trainable = kwargs.get('trainable', True)

      self.sess = sess

      self.frame_shape = tuple(frame_shape)
      self.num_actions = num_actions

      with tf.variable_scope(self._network_name):
        self.input = tf.placeholder(tf.float32, shape=(None,) + tuple(frame_shape), name='input')

        # Build up the hidden layers for the network
        # Start by reshaping the input to work with the 2D convolutional tensors
        current_layer = self.input

        for layer in hidden_layers:
          current_layer, w, b = layer.build(current_layer, trainable=self._trainable)
          if w:
            self.params['w_'+layer.name] = w
          if b:
            self.params['b_'+layer.name] = b


        self.output = current_layer
        self.q = self.output

      # Set the objective to the L2-norm of the residual
      if self._trainable:
        self.optimizer = Optimizer(self.output, self.num_actions, self.sess)
      else:
        self._objective = None
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

      Q = self.sess.run(self.output, feed_dict={self.input: _input})

      if single_state:
        Q = np.reshape(Q, (self.num_actions,))

      return Q


   def train(self, data):
      """
      Train on the input data (x) and the target (y).
      """

      loss = 0.0
      fd = {self.input: data['input'], self.optimizer.target_q: data['target'], self.optimizer.action: data['action']}

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

#      for name, param in self.params.items():
#         values = np.load(directory + '/' + name + '.npy')

#         self.sess.run(param.assign(values))

      print directory

      self.saver.restore(self.sess, directory + '/dqn_model-4000000' )



