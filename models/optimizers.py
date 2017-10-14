## optimizers.py				Dana Hughes								21-Sept-2017
##
## Optimizers for DQN

import tensorflow as tf

class ClippedRMSPropOptimizer(object):
  """
  Optimizer for the DQN.  Somewhat pilfered from the devsisters implementation...
  """

  def __init__(self, dqn, **kwargs):
    """
    """

    self._name = kwargs.get('name', 'optimizer')

    # Discount factor, learning rate, momentum, etc.
    self.learning_rate = kwargs.get('learning_rate', 0.00025)
    self.momentum = kwargs.get('momentum', 0.95)
    self.epsilon = kwargs.get('epsilon', 1e-6)
    self.decay = kwargs.get('decay', 0.99)

    # Alternative has RMS Params as: Learning Rate = 0.00025, Decay = 0.99, Momentum = 0.0, Epsilon=1e-6

    with tf.variable_scope(self._name):
      # Input to the optimizer is the DQN output, the action performed and the Q value of the target DQN
      self.action = tf.placeholder(tf.uint8, [None], name='action')
      self.target_Q = tf.placeholder(tf.float32, [None], name='target_Q_value')
      self.weights = tf.placeholder(tf.float32, [None], name='weights')

      action_one_hot = tf.one_hot(self.action, dqn.num_actions, 1.0, 0.0)
      action_Q = tf.reduce_sum(dqn.Q*action_one_hot, reduction_indices=1, name='action_q')

      # Create the loss function (squared difference)
      delta = self.target_Q - action_Q
      squared_loss = 0.5*tf.square(delta)
      huber_loss = tf.where(tf.abs(delta) < 1.0, squared_loss, tf.abs(delta) - 0.5)

#      huber_loss = tf.losses.huber_loss(self.target_Q, action_Q)
      weighted_loss = self.weights * huber_loss

      self.loss = tf.reduce_mean(weighted_loss, name='loss')

      # Create the optimization operation
      self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum, epsilon=self.epsilon)
      self.train_step = self.optimizer.minimize(self.loss)

      # Need to clip gradients between -1 and 1 to stabilize learning
#      grads_and_vars = self.optimizer.compute_gradients(self.loss)
#      capped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) if grad is not None else (None, var) for grad, var in grads_and_vars]
#      self.train_step = self.optimizer.apply_gradients(capped_grads_and_vars)