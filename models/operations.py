## operations.py		Dana Hughes				21-Sept-2017
##
## Common operations used for DQN networks


import tensorflow as tf

class Update(object):
  """
  Perform an update step between a DQN and a target DQN
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
