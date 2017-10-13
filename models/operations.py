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


class TensorflowCheckpoint:
  """
  Class to save and restore a tensorflow graph
  """

  def __init__(self, save_path, counter, sess):
    """
    """

    self.path = save_path
    self.saver = tf.train.Saver()
    self.counter = counter
    self.sess = sess


  def save(self):
    """
    Save the current parameters in the graph
    """

    self.saver.save(self.sess, self.path + '/tensorflow-model', global_step=self.counter.count)


  def restore(self, checkpoint_frame=None):
    """
    Restores a checkpoint.

    checkpoint_frame - Frame number of desired checkpoint to be loaded.  If None is provided,
                       the most recent checkpoint 
    """

    if checkpoint_frame:
      self.saver.restore(self.sess, self.path + '/tensorflow-model-%d' % checkpoint_frame)
    else:
      self.saver.restore(self.sess, self.saver.latest_checkpoint())