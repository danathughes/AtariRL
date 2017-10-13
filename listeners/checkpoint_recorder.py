###
###
###

import os

from models.operations import TensorflowCheckpoint

class CheckpointRecorder(object):
	"""
	"""

	def __init__(self, dqn, replay_memory, counter, directory, sess, **kwargs):
		"""
		"""

		self.dqn = dqn
		self.replay_memory = replay_memory
		self.checkpoint_dir = directory

		self.counter = counter

		self.sess = sess

		# Check to see if the directory exists.  If not, create a new one
		if not os.path.exists(self.checkpoint_dir + '/replay_memory'):
			os.makedirs(self.checkpoint_dir + '/replay_memory')
		if not os.path.exists(self.checkpoint_dir + '/dqn'):
			os.makedirs(self.checkpoint_dir + '/dqn')
		if not os.path.exists(self.checkpoint_dir + '/tensorflow'):
			os.makedirs(self.checkpoint_dir + '/tensorflow')

		self.tensorflow_checkpoint = TensorflowCheckpoint(self.checkpoint_dir + '/tensorflow', self.counter, self.sess)


	def save_memory(self):
		"""
		"""

		print "Saving Replay Memory..."
		self.replay_memory.save(self.checkpoint_dir + '/replay_memory/' + str(self.counter.count))


	def restore_memory(self, frame_number):
		"""
		"""

		print "Restoring Replay Memory..."
		self.replay_memory.load(self.checkpoint_dir + '/replay_memory/%d' % frame_number)


	def save_dqn(self):
		"""
		"""

		print "Saving DQN Model Parameters..."
		self.dqn.save(self.checkpoint_dir + '/dqn/' + str(self.counter.count), self.counter.count)


	def restore_dqn(self, frame_number):
		"""
		"""

		print "Restoring DQN Model Parameters..."
		self.dqn.restore(self.checkpoint_dir + '/dqn/' + str(frame_number))


	def save_tensorflow(self):
		"""
		"""

		print "Saving Tensorflow Graph..."
		self.tensorflow_checkpoint.save()


	def restore_tensorflow(self, frame_number):
		"""
		"""

		print "Restoring Tensorflow Graph..."
		self.tensorflow_checkpoint.restore(frame_number)