###
###
###

import os

class CheckpointRecorder(object):
	"""
	"""

	def __init__(self, dqn, replay_memory, counter, directory, **kwargs):
		"""
		"""

		self.dqn = dqn
		self.replay_memory = replay_memory
		self.checkpoint_dir = directory

		self.counter = counter

		# Check to see if the directory exists.  If not, create a new one
		if not os.path.exists(self.checkpoint_dir + '/replay_memory'):
			os.makedirs(self.checkpoint_dir + '/replay_memory')
		if not os.path.exists(self.checkpoint_dir + '/dqn'):
			os.makedirs(self.checkpoint_dir + '/dqn')

		# How often should the recorder save the state of the dqn and memory?
		self.dqn_record_frequency = kwargs.get('dqn_record_frequency', 50000)
		self.memory_record_frequency = kwargs.get('replay_memory_record_frequency', 1000000)


	def start_episode(self, **kwargs):
		"""
		"""

		pass


	def end_episode(self, **kwargs):
		"""
		"""

		pass


	def step(self, **kwargs):
		"""
		"""

		# Should the network and memory be saved?
		if self.counter.count % self.memory_record_frequency == 0:
			print "Saving Replay Memory..."
			self.replay_memory.save(self.checkpoint_dir + '/replay_memory/' + str(self.counter.count))
		if self.counter.count % self.dqn_record_frequency == 0:
			print "Saving DQN Model..."
			self.dqn.save(self.checkpoint_dir + '/dqn/' + str(self.counter.count))


	def restore(self, frame_number=None):
		"""
		Restore the DQN and replay memory to the provided frame number, or the
		most current one if not provided
		"""

		pass






