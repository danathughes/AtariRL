import numpy as np
import os

import cPickle as pickle

import scipy.ndimage as ndimage
from memory import ReplayMemory

class PriorityReplayMemory:
	"""
	"""

	def __init__(self, memory_size=1000000, base_memory, num_Q_functions):
		"""
		Create a recorder to record the dataset
		"""

		self.memory_size = memory_size

		# The current index of the buffer.  Assume a circular buffer
		self._idx = 0
		self.filled = False		# Has the buffer been filled?

		# Build an actual memory buffer
		self.memory = base_memory

		# Mask for which Q function(s) should learn from this 
		self.masks = np.zeros((memory_size,num_Q_functions))


	def record(self, frame, action, reward, is_terminal):
		"""
		Store this state, action and reward.  Flush and start a new batch if necessary
		"""

		self.memory.record(frame, action, reward, is_terminal)


	def update(self, indices, TD_error):
		"""
		Update the priorities at the indices with the new TD error
		"""

		self.memory.update(indices, TD_error)


	def get_samples(self, size, history_length=4):
		"""
		Return an array of samples
		"""

		return self.memory.get_samples(size, history_length)


	def save(self, path):
		"""
		"""

		self.memory.save(path)


	def load(self, path):
		"""
		"""

		self.memory.load(path)