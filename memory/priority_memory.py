import numpy as np
import os

import cPickle as pickle

import scipy.ndimage as ndimage
from memory import ReplayMemory

class PriorityReplayMemory:
	"""
	"""

	def __init__(self, memory_size=1000000, height=84, width=84, alpha = 0.6, beta = 0.4):
		"""
		Create a recorder to record the dataset
		"""

		self.width = width
		self.height = height

		self.memory_size = memory_size

		# The current index of the buffer.  Assume a circular buffer
		self._idx = 0
		self.filled = False		# Has the buffer been filled?

		# Build an actual memory buffer
		self.memory = ReplayMemory(memory_size, height, width)

		# Build lists of priority, probability and weights
		self.priority = np.zeros((memory_size,))
#		self.probability = np.zeros((memory_size,))
#		self.weights = np.zeros((memory_size,))

		# Parameters - Just a placeholder until this works
		self.alpha = alpha
		self.beta = beta

		self.epsilon = 1e-6

		# Set the initial priority
		self.priority[0] = 1.0


	def record(self, frame, action, reward, is_terminal):
		"""
		Store this state, action and reward.  Flush and start a new batch if necessary
		"""

		# Set the priority, then record the memory, to ensure the indices are the same
		self.priority[self._idx] = np.max(self.priority)

		self.memory.record(frame, action, reward, is_terminal)

		# Update the index to match the memory's index
		self._idx = self.memory._idx
		self.filled = self.memory.filled
		self.memory_size = self.memory.memory_size


	def update(self, indices, TD_error):
		"""
		Update the priorities at the indices with the new TD error
		"""

		self.priority[indices] = self.epsilon + np.abs(TD_error)


	def get_samples(self, size, history_length=4):
		"""
		Return an array of samples
		"""

		# Figure out valid indices
		if self.filled:
			max_idx = self.memory_size
		else:
			max_idx = self._idx

		# Determine the probability of each sample
#		self.probability = self.priority ** self.alpha
#		self.probability /= np.sum(self.probability)

		# Crate a list of valid indices, then sample with the given probability distribution
		valid_indices = np.arange(3, max_idx - 1) + self._idx
		valid_indices = valid_indices % max_idx

		probs = self.priority[valid_indices]
		probs = probs ** self.alpha
		probs = probs / np.sum(probs)


		indices = np.random.choice(valid_indices, 32, False, probs)

		# Populate the experiences as normally
		experiences = self.memory.pull_experiences(indices, history_length, max_idx)

		# Calculate the weights of each experience
		weights = (size*probs) ** (-self.beta)

		return experiences, indices, weights


	def save(self, path):
		"""
		"""

		self.memory.save(path)


	def load(self, path):
		"""
		"""

		self.memory.load(path)