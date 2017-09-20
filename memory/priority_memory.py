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

		# Crate a list of valid indices, then sample with the given probability distribution
		valid_indices = np.arange(history_length - 1, max_idx - 1) + self._idx
		valid_indices = valid_indices % max_idx

#		probabilities = self.priority[valid_indices]
#		probabilities = probabilities ** self.alpha
		probabilities = self.priority ** self.alpha
		probabilities /= np.sum(probabilities)

		probs = probabilities[valid_indices] / np.sum(probabilities[valid_indices])

		indices = np.random.choice(valid_indices, size, False, probs)

		# Populate the experiences as normally
		experiences = self.memory.pull_experiences(indices, history_length, max_idx)

		# Calculate the weights of each experience
		weights = (size*probabilities[indices]) ** (-self.beta)

		return experiences, indices, weights


	def save(self, path):
		"""
		"""

		self.memory.save(path)


	def load(self, path):
		"""
		"""

		self.memory.load(path)




class RankedPriorityReplayMemory:
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

		# Parameters - Just a placeholder until this works
		self.alpha = alpha
		self.beta = beta

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


	def update(self, indices, TD_error):
		"""
		Update the priorities at the indices with the new TD error
		"""

		self.priority[indices] = np.abs(TD_error)


	def get_samples(self, size, history_length=4):
		"""
		Return an array of samples
		"""

		# Figure out valid indices
		if self.filled:
			max_idx = self.memory_size
		else:
			max_idx = self._idx

		# Create a list of valid indices, then sample with the given probability distribution
		valid_indices = np.arange(history_length - 1, max_idx - 1) + self._idx
		valid_indices = valid_indices % max_idx

		# Determine the rank of each index, and the prioritization of each item 
		index_rank = np.lexsort([self.priority])
		index_rank = np.flip(index_rank, 0)
		rank_priority = 1.0/(index_rank + 1)

		# Calculate the probability of each item in the valid indices
		probabilities = rank_priority ** self.alpha
		probabilities = probabilities / np.sum(probabilities)

		probs = probabilities[valid_indices] / np.sum(probabilities[valid_indices])

		indices = np.random.choice(valid_indices, size, False, probs)

		# Populate the experiences as normally
		experiences = self.memory.pull_experiences(indices, history_length, max_idx)

		# Calculate the weights of each experience
		weights = (size*probabilities[indices]) ** (-self.beta)

		return experiences, indices, weights


	def save(self, path):
		"""
		"""

		self.memory.save(path)


	def load(self, path):
		"""
		"""

		self.memory.load(path)