import numpy as np
import os

import cPickle as pickle

import scipy.ndimage as ndimage
from memory import ReplayMemory

class PriorityReplayMemory(ReplayMemory):
	"""
	An extension of Experience Replay Memory which introduces a priority for each frame.
	"""

	def __init__(self, memory_size=1000000, frame_size=(84,84), alpha = 0.6, beta = 0.4, epsilon = 1e-6):
		"""
		Create a recorder to record the dataset
		"""

		# Initialize the base memory
		ReplayMemory.__init__(self, memory_size, frame_size)

		# Build lists of priority, probability and weights
		self.priority = np.zeros((memory_size,))

		# Parameters - Just a placeholder until this works
		self.alpha = alpha
		self.beta = beta

		self.epsilon = epsilon

		# Set the initial priority
		self.priority[0] = 1.0


	def record(self, frame, action, reward, is_terminal):
		"""
		Store this state, action and reward.  Flush and start a new batch if necessary
		"""

		# Set the priority, then record the memory, to ensure the indices are the same
		self.priority[self._idx] = np.max(self.priority)

		ReplayMemory.record(self, frame, action, reward, is_terminal)


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

		probabilities = self.priority ** self.alpha
		probabilities /= np.sum(probabilities)

		probs = probabilities[valid_indices] / np.sum(probabilities[valid_indices])

		indices = np.random.choice(valid_indices, size, False, probs)

		# Populate the experiences as normally
		experiences = self.pull_experiences(indices, history_length, max_idx)

		# Calculate the weights of each experience
		weights = (size*probabilities[indices]) ** (-self.beta)

		return experiences, indices, weights


	def save(self, path):
		"""
		"""

		ReplayMemory.save(self, path)

		f = open(path + '/priority.npy', 'wb')
		np.save(f, self.priority)
		f.close()


	def load(self, path):
		"""
		"""

		ReplayMemory.load(self, path)

		self.priority = None

		f = open(path + '/priority.npy')
		self.priority = np.load(f)
		f.close()


class RankedPriorityReplayMemory(PriorityReplayMemory):
	"""
	An variation of Priority Replay Memory where priority is rank-based (i.e., a power-law distribution)
	"""

	def __init__(self, memory_size=1000000, frame_size = (84,84), alpha = 0.6, beta = 0.4):
		"""
		Create a recorder to record the dataset
		"""

		PriorityReplayMemory.__init__(self, memory_size, frame_size, alpha, beta, epsilon=0.0)


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
		experiences = self.pull_experiences(indices, history_length, max_idx)

		# Calculate the weights of each experience
		weights = (size*probabilities[indices]) ** (-self.beta)

		return experiences, indices, weights
