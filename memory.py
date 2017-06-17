import numpy as np
import os

import scipy.ndimage as ndimage

class ReplayMemory:
	"""
	"""

	def __init__(self, memory_size=1000000):
		"""
		Create a recorder to record the dataset
		"""

		# Buffers to store the data
		self.states = np.ones((memory_size, 84, 84), np.uint8)
		self.actions = np.ones((memory_size,), np.uint8)
		self.rewards = np.ones((memory_size,))
		self.terminal = np.ones((memory_size,), np.bool)

		self.memory_size = memory_size

		# The current index of the buffer.  Assume a circular buffer
		self._idx = 0
		self.filled = False		# Has the buffer been filled?


	def record(self, frame, action, reward, is_terminal):
		"""
		Store this state, action and reward.  Flush and start a new batch if necessary
		"""

		self.states[self._idx,:,:] = frame
		self.actions[self._idx] = action
		self.rewards[self._idx] = reward
		self.terminal[self._idx] = is_terminal

		self._idx += 1

		# Reset the circular buffer 
		if self._idx == self.memory_size:
			self._idx = 0
			self.filled = True


	def get_sample(self, history_length=4):
		"""
		Return a single sample
		"""

		state = np.zeros((84,84,4), np.float32)
		next_state = np.zeros((84,84,4), np.float32)

		# Get an appropriate index 
		if self.filled:
			max_idx = self.memory_size
			base_idx = np.random.randint(max_idx)
			# Avoid indices where the next state is unavailable, or 3 prior states are not
			while base_idx >= self._idx and base_idx < self._idx + 4:
				base_idx = np.random.randint(max_idx)
		else:
			# Can only sample up to the previous frame, so the next state can be generated
			max_idx = self._idx
			# NOTE:  This needs to be fixed, because it can make the next stat have all zeros for the current frame, etc.
			#        And also allows for the current state's previous frames to be zero...
			idx = np.random.randint(max_idx)
			# Avoid the last index, to ensure that the next state is available


		# Get the current and next state
		for i in range(4):
			# Sample the prior 4 frames
			n = (idx - i) % max_idx
			state[:,:,i] = self.states[n,:,:].astype(np.float32) / 255.0
		next_state[:,:,1:4] = state[:,:,0:3]
		n = (idx + 1) % max_idx
		next_state[:,:,0] = self.states[n,:,:].astype(np.float32) / 255.0

		return state, self.actions[idx], self.rewards[idx], next_state, self.terminal[n]



	def get_samples(self, size, history_length=4):
		"""
		Return a single sample
		"""

		state = np.zeros((size,84,84,4), np.float32)
		next_state = np.zeros((size,84,84,4), np.float32)


		# Figure out to how big the current array is
		if self.filled:
			max_idx = self.memory_size
		else:
			max_idx = self._idx

		# Generate indices to use.  Note that the highest index should be one less than the max
		# so that the next state value can be generated, and should be at least history_length,
		# so enough history is available to fill up the state.
		# Also note that the indices should be shifted by the current index, and modulo'd.  This
		# makes sure that the valid range of indices are mapped propertly to the array
		indices = np.random.randint(3, max_idx-1, (size,)) + self._idx
		indices = indices % max_idx

		# Get the current and next state
		for i in range(4):
			# Sample the prior 4 frames
			n = (indices - i) % max_idx
			state[:,:,:,i] = self.states[n,:,:].astype(np.float32) / 255.0
		next_state[:,:,:,1:4] = state[:,:,:,0:3]
		n = (indices + 1) % max_idx
		next_state[:,:,:,0] = self.states[n,:,:].astype(np.float32) / 255.0

		return state, self.actions[indices], self.rewards[indices], next_state, self.terminal[n]
			