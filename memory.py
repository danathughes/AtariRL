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
		self.states = np.zeros((memory_size, 84, 84), np.uint8)
		self.actions = np.zeros((memory_size,), np.uint8)
		self.rewards = np.zeros((memory_size,))
		self.terminal = np.zeros((memory_size,), np.bool)

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
			idx = np.random.randint(self.memory_size)
			# Avoid indices where the next state is unavailable, or 3 prior states are not
			while idx >= self._idx and idx < self._idx + 4:
				idx = np.random.randint(self.memory_size)
		else:
			# NOTE:  This needs to be fixed, because it can make the next stat have all zeros for the current frame, etc.
			#        And also allows for the current state's previous frames to be zero...
			idx = np.random.randint(self.memory_size)

		# Get the current and next state
		for i in range(4):
			state[:,:,i] = self.states[idx-i,:,:].astype(np.float32) / 255.0
		next_state[:,:,1:4] = state[:,:,0:3]
		next_state[:,:,0] = self.states[(idx+1)%self.memory_size,:,:].astype(np.float32) / 255.0

		return state, self.actions[idx], self.rewards[idx], next_state, self.terminal[(idx+1)%self.memory_size]
			