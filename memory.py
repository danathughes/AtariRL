import numpy as np
import os

import cPickle as pickle

import scipy.ndimage as ndimage

class ReplayMemory:
	"""
	"""

	def __init__(self, memory_size=1000000, height=84, width=84):
		"""
		Create a recorder to record the dataset
		"""

		self.width = width
		self.height = height

		# Buffers to store the data
		self.frames = np.ones((memory_size, height, width), np.uint8)
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

		self.frames[self._idx,:,:] = frame
		self.actions[self._idx] = action
		self.rewards[self._idx] = reward
		self.terminal[self._idx] = is_terminal

		self._idx += 1

		# Reset the circular buffer 
		if self._idx == self.memory_size:
			self._idx = 0
			self.filled = True


	def get_samples(self, size, history_length=4):
		"""
		Return an array of samples
		"""

		state = np.zeros((size, self.height, self.width, 4), np.float32)
		next_state = np.zeros((size, self.height, self.width, 4), np.float32)


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
			n = (indices - 3 + i) % max_idx
			state[:,:,:,i] = self.frames[n,:,:].astype(np.float32)# / 255.0
		next_state[:,:,:,0:3] = state[:,:,:,1:4]
		n = (indices + 1) % max_idx
		next_state[:,:,:,3] = self.frames[n,:,:].astype(np.float32)# / 255.0

		return state, self.actions[indices], self.rewards[indices], next_state, self.terminal[n]


	def save(self, path):
		"""
		"""

		# Make a directory

		if not os.path.exists(path):
			os.makedirs(path)

		filenames = ['frames.npy', 'actions.npy', 'rewards.npy', 'terminal.npy']
		values = [self.frames, self.actions, self.rewards, self.terminal]

		for filename, values in zip(filenames, values):
			f = open(path + '/' + filename, 'wb')
			np.save(f, values)
			f.close()

		f=open(path + '/params.txt', 'w')
		f.write("%d,%d,%d"%(self._idx, self.memory_size, self.filled))
		f.close()


	def load(self, path):
		"""
		"""

		print "Restoring Experience Replay Memory..."

		self.frames = None
		self.actions = None
		self.rewards = None
		self.terminal = None

		f = open(path + '/frames.npy')
		self.frames = np.load(f)
		f.close()

		f = open(path + '/actions.npy')
		self.actions = np.load(f)
		f.close()

		f = open(path + '/rewards.npy')
		self.rewards = np.load(f)
		f.close()

		f = open(path + '/terminal.npy')
		self.terminal = np.load(f)
		f.close()

		f=open(path + '/params.txt')
		info = f.readline()
		f.close()

		values = [int(x) for x in info.split(',')]

		self._idx = values[0]
		self.memory_size = values[1]
		self.filled = bool(values[2])

		print "  Replay Memory Current Index:", self._idx
		print "  Replay Memory Size:", self.memory_size
		print "  Replay Memory is Filled:", self.filled

