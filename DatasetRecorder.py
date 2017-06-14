class DatasetRecorder:
	"""
	"""

	def __init__(self, memory_size=100000, storage_path="./dataset/", file_prefix=""):
		"""
		Create a recorder to record the dataset
		"""

		# Buffers to store the data
		self.states = np.zeros((memory_size, 84, 84, 4))
		self.actions = np.zeros((memory_size,))
		self.rewards = np.zeros((memory_size,))

		self.memory_size = memory_size

		# Current frame and batch
		self.frame_number = 0
		self.batch_number = 0

		# Where to store this stuff
		self.path = storage_path
		self.prefix = file_prefix


	def record(self, frame, action, reward):
		"""
		Store this state, action and reward.  Flush and start a new batch if necessary
		"""

		self.states[self.frame_number,:,:,0] = frame
		if self.frame_number > 0:
			self.states[self.frame_number-1,:,:,1] = frame
		if self.frame_number > 1:
			self.states[self.frame_number-2,:,:,2] = frame
		if self.frame_number > 2:
			self.states[self.frame_number-3,:,:,3] = frame

		self.actions[self.frame_number] = action
		self.rewards[self.frame_number] = reward

		self.frame_number += 1

		# Has the current buffer been filled?
		if self.frame_number == self.memory_size:
			# Flush out the current buffer to a file and restart, writing the last 3 frames to the appropriate spots
			self.flush()
			self.frame_number = 0
			self.batch_number += 1

			self.states[0,:,:,1:3] = self.states[-1,:,:,0:2]
			self.states[1,:,:,2:3] = self.states[-1,:,:,0:1]
			self.states[2,:,:,3] = self.states[-1,:,:,0]


	def reset(self, storage_prefix="", file_prefix=""):
		"""
		Reset the buffer and prefix
		"""

		self.frame_number = 0
		self.batch_number = 0
		self.path = storage_prefix
		self.prefix = file_prefix


	def flush(self):
		"""
		Record the current buffer into memory
		"""

		start_frame = self.batch_number*self.memory_size

		filename = "%s%sstates_%d_%d.npy" % (self.path, self.prefix, start_frame, start_frame + self.frame_number)
		outfile = open(filename, 'wb')
		np.save(outfile, self.states[:self.frame_number,:,:,:])
		outfile.close()

		filename = "%s%sactions_%d_%d.npy" % (self.path, self.prefix, start_frame, start_frame + self.frame_number)
		outfile = open(filename, 'wb')
		np.save(outfile, self.actions[:self.frame_number])
		outfile.close()

		filename = "%s%srewards_%d_%d.npy" % (self.path, self.prefix, start_frame, start_frame + self.frame_number)
		outfile = open(filename, 'wb')
		np.save(outfile, self.rewards[:self.frame_number])
		outfile.close()