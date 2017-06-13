##
##
##

# Import libraries to simulate Atari and display results
from ale_python_interface import ALEInterface
import pygame
from pygame.locals import *

import numpy as np
import os


class RandomController:
	"""
	"""

	def __init__(self, num_actions):
		"""
		"""

		self.num_actions = num_actions


	def act(self, state):
		"""
		"""

		return np.random.randint(self.num_actions)


class HumanController:
	"""
	"""

	def __init__(self, num_actions):
		"""
		"""

		pass

	def act(self, state):
		"""
		Use pygame to get the action from the user
		"""

		action = 0

		pygame.event.get()

		keys = pygame.key.get_pressed()

		if keys[K_LEFT]:
				action = 3
		elif keys[K_RIGHT]:
				action = 2 
		elif keys[K_SPACE]:
			action = 1
		else:
			action = 0

		return action


def calculate_Q(path, discount):
	"""
	Load all rewards and calculate the value for Q
	"""

	filenames = os.listdir(path)

	# Get the start and end indices of each file
	# NOTE:  We assume that the filename is like: *****_<begin>_<end>.npy

	indices = []
	for name in filenames:
		idx = name[:-4].split('_')[-2:]
		idx = (name,) + tuple([int(x) for x in idx])
		indices.append(idx)

	# Figure out how big to make the numpy array, and make an array for the Q values
	max_idx = max([x[2] for x in indices])

	rewards = np.zeros((max_idx,))
	Qs = np.zeros((max_idx,))

	# Load the rewards into the rewards array
	for start, end, filename in indices:
		reward_file = open(path + filename)
		rewards[start:end] = np.load(reward_file)

	# Calculate the Q values


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


class ReplayMemory:
	"""
	"""

	def __init__(self, memory_size=100000):
		"""
		Create a recorder to record the dataset
		"""

		# Buffers to store the data
		self.states = np.zeros((memory_size, 84, 84))
		self.actions = np.zeros((memory_size,))
		self.rewards = np.zeros((memory_size,))
		self.terminal = np.zeros((memory_size,), np.bool)

		self.memory_size = memory_size

		# The current index of the buffer.  Assume a circular buffer
		self._idx = 0


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
			

class AtariGameInterface:
	"""
	"""

	def __init__(self, game_filename, controller, replay_memory):
		"""
		Load the game and create a display using pygame
		"""

		# Create the pygame screen
		pygame.init()
		self.screen = pygame.display.set_mode((160,210))

		# Buffers for grabbing the screen from ALE and displaying via pygame
		self.screen_buffer = np.zeros((110800,), np.uint8)

		# Create the ALE interface and load the game
		self.ale = ALEInterface()
		self.ale.setBool('color_averaging', True)
		self.ale.loadROM(game_filename)

		# Grab the set of available moves
		self.move_list = self.ale.getMinimalActionSet()

		# Show the first screen
		self.update_screen()
		
		# Hang on to the provided controller and replay memory
		self.controller = controller
		self.replay_memory = replay_memory


	def update_screen(self):
		"""
		Grab the current screen from ALE and display it via pygame
		"""

		self.ale.getScreenRGB(self.screen_buffer)

		screen_pix = pygame.PixelArray(self.screen)
		# Display on the pygame screen
		for i in range(210):
			for j in range(160):
				screen_pix[j][i] = (self.screen_buffer[(160*3)*i + 3*j], self.screen_buffer[(160*3)*i + 3*j + 1], self.screen_buffer[(160*3)*i + 3*j + 2])

		del screen_pix

		pygame.display.update()


	def get_reduced_screen(self):
		"""
		Convert current screen to 84x84 np array of luminescence values.  Scale values
		from 0.0 to 1.0 to work with Tensorflow
		"""

		gray_screen = np.zeros((84,84))

		small_screen = pygame.transform.scale(self.screen, (84,84))

		for x in range(84):
			for y in range(84):
				r,g,b,a = small_screen.get_at((x,y))
				gray_screen[y,x] = (0.3*r + 0.59*g + 0.11*b) / 255.0

		return gray_screen


	def play(self):
		"""
		Allow for user to play the game
		"""

		# Reset the game to start a new episode
		self.ale.reset_game()

		while not self.ale.game_over():
			self.update_screen()

			state = self.get_reduced_screen()
			action_num = self.controller.act(state)
			action = self.move_list[action_num]
			reward = self.ale.act(action)

			self.replay_memory.record(state, action_num, reward, not self.ale.game_over())


#controller = HumanController(4)
controller = RandomController(4)
replay_memory = ReplayMemory()
agi = AtariGameInterface('Breakout.bin', controller, replay_memory)