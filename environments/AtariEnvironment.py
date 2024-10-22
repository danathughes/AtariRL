from ale_python_interface import ALEInterface

import numpy as np

import scipy.ndimage as ndimage

import pygame
from pygame.locals import *


class AtariEnvironment:
	"""
	Environment for playing Atari games using ALE Interface
	"""

	def __init__(self, game_path, **kwargs):
		"""
		Create an environment with the provided game
		"""

		# Optional parameters
		self.screen_size = kwargs.get('screen_size', (84,84))
#		self.random_seed = kwargs.get('seed', 123)
		self.random_seed = kwargs.get('seed', 0)

		# Buffer for grabbing the screen from ALE
		self.screen_buffer = np.zeros((100800,), np.uint8)

		# Create the ALE interface and load the game
		self.ale = ALEInterface()
		self.ale.setBool('color_averaging', True)
		self.ale.setFloat('repeat_action_probability', 0.0)
		self.ale.setInt('random_seed', self.random_seed)
		self.ale.loadROM(game_path)

		# Grab the set of available moves for this game
		self.move_list = self.ale.getMinimalActionSet()
		self.num_actions = len(self.move_list)
		print "Number of Actions:", self.num_actions

		self.listeners = []

		self.screen = None
#		self.screen = pygame.display.set_mode((160,210))


	def get_state(self):
		"""
		Convert current screen to 84x84 np array of luminescence values. 
		"""

		self.ale.getScreenRGB(self.screen_buffer)

		# Reshape the screen buffer to an appropriate shape
		game_screen = self.screen_buffer.reshape((210,160,3))

		# Convert to luminosity and scale to the desired screen size
		gray_screen = np.dot(game_screen, np.array([0.299, 0.587, 0.114])).astype(np.uint8)
		gray_screen = ndimage.zoom(gray_screen, (0.4, 0.525))

		return gray_screen


	def act(self, action):
		"""
		Perform an action on the environment
		"""

		ale_action = self.move_list[action]

		return self.ale.act(ale_action)


	def terminal(self):
		"""
		Return if the state is a terminal state
		"""

		return self.ale.game_over()


	def lives(self):
		"""
		How many lives are left
		"""

		return self.ale.lives()


	def reset_game(self):
		"""
		"""

		self.ale.reset_game()


	def display(self):
		"""
		"""

		game_screen = self.screen_buffer.reshape((210,160,3))

		game_surf = pygame.surfarray.make_surface(game_screen)
		game_surf = pygame.transform.rotate(game_surf, -90)
		self.screen.blit(game_surf, (0,0))
		pygame.display.flip()

