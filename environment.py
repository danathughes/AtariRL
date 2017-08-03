from ale_python_interface import ALEInterface

import pygame
from pygame.locals import *

import numpy as np
import os

import scipy.ndimage as ndimage


class AtariEnvironment:
	"""
	Environment for playing Atari games using ALE Interface
	"""

	def __init__(self, game_filename, **kwargs):
		"""
		Create an environment with the provided game
		"""

		pygame.init()

		self.screen = pygame.display.set_mode((160,210))
		self.fps_clock = pygame.time.Clock()

		self.show_while_training = True

		# Buffer for grabbing the screen from ALE
		self.screen_buffer = np.zeros((100800,), np.uint8)

		# Create the ALE interface and load the game
		self.ale = ALEInterface()
		self.ale.setBool('color_averaging', True)
		self.ale.setFloat('repeat_action_probability', 0.0)
		self.ale.loadROM(game_filename)

		# Grab the set of available moves for this game
		self.move_list = self.ale.getMinimalActionSet()

		self.listeners = []


	def update_screen(self):
		"""
		Grab the current screen from ALE and display it via pygame
		"""

		self.ale.getScreenRGB(self.screen_buffer)

#		if self.show_while_training:
#			game_screen = self.screen_buffer.reshape((210,160,3))
#			game_screen = np.transpose(game_screen, (1,0,2))
#
#			game_surface = pygame.surfarray.make_surface(game_screen)
#			self.screen.blit(game_surface, (0,0))

#			pygame.display.flip()



	def get_reduced_screen(self):
		"""
		Convert current screen to 84x84 np array of luminescence values.  Scale values
		from 0.0 to 1.0 to work with Tensorflow
		"""

		# Reshape the screen buffer to an appropriate shape
		game_screen = self.screen_buffer.reshape((210,160,3))

		# Convert to luminosity
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