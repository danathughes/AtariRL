##
##
##

# Import libraries to simulate Atari and display results
from ale_python_interface import ALEInterface
import pygame
from pygame.locals import *

import numpy as np


class HumanController:
	"""
	"""

	def __init__(self):
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
				action = 4
		elif keys[K_RIGHT]:
				action = 3 
		elif keys[K_SPACE]:
			action = 1
		else:
			action = 0

		return action


class AtariGameInterface:
	"""
	"""

	def __init__(self, game_filename, screen_dir = './screens/'):
		"""
		Load the game and create a display using pygame
		"""

		# Create the pygame screen
		pygame.init()
		self.screen = pygame.display.set_mode((160,210))

		self.screen_dir = screen_dir

		# Buffers for grabbing the screen from ALE and displaying via pygame
		self.screen_buffer = np.zeros((110800,), np.uint8)

		# Create the ALE interface and load the game
		self.ale = ALEInterface()
		self.ale.setBool('color_averaging', True)
		self.ale.loadROM(game_filename)

		# Grab the set of available moves
		self.move_list = self.ale.getMinimalActionSet()

		self.update_screen()
		
		# Create a human controller to control input to the game
		self.controller = HumanController()


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

		actions = []
		rewards = []
		states = []

		self.ale.reset_game()

#		fpsClock = pygame.time.Clock()

		while not self.ale.game_over():
			self.update_screen()

			state = self.get_reduced_screen()
			action = self.controller.act(state)
			reward = self.ale.act(action)

#			fpsClock.tick(30)

			states.append(state)
			actions.append(action)
			rewards.append(reward)

		return states, actions, rewards


agi = AtariGameInterface('Breakout.bin')