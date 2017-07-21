##
##
##

# Import libraries to simulate Atari and display results
from ale_python_interface import ALEInterface
import pygame
from pygame.locals import *

import numpy as np
import os

import scipy.ndimage as ndimage

from models.DeepQNetwork import *

from controllers import DQNController, EpsilonController
from NetworkParameters import NATURE, ARXIV
from memory import ReplayMemory

#from replay_memory import *


class AtariGameInterface:
	"""
	"""

	def __init__(self, game_filename, controller, replay_memory, **kwargs):
		"""
		Load the game and create a display using pygame
		"""

		# Create the pygame screen
		pygame.init()
		self.screen = pygame.display.set_mode((160,210))

		# Buffers for grabbing the screen from ALE and displaying via pygame
		self.screen_buffer = np.zeros((100800,), np.uint8)

		# Create the ALE interface and load the game
		self.ale = ALEInterface()
		self.ale.setBool('color_averaging', True)
		self.ale.loadROM(game_filename)

		# Grab the set of available moves
		self.move_list = self.ale.getMinimalActionSet()

		self.show_while_training = False

		# Show the first screen
		self.update_screen()
		
		# Hang on to the provided controller and replay memory
		self.controller = controller
		self.replay_memory = replay_memory

		self.evaluate = False

		# Maximum number of no-op that can be performed at the start of an episode
		self.noop_max = kwargs.get('noop_max', 30)
		self.action_repeat = kwargs.get('action_repeat', 4)

		self.frame_number = 0

		

	def update_screen(self):
		"""
		Grab the current screen from ALE and display it via pygame
		"""

		self.ale.getScreenRGB(self.screen_buffer)

		if self.show_while_training:
			game_screen = self.screen_buffer.reshape((210,160,3))
			game_screen = np.transpose(game_screen, (1,0,2))

			game_surface = pygame.surfarray.make_surface(game_screen)
			self.screen.blit(game_surface, (0,0))

			pygame.display.flip()


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


	def learn(self):
		"""
		Allow for controller to learn while playing the game
		"""

		# Reset the game to start a new episode
		self.ale.reset_game()

		num_lives = self.ale.lives()	

		score = 0

		# Wait a random number of frames before starting
		for i in range(np.random.randint(self.noop_max)):
			self.ale.act(0)

		while not self.ale.game_over():
			self.update_screen()

			state = self.get_reduced_screen()
			action_num = self.controller.act(state)
			action = self.move_list[action_num]

			# Run the action 4 times
			reward = 0.0
			for i in range(self.action_repeat):
				reward += self.ale.act(action)

			score += reward

			self.frame_number += 1

			# Cap reward to be between -1 and 1
			reward = min(max(reward, -1.0), 1.0)

			is_terminal = self.ale.game_over() or self.ale.lives() != num_lives
			num_lives = self.ale.lives()

			self.replay_memory.record(state, action_num, reward, is_terminal)

			if self.frame_number % 500000 == 0:
				print "Saving model..."
				self.controller.save(self.frame_number)

		return score


	def eval_controller(self, num_games=20):
		"""
		"""

		total_score = 0.0

		for i in range(num_games):
			total_score += self.play()

		return total_score / num_games


	def play(self, epsilon=0.1):
		"""
		Allow the controller to play the game
		"""

		total_score = 0

		# Ignore whether or not to update the screen
		old_show_while_training = self.show_while_training
		self.show_while_training = True

		# Reset the game to start a new episode
		self.ale.reset_game()

		while not self.ale.game_over():
			self.update_screen()

			state = self.get_reduced_screen()
			action_num = self.controller.base_controller.act(state)
			if np.random.random() < epsilon:
				action_num = np.random.randint(len(self.move_list))

			action = self.move_list[action_num]
			reward = self.ale.act(action)

			total_score += reward

		self.show_while_training = old_show_while_training

		return total_score


replay_memory = ReplayMemory(1000000)
#replay_memory.load('memory')

dqn_controller = DQNController((84,84,4), NATURE, 4, replay_memory)
controller = EpsilonController(dqn_controller, 4)
agi = AtariGameInterface('Breakout.bin', controller, replay_memory)

# Restore things

def run():
	cur_episode = 0
	num_frames = 0
	while agi.frame_number < 50000000:
		score = agi.learn()
		elapsed_frames = agi.frame_number - num_frames
		num_frames = agi.frame_number
		print "Episode %d:  Total Score = %d\t# Frames = %d\tTotal Frames = %d\tEpsilon: %f" % (cur_episode, score, elapsed_frames, num_frames, controller.epsilon)
		cur_episode += 1

	print
	print "Done Training.  Playing..."

	for i in range(25):
		print "  Game #" + str(i), "- Score:", agi.play()

if __name__ == '__main__':
	run()

