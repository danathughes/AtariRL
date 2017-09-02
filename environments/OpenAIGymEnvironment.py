import gym

import numpy as np

import scipy.ndimage as ndimage


class OpenAIGymEnvironment:
	"""
	Environment for playing Atari games using OpenAI's gym
	"""

	def __init__(self, game_ame, **kwargs):
		"""
		Create an environment with the provided game
		"""

		# Create the gym environment with the provided frame
		self.env = gym.make(game_name)

		# Reset the game and grab the current screen, just so it's available, 
		# and set the game to not terminal
		self.game_screen = self.env.reset()
		self.is_terminal = False

		# Grab the set of available moves for this game
		self.move_list = range(env.action_space.n)
		print "Number of Moves:", len(self.move_list)

		self.listeners = []


	def get_state(self):
		"""
		Convert current screen to 84x84 np array of luminescence values. 
		"""

		# Convert the game screen to luminosity (gray scale)
		gray_screen = np.dot(self.game_screen, np.array([0.299, 0.587, 0.114])).astype(np.uint8)
		gray_screen = ndimage.zoom(gray_screen, (0.4, 0.525))

		return gray_screen


	def act(self, action):
		"""
		Perform an action on the environment
		"""

		self.game_screen, reward, self.is_terminal, info = env.step(action)




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

		self.game_screen = self.env.reset()
		self.is_terminal = False