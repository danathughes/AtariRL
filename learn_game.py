##
##
##

# Import libraries to simulate Atari and display results
from ale_python_interface import ALEInterface
import pygame
from pygame.locals import *

import numpy as np
import os

from models.DeepQNetwork import *


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


class EpsilonController:
	"""
	"""

	def __init__(self, base_controller, num_actions, initial_exploration=1.0, final_exploration=0.1, final_frame=1000000):
		"""
		"""

		self.epsilon = initial_exploration
		self.eps_init = initial_exploration
		self.eps_final = final_exploration

		self.current_frame = 0
		self.final_frame = final_frame

		self.base_controller = base_controller
		self.num_actions = num_actions


	def act(self, state):
		"""
		"""

		action = self.base_controller.act(state)

		self.current_frame += 1

		if self.current_frame < self.final_frame:
			self.epsilon = self.eps_init + (self.eps_final - self.eps_init)*(float(self.current_frame)/self.final_frame)
			if self.current_frame % 1000 == 0:
				print "Epsilon =", self.epsilon
		else:
			self.epsilon = self.eps_final

		if np.random.random() < self.epsilon:
			action = np.random.randint(self.num_actions)

		return action



class DQNController:
	"""
	"""

	def __init__(self, input_shape, hidden_layers, num_actions, replay_memory, **kwargs):
		"""
		action_update_rate - number of frames to repeat an action
		"""

		# Need to query the replay memory for training examples
		self.replay_memory = replay_memory
		self.replay_start_size = kwargs.get('replay_start_size', 50000)

		# Discount factor, learning rate, momentum, etc.
		self.learning_rate = kwargs.get('learning_rate', 0.00025)
		self.momentum = kwargs.get('momentum', 0.95)
		self.squared_momentum = kwargs.get('squared_momentum', 0.95)
		self.min_squared_gradient = kwargs.get('min_squared_gradient', 0.01)
		self.discount_factor = kwargs.get('discount_factor', 0.99)
		self.minibatch_size = kwargs.get('minibatch_size', 32)

		# Count the actions to determine action repeats and update frequencies
		self.action_repeat = kwargs.get('action_repeat', 4)
		self.update_frequency = kwargs.get('update_frequency', 4)
		self.action_count = 0
		self.current_action = 0

		# Keep track of frames to know when to train, switch networks, etc.
		self.frame_number = 0

		# Maximum number of no-op that can be performed at the start of an episode
		self.noop_max = kwargs.get('noop_max', 30)
		self.noop_count = 0
		self.no_noop = False

		# Initialize a Tensorflow session and create two DQNs
		self.sess = tf.InteractiveSession()
		self.current_DQN = DeepQNetwork(input_shape, hidden_layers, num_actions)
		self.target_DQN = DeepQNetwork(input_shape, hidden_layers, num_actions)
		self.trainer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum)
		self.current_train_step = self.trainer.minimize(self.current_DQN.objective())
		self.target_train_step = self.trainer.minimize(self.target_DQN.objective())
		self.sess.run(tf.initialize_all_variables())

		# Maintain a history of the previous states
		self.state_history = np.zeros((84,84,4))


	def act(self, state):
		"""
		Update the state history and select an action
		"""

		self.state_history[:,:,1:4] = self.state_history[:,:,0:3]
		self.state_history[:,:,0] = state

		# Time to generate a new action?
		if self.frame_number % self.action_repeat == 0:
			Qs = self.current_DQN.get_Qs(self.state_history, self.sess)
			self.current_action = np.argmax(Qs)
			self.action_count += 1

		self.frame_number += 1

		# Should the neural net be trained?
		if self.action_count % self.update_frequency == 0 and self.frame_number >= self.replay_start_size:
			self.train()

		return self.current_action


	def train(self):
		"""
		"""

		print "Training..."


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


	def get_sample(self, history_length):
		"""
		Return a single sample
		"""

		state = np.zeros((84,84,4))
		next_state = np.zeros((84,84,4))

		# Get an appropriate index 
		if self.filled:
			idx = np.random.randint(self.memory_size)
			# Avoid indices where the next state is unavailable, or 3 prior states are not
			while idx >= self._idx and idx < self._idx + 4:
				idx = np.random.randint(self.memory_size)

		else:
			idx = np.random.randint(3, self._idx-2)

		# Get the current and next state
		for i in range(3):
			state[:,:,i] = self.states[idx-i,:,:]



			

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
#controller = RandomController(4)

replay_memory = ReplayMemory()
dqn_controller = DQNController((84,84,4), DEEPMIND_LAYERS, 4, replay_memory)
controller = EpsilonController(dqn_controller, 4)
agi = AtariGameInterface('Breakout.bin', controller, replay_memory)