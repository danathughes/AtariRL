import numpy as np
import os

import scipy.ndimage as ndimage

import tensorflow as tf
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

		self.num_actions = num_actions

		self.dqn_layers = hidden_layers[0]
		self.target_dqn_layers = hidden_layers[1]

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
		self.target_update_frequency = kwargs.get('target_update_frequency', 10000)
		self.frame_number = 0
		self.param_updates = 0

		# Maximum number of no-op that can be performed at the start of an episode
		self.noop_max = kwargs.get('noop_max', 30)
		self.noop_count = 0
		self.no_noop = False

		# Initialize a Tensorflow session and create two DQNs
		self.current_DQN = DeepQNetwork(input_shape, self.dqn_layers, num_actions)
		self.target_DQN = DeepQNetwork(input_shape, self.target_dqn_layers, num_actions)

		# Session and training stuff
		self.sess = tf.InteractiveSession()
		self.trainer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum)
		self.train_step = self.trainer.minimize(self.current_DQN.objective())
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


	def createDataset(self, size):
		"""
		"""

		# Create and populate arrays for the input, target and mask for the DQN
		states = np.zeros((size,84,84,4))
		targets = np.zeros((size,self.num_actions))
		masks = np.zeros((size,self.num_actions))

		for i in range(size):
			state, action, reward, next_state, terminal = self.replay_memory.get_sample()
			states[i,:,:,:] = state

			# Calculate the target value
			if terminal:
				targets[i,action] = reward
			else:
				Q = self.target_DQN.get_Qs(next_state, self.sess)
				y = reward + self.discount_factor * np.max(Q)
				# Clip the target between -1 and 1
				y = max(y,-1.0)
				y = min(y,1.0)
				targets[i,action] = y

			# Only want to train for this particular action
			masks[i,action] = 1.0

		return states, targets, masks


	def update_target_network(self):
		"""
		"""

		print "Updating Target DQN..."
		
		# Copy the variables from the current DQN to the target DQN
		for i in range(len(self.dqn_layers)):
			dqn_layer = self.dqn_layers[i]
			target_layer = self.target_dqn_layers[i]

			params = dqn_layer.get_params(self.sess)
			target_layer.set_params(self.sess, params)


	def train(self):
		"""
		"""

		# Get the training data
		inputs, targets, masks = self.createDataset(self.minibatch_size)
		data = {'input': inputs, 'target': targets, 'mask': masks}

		# Train the network
		self.current_DQN.train(self.train_step, data)
		self.param_updates += 1

		if self.param_updates % self.target_update_frequency == 0:
			self.update_target_network()
