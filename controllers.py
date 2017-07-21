import numpy as np
import os

import scipy.ndimage as ndimage

import tensorflow as tf
from models.DeepQNetwork import *

from monitor import *


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
		else:
			self.epsilon = self.eps_final

		if np.random.random() < self.epsilon:
			action = np.random.randint(self.num_actions)

		return action


	def save(self, frame_number):
		"""
		"""

		self.base_controller.save(frame_number)



class DQNController:
	"""
	"""

	def __init__(self, input_shape, hidden_layers, num_actions, replay_memory, **kwargs):
		"""
		action_update_rate - number of frames to repeat an action
		"""

		self.num_actions = num_actions
		self.dqn_layers = hidden_layers

		# Need to query the replay memory for training examples
		self.replay_memory = replay_memory
		self.replay_start_size = kwargs.get('replay_start_size', 5000)

		# Discount factor, etc.
		self.discount_factor = kwargs.get('discount_factor', 0.99)
		self.minibatch_size = kwargs.get('minibatch_size', 32)

		# Count the actions to determine action repeats and update frequencies
		self.update_frequency = kwargs.get('update_frequency', 4)

		# Where to store checkpoints, Tensorboard logs, etc
		self.save_path = kwargs.get('save_path', './checkpoints/model_checkpoint')
		self.log_dir = kwargs.get('tensorboard_log_dir', '/home/dana/Research/AtariRL/tensorboard/')

		# How many frames has the controller seen?
		self.episode_number = 0
		self.frame_number = 0
		
		# Should the network train?
		self.can_train = False

		# Keep track of frames to know when to train, switch networks, etc.
		self.target_update_frequency = kwargs.get('target_update_frequency', 1000)

		self.best_Q = np.zeros((self.num_actions,))

		# Did the user provide a session?
		self.sess = kwargs.get('tf_session', tf.InteractiveSession())

		# Initialize a Tensorflow session and create two DQNs
		self.dqn = DeepQNetwork(input_shape, self.dqn_layers, num_actions, self.sess, network_name='dqn')
		self.target_dqn = DeepQNetwork(input_shape, self.dqn_layers, num_actions, self.sess, network_name='target_dqn', trainable=False)

		self.update_operation = UpdateOperation(self.dqn, self.target_dqn, self.sess)

		# Create an operator to update the target weights from the current DQN
#		self.update_operations = []

#		with tf.variable_scope('update_operation'):
#			for name in self.dqn.params:
#				op = self.target_DQN.params[name].assign(self.dqn.params[name].value())
#				self.update_operations.append(op)


		# Be able to save and restore the dqn checkpoints
		self.saver = tf.train.Saver(var_list=self.dqn.params, max_to_keep=1)

		tf.global_variables_initializer().run()

		# Should a model be loaded?
#		self.restore_path = kwargs.get('restore_path', '/home/dana/Research/AtariRL/checkpoints')
		self.restore_path = None
		if self.restore_path is not None:
			print "Restoring Model..."
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.restore_path))

		# Summaries
		self.tensorboard_monitor = TensorboardMonitor(self.log_dir, self.sess)
		self.tensorboard_monitor.add_dqn_summary(self.dqn)

		# Maintain a history of the previous states for use as input
		self.state_history = np.zeros((84,84,4))

		# Make sure that the DQN and target network are the same before beginning.
		self.update_target_network()


	def save(self, frame_number):
		"""
		Save the DQN model
		"""

		self.saver.save(self.sess, self.save_path, global_step=frame_number)
		self.replay_memory.save('./memory')


	def act(self, state):
		"""
		Update the state history and select an action
		"""

		self.state_history[:,:,0:3] = self.state_history[:,:,1:4]
		self.state_history[:,:,3] = state

		# Select an action
		Q = self.dqn.get_Qs(self.state_history)
		action = np.argmax(Q)


		for i in range(self.num_actions):
			self.best_Q[i] = max(self.best_Q[i], Q[i])

		# Has enough frames occured to start training?
		if self.frame_number == self.replay_start_size:
			self.can_train = True
			print "Start Training..."

		# Should training occur?  
		if self.frame_number % self.update_frequency == 0 and self.can_train:
			self.train()

		# Should the target network be updated?
		if self.frame_number % self.target_update_frequency == 0:
			self.update_target_network()

		self.frame_number += 1

		return action


	def createDataset(self, size):
		"""
		"""

		# Create and populate arrays for the input, target and mask for the DQN
		states, actions, rewards, next_states, terminals = self.replay_memory.get_samples(32)

		# Get what the normal output would be for the DQN
		targets = self.dqn.get_Qs(states)

		target_Q = np.zeros((size,))

		# Update the Q value of only the action
		Qmax = np.max(self.target_dqn.get_Qs(next_states), axis=1)
		Qnext = (1.0 - terminals.astype(np.float32)) * self.discount_factor * Qmax

		idx = np.arange(size)

		targets[idx,actions] = rewards[idx] + Qnext[idx]

		target_Q = rewards + Qnext

		return states, targets, actions, target_Q


	def update_target_network(self):
		"""
		Copy the currently trained DQN into the target network
		"""

		print "Updating Target DQN..."
		
		self.update_operation.run()


	def train(self):
		"""
		Train the network
		"""

		# Get the training data
		inputs, targets, actions, target_Q = self.createDataset(self.minibatch_size)
		data = {'input': inputs, 'target': target_Q, 'action': actions}

		# Train the network
		loss = self.dqn.train(data)

		# Summarize the Q values
		self.tensorboard_monitor.summarize(['q_summary'], self.frame_number, {self.dqn.input: inputs})
