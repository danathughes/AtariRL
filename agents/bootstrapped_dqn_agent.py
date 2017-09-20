## bootstrapped_dqn_agent.py
##
## Agent which implements a Bootstrapped Deep Q Network as a control policy
##

import numpy as np
import os

import scipy.ndimage as ndimage

import tensorflow as tf
from models.BootstrappedDeepQNetwork import *
# from models.DuelingDeepQNetwork import *

from listeners.tensorboard_monitor import *

class Bootstrapped_DQN_Agent:
	"""
	Agent which implements a Bootstrapped DQN to learn a policy
	"""

	def __init__(self, input_shape, shared_layers, head_layers, num_actions, num_heads, replay_memory, counter, config, **kwargs):
		"""
		action_update_rate - number of frames to repeat an action
		"""

		self.num_actions = num_actions
		self.num_heads = num_heads
		self.head_number = 0

		# Which frame / step are we on 
		self.counter = counter

		# Need to query the replay memory for training examples
		self.replay_memory = replay_memory
		self.replay_start_size = kwargs.get('replay_start_size', 5000)

		# Discount factor, etc.
		self.discount_factor = kwargs.get('discount_factor', 0.99)
		self.minibatch_size = kwargs.get('minibatch_size', 32)

		# Count the actions to determine action repeats and update frequencies
		self.update_frequency = kwargs.get('update_frequency', 4)

		# Should the network train?
		if self.counter.count >= self.replay_start_size:
			self.can_train = True
			print "DQN can train..."
		else:
			self.can_train = False

		# Keep track of frames to know when to train, switch networks, etc.
		self.target_update_frequency = kwargs.get('target_update_frequency', 1000)

		# Did the user provide a session?
		self.sess = kwargs.get('tf_session', tf.InteractiveSession())

		# Initialize a Tensorflow session and create two DQNs
		self.dqn = BootstrappedDeepQNetwork(input_shape, shared_layers, head_layers, num_actions, self.num_heads, self.sess, network_name='dqn')
		self.target_dqn = BootstrappedDeepQNetwork(input_shape, shared_layers, head_layers, num_actions, self.num_heads, self.sess, network_name='target_dqn', trainable=False)

		self.update_operation = UpdateOperation(self.dqn, self.target_dqn, self.sess)

		# Maintain a history of the previous states for use as input
		self.state_history = np.zeros((84,84,4))

		self.listeners = []


	def add_listener(self, listener):
		"""
		"""

		self.listeners.append(listener)



	def start_episode(self):
		"""
		"""

		self.head_number = np.random.randint(self.num_heads)		


	def observe(self, state):
		"""
		Receive an observation
		"""

		self.state_history[:,:,0:3] = self.state_history[:,:,1:4]
		self.state_history[:,:,3] = state


	def act(self):
		"""
		Select an action based on the current state history
		"""

		# Figure out the action selected by each head
		Qs = self.dqn.get_Qs(self.state_history)
		actions = np.argmax(Qs, axis=1)

		# Select the action of the control head
		action = actions[self.head_number]
		Q = Qs[self.head_number]

		return action, Q


	def learn(self, action, reward, is_terminal):
		"""
		Learn from the action taken (maybe due to environmental influence, etc), provided reward and next state
		"""

		# Add the experience to the replay memory
		self.replay_memory.record(self.state_history[:,:,3], action, reward, is_terminal)		

		# Has enough frames occured to start training?
		if self.counter.count == self.replay_start_size:
			self.can_train = True
			print "Start Training..."

		# Should training occur?  
		if self.counter.count % self.update_frequency == 0 and self.can_train:
			self.train()

		# Should the target network be updated?
		if self.counter.count % self.target_update_frequency == 0:
			self.update_target_network()


	def createDataset(self, size):
		"""
		"""

		# Create and populate arrays for the input, target and mask for the DQN
		experiences, indices, weights = self.replay_memory.get_samples(32)
		states, actions, rewards, next_states, terminals = experiences
		masks = self.replay_memory.get_masks(indices)

		# Get what the normal output would be for the DQN
		targets = self.dqn.get_Qs(states)

		target_Q = np.zeros((size,self.num_heads))

		# Update the Q value of only the action
		target_DQN_Qs = self.target_dqn.get_Qs(next_states)
		Qmax = np.max(self.target_dqn.get_Qs(next_states), axis=2)
		Qnext = np.reshape((1.0 - terminals.astype(np.float32)), (size,1)) * self.discount_factor * Qmax

		idx = np.arange(size)

		targets[idx,:,actions] = np.reshape(rewards, (size,1))[idx] + Qnext[idx]

		target_Q = np.reshape(rewards, (size,1)) + Qnext

		# Calculate the TD error and inform the memory, for possible update
		#TD_error = target_Q - self.dqn.get_Qs(states)[idx, :, actions]
		#self.replay_memory.update(indices, TD_error)

		return states, targets, actions, target_Q, weights, masks


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
		inputs, targets, actions, target_Q, weights, masks = self.createDataset(self.minibatch_size)
		data = {'input': inputs, 'target': target_Q, 'action': actions, 'weights': weights, 'masks': masks}

		# Train the network
		loss = self.dqn.train(data)
		Qs = self.dqn.get_Qs(inputs)[self.head_number]

		# Summarize the Q values and training loss
		training_data = {'training_loss': loss}
		for i in range(self.num_actions):
			training_data['Q%d_training' % i] = Qs[:,i]

		for listener in self.listeners:
			listener.record(training_data)
		