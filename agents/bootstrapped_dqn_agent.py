## bootstrapped_dqn_agent.py
##
## Agent which implements a Bootstrapped Deep Q Network as a control policy
##

import numpy as np
import os

import tensorflow as tf
from models import operations

class Bootstrapped_DQN_Agent:
	"""
	Agent which implements a Bootstrapped DQN to learn a policy
	"""

	def __init__(self, frame_shape, num_actions, history_size, network_builder, replay_memory, num_heads, **kwargs):
		"""
		action_update_rate - number of frames to repeat an action
		"""

		self.num_actions = num_actions
		self.history_size = history_size

		self.num_heads = num_heads
		self.head_number = 0

		# Need to query the replay memory for training examples
		self.replay_memory = replay_memory

		# Discount factor, etc.
		self.discount_factor = kwargs.get('discount_factor', 0.99)
		self.minibatch_size = kwargs.get('minibatch_size', 32)

		# Initialize a Tensorflow session and create two DQNs
		input_shape = frame_shape + (history_size,)

#		self.dqn = network_builder(input_shape, num_actions, network_name='dqn')
#		self.target_dqn = network_builder(input_shape, num_actions, network_name='target_dqn', trainable=False)

		self.dqn = dqn
		self.target_dqn = target_dqn


		self.update_operation = operations.Update(self.dqn, self.target_dqn)

		# Maintain a history of the previous states for use as input
		self.state_history = np.zeros(input_shape)

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


	def createDataset(self, size):
		"""
		"""

		# Create and populate arrays for the input, target and mask for the DQN
		experiences, indices, weights = self.replay_memory.get_samples(size)
		states, actions, rewards, next_states, terminals = experiences
		masks = self.replay_memory.get_masks(indices)

		# The rewards and terminals need to be reshaped to work with multiple heads
		rewards = np.reshape(rewards, (size,1))
		terminals = np.reshape(terminals, (size,1))

		# Get what the normal output would be for the DQN
		targets = self.dqn.get_Qs(states)

		# Update the Q value of only the action
		Qmax = np.max(self.target_dqn.get_Qs(next_states), axis=2)
		Qnext = (1.0 - terminals.astype(np.float32)) * self.discount_factor * Qmax

		idx = np.arange(size)

		targets[idx,:,actions] = rewards[idx] + Qnext[idx]

		target_Q = rewards + Qnext

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
		