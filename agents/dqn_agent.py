## dqn_agent.py
##
## Agent which implements a Deep Q Network as a control policy
##

import tensorflow as tf
import numpy as np
import os

from models.DeepQNetwork import UpdateOperation

class DQN_Agent:
	"""
	Agent which implements a DQN to learn a policy
	"""

	def __init__(self, frame_shape, num_actions, history_size, network_builder, replay_memory, counter, **kwargs):

		"""
		action_update_rate - number of frames to repeat an action
		"""

		self.num_actions = num_actions
		self.history_size = history_size

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
		input_shape = frame_shape + (history_size,)
		self.dqn = network_builder(input_shape, num_actions, self.sess, network_name='dqn')
		self.target_dqn = network_builder(input_shape, num_actions, self.sess, network_name='target_dqn', trainable=False)

		self.update_operation = UpdateOperation(self.dqn, self.target_dqn, self.sess)

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

		pass


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

		# Select an action
		Q = self.dqn.get_Qs(self.state_history)
		action = np.argmax(Q)

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

		# Get what the normal output would be for the DQN
		targets = self.dqn.get_Qs(states)

		target_Q = np.zeros((size,))

		# Update the Q value of only the action
		Qmax = np.max(self.target_dqn.get_Qs(next_states), axis=1)
		Qnext = (1.0 - terminals.astype(np.float32)) * self.discount_factor * Qmax

		idx = np.arange(size)

		targets[idx,actions] = rewards[idx] + Qnext[idx]

		target_Q = rewards + Qnext

		# Calculate the TD error and inform the memory, for possible update
		TD_error = target_Q - self.dqn.get_Qs(states)[idx, actions]
		self.replay_memory.update(indices, TD_error)

		return states, targets, actions, target_Q, weights


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
		inputs, targets, actions, target_Q, weights = self.createDataset(self.minibatch_size)
		data = {'input': inputs, 'target': target_Q, 'action': actions, 'weights': weights}

		# Train the network
		loss = self.dqn.train(data)
		Qs = self.dqn.get_Qs(inputs)

		# Summarize the Q values and training loss
		training_data = {'training_loss': loss}
		for i in range(self.num_actions):
			training_data['Q%d_training' % i] = Qs[:,i]

		for listener in self.listeners:
			listener.record(training_data)
		


class DoubleDQN_Agent(DQN_Agent):
	"""
	Agent which implements a Double DQN to learn a policy
	"""

	def __init__(self, frame_shape, num_actions, history_size, network_builder, replay_memory, counter, **kwargs):
		"""
		"""

		# The Double DQN agent is almost exactly the same as a DQN agent, so this'll just
		# subclass a DQN agent and change the appropriate methods

		DQN_Agent.__init__(self, frame_shape, num_actions, history_size, network_builder, replay_memory, counter, **kwargs)


	def createDataset(self, size):
		"""
		"""

		# Create and populate arrays for the input, target and mask for the DQN
		experiences, indices, weights = self.replay_memory.get_samples(size)
		states, actions, rewards, next_states, terminals = experiences

		# Get what the normal output would be for the DQN
		targets = self.dqn.get_Qs(states)

		target_Q = np.zeros((size,))

		# Update the Q value of only the action

		# Double-Q learning:  let the DQN select which action would be performed on the next state
		dqn_next_Qs = self.dqn.get_Qs(next_states)
		dqn_next_actions = np.argmax(dqn_next_Qs, axis=1)

		idx = np.arange(size)

		# Use the Q value of the action selected by the DQN, *not* the max Q value
		Q_value = self.target_dqn.get_Qs(next_states)[idx, dqn_next_actions]
		Qnext = (1.0 - terminals.astype(np.float32)) * self.discount_factor * Q_value

		targets[idx,actions] = rewards[idx] + Qnext[idx]

		target_Q = rewards + Qnext

		# Calculate the TD error and inform the memory, for possible update
		TD_error = target_Q - self.dqn.get_Qs(states)[idx, actions]
		self.replay_memory.update(indices, TD_error)

		return states, targets, actions, target_Q, weights