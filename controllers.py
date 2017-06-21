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

		self.dqn_layers = hidden_layers[0]
		self.target_dqn_layers = hidden_layers[1]

		# Need to query the replay memory for training examples
		self.replay_memory = replay_memory
		self.replay_countdown = kwargs.get('replay_start_size', 50000)

		# Discount factor, learning rate, momentum, etc.
		self.learning_rate = kwargs.get('learning_rate', 0.00025)
		self.momentum = kwargs.get('momentum', 0.95)
		self.discount_factor = kwargs.get('discount_factor', 0.99)
		self.minibatch_size = kwargs.get('minibatch_size', 32)

		self.epsilon = kwargs.get('epsilon', 1e-10)
		self.decay = kwargs.get('decay', 0.01)

		# Count the actions to determine action repeats and update frequencies
		self.action_repeat = kwargs.get('action_repeat', 4)
		self.update_frequency = kwargs.get('update_frequency', 4)
		self.action_count = 0
		self.update_count = 0
		self.current_action = 0
		self.frame_count = 0

		# Should the network train?
		self.can_train = False

		# Keep track of frames to know when to train, switch networks, etc.
		self.target_update_frequency = kwargs.get('target_update_frequency', 10000)
		self.param_updates = 0

		# Initialize a Tensorflow session and create two DQNs
		with tf.name_scope('dqn'):
			self.current_DQN = DeepQNetwork(input_shape, self.dqn_layers, num_actions, namespace='dqn')
		with tf.name_scope('target_dqn'):
			self.target_DQN = DeepQNetwork(input_shape, self.target_dqn_layers, num_actions, namespace='target_dqn', trainable=False)
		


		# Session and training stuff
		self.sess = tf.InteractiveSession()
		self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay, momentum=self.momentum, epsilon=self.epsilon)
#		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

		# Need to clip gradients between -1 and 1 to stabilize learning
		grads_and_vars = self.optimizer.compute_gradients(self.current_DQN.objective())
		capped_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) if grad is not None else (None, var) for grad, var in grads_and_vars]
		self.train_step = self.optimizer.apply_gradients(capped_grads_and_vars)

#		self.train_step = self.trainer.minimize(self.current_DQN.objective())

		self.merged_summaries = tf.summary.merge_all()
		self._writer = tf.summary.FileWriter('./tensorboard/', self.sess.graph)

		# Be able to save and restore checkpoints
		self.saver = tf.train.Saver()
		self.save_path = kwargs.get('save_path', './checkpoints/model_checkpoint')

		tf.global_variables_initializer().run()

		# Maintain a history of the previous states
		self.state_history = np.zeros((84,84,4))

		self.update_target_network()


	def save(self, frame_number):
		"""
		Save the DQN model
		"""

		self.saver.save(self.sess, self.save_path, global_step=frame_number)



	def act(self, state):
		"""
		Update the state history and select an action
		"""

		self.state_history[:,:,1:4] = self.state_history[:,:,0:3]
		self.state_history[:,:,0] = state

		# Time to generate a new action?	
		if self.action_count == 0:
			Q = self.current_DQN.get_Qs(self.state_history, self.sess)
			self.current_action = np.argmax(Q)
			self.action_count = self.action_repeat + 1
			self.update_count += 1

			if self.replay_countdown % 1000 == 0:
				print "Q =", Q

		self.action_count -= 1

		# Should the neural net be trained?
		if self.update_count >= self.update_frequency and self.can_train:
			self.update_count = 0
			self.train()

		self.param_updates += 1

		if self.param_updates >= self.target_update_frequency and self.can_train:
			self.param_updates = 0
			self.update_target_network()

		# Decrement the replay countdown and allow training if it reaches 0
		self.replay_countdown -= 1
		if self.replay_countdown == 0:
			self.can_train = True
			print "Start Training..."

		self.frame_count += 1


		return self.current_action


	def createDataset(self, size):
		"""
		"""

		# Create and populate arrays for the input, target and mask for the DQN
		states, actions, rewards, next_states, terminals = self.replay_memory.get_samples(size)

		# Get what the normal output would be for the DQN
		targets = self.current_DQN.get_Qs(states, self.sess)

		# Update the Q value of only the action
		Qmax = np.max(self.target_DQN.get_Qs(next_states, self.sess), axis=1)
		Qfuture = (1.0 - terminals.astype(np.int)) * self.discount_factor * Qmax

		idx = np.arange(size)

		targets[idx,actions] = rewards[idx] + Qfuture[idx]

		return states, targets


	def update_target_network(self):
		"""
		Copy the currently trained DQN into the target network
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
		Train the network
		"""
		
		# Get the training data
		inputs, targets = self.createDataset(self.minibatch_size)
		data = {'input': inputs, 'target': targets}

		# Train the network
		self.current_DQN.train(self.train_step, data)

		# Summarize data
		summaries = self.current_DQN.get_summary(self.merged_summaries, data)
		self._writer.add_summary(summaries, self.frame_count)
#			self.param_updates += 1

#			if self.param_updates == self.target_update_frequency:
#				self.param_updates = 0
#				self.update_target_network()

#			self._writer.add_summary(summary, self.param_updates)
