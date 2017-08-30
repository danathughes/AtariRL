## epsilon_agent.py
##
## Agent which implements a Deep Q Network as a control policy
##

import numpy as np

class EpsilonAgent:
	"""
	Decorator Agent which implements an epsilon-greedy policy.  Uses a second agent to produce non-random actions
	"""

	def __init__(self, base_controller, num_actions, counter, initial_exploration=1.0, final_exploration=0.1, final_frame=1000000):
		"""
		"""

		self.epsilon = initial_exploration
		self.eps_init = initial_exploration
		self.eps_final = final_exploration

		self.counter = counter
		self.final_frame = final_frame

		self.base_controller = base_controller
		self.num_actions = num_actions


	def act(self, state):
		"""
		"""

		action, Q = self.base_controller.act(state)

		if self.counter.count < self.final_frame:
			self.epsilon = self.eps_init + (self.eps_final - self.eps_init)*(float(self.counter.count)/self.final_frame)
		else:
			self.epsilon = self.eps_final

		if np.random.random() < self.epsilon:
			action = np.random.randint(self.num_actions)

		return action, Q
