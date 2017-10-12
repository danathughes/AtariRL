## epsilon_agent.py
##
## Agent which implements a Deep Q Network as a control policy
##

import numpy as np

class EpsilonAgent:
	"""
	Decorator Agent which implements an epsilon-greedy policy.  Uses a second agent to produce non-random actions
	"""

	def __init__(self, agent, counter, initial_exploration=1.0, final_exploration=0.1, initial_frame=0, final_frame=1000000):
		"""
		"""

		self.epsilon = initial_exploration
		self.eps_init = initial_exploration
		self.eps_final = final_exploration

		self.counter = counter
		self.initial_frame = initial_frame
		self.final_frame = final_frame

		self.base_agent = agent
		self.num_actions = agent.num_actions


	def start_episode(self):
		"""
		"""

		self.base_agent.start_episode()


	def observe(self, state):
		"""
		Receive an observation
		"""

		self.base_agent.observe(state)



	def act(self):
		"""
		"""

		# The base agent may require acting for internal purposes, so allow the agent ot act
		action, Q = self.base_agent.act()

		if self.counter.count < self.initial_frame:
			self.epsilon = self.eps_init
		elif self.counter.count < self.final_frame:
			self.epsilon = self.eps_init + (self.eps_final - self.eps_init)*(float(self.counter.count - self.initial_frame)/self.final_frame)
		else:
			self.epsilon = self.eps_final

		# Should the agent perform a random action?
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.num_actions)

		return action, Q


	def learn(self, action, reward, is_terminal):
		"""
		Learn from the action taken (maybe due to environmental influence, etc), provided reward and next state
		"""

		self.base_agent.learn(action, reward, is_terminal)
