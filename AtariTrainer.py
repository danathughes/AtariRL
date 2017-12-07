## AtariTrainer.py 			Dana Hughes				21-Sept-2017
##
## Class to perform training and evaluation on Atari agents.

import numpy as np

class AtariTrainer:
	"""
	"""

	def __init__(self, environment, agent, counter, **kwargs):
		"""
		Load the game and create a display using pygame
		"""

		self.environment = environment
		
		# Hang on to the provided controller and replay memory
		self.agent = agent
		self.eval_agent = kwargs.get('eval_agent', agent)

		self.evaluate = False

		# Maximum number of no-op that can be performed at the start of an episode
		self.noop_max = kwargs.get('noop_max', 30)
		self.action_repeat = kwargs.get('action_repeat', 4)

		self.counter = counter

		# Listeners for storing parameters, tensorboard, etc.
		self.listeners = []
		

	def add_listener(self, listener):
		"""
		"""

		self.listeners.append(listener)


	def learn_episode(self):
		"""
		Allow for controller to learn while playing the game
		"""

		# Reset the game to start a new episode, and let the agent know
		self.environment.reset_game()
		self.agent.start_episode()

		num_lives = self.environment.lives()	

		score = 0

		for listener in self.listeners:
			listener.start_episode({})

		# Wait a random number of frames before starting
		for i in range(np.random.randint(self.noop_max)):
			self.environment.act(0)

		while not self.environment.terminal():
			state = self.environment.get_state()

			# Have the agent observe the environment, then act
			self.agent.observe(state)
			action, Q = self.agent.act()

			# Run the action 4 times
			reward = 0.0
			for i in range(self.action_repeat):
				reward += self.environment.act(action)

			score += reward

			self.counter.step()

			# Cap reward to be between -1 and 1
			reward = min(max(reward, -1.0), 1.0)

			for listener in self.listeners:
				listener.record({'Q': np.max(Q), 'reward': reward, 'action': action})

			is_terminal = self.environment.terminal() or self.environment.lives() != num_lives
			num_lives = self.environment.lives()

			self.agent.learn(action, reward, is_terminal)

		for listener in self.listeners:
			listener.end_episode({'score': score})

		return score


	def play_episode(self, num_noop = 0):
		"""
		Allow the eval agent to play
		"""

		total_score = 0

		# Reset the game to start a new episode
		self.environment.reset_game()

#		self.environment.display()

		# Perform a certain number of noops
		for i in range(num_noop):
			_ = self.environment.act(0)

		while not self.environment.terminal():
			state = self.environment.get_state()
#			self.environment.display()

			self.eval_agent.observe(state)
			action, Q = self.eval_agent.act()

			for i in range(self.action_repeat):
				reward = self.environment.act(action)
				total_score += reward

		return total_score