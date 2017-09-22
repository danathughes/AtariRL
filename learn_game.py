##
##
##

# Import libraries to simulate Atari and display results
#from ale_python_interface import ALEInterface
#import pygame
#from pygame.locals import *

# Core components
import tensorflow as tf
import numpy as np
import os

from utils.config_loader import load as load_config

# Listeners to perform various bookkeeping tasks
from listeners.checkpoint_recorder import *
from listeners.tensorboard_monitor import *



class AtariGameInterface:
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
		Allow the controller to play the game
		"""

		total_score = 0

		# Reset the game to start a new episode
		self.environment.reset_game()

		self.environment.display()

		# Perform a certain number of noops
		for i in range(num_noop):
			_ = self.environment.act(0)

		while not self.environment.terminal():
			state = self.environment.get_state()
			self.environment.display()

			self.agent.observe(state)
			action, Q = self.agent.act()

			for i in range(self.action_repeat):
				reward = self.environment.act(action)
				total_score += reward

		return total_score


sess = tf.InteractiveSession()

environment, agent, eval_agent, counter = load_config('example.cfg', sess)
dqn_agent = agent.base_agent

# Put it all together!
agi = AtariGameInterface(environment, agent, counter, eval_agent=eval_agent)

# Create a Tensorboard monitor and populate with the desired summaries
tensorboard_monitor = TensorboardMonitor('./log/breakout/dueling-dqn/', sess, counter)
tensorboard_monitor.add_scalar_summary('score', 'per_game_summary')
tensorboard_monitor.add_scalar_summary('training_loss', 'training_summary')
for i in range(agent.num_actions):
	tensorboard_monitor.add_histogram_summary('Q%d_training' % i, 'training_summary')

checkpoint_monitor = CheckpointRecorder(dqn_agent.dqn, dqn_agent.replay_memory, counter, './checkpoints/breakout/dueling-dqn/', sess)
agi.add_listener(checkpoint_monitor)
agi.add_listener(tensorboard_monitor)
dqn_agent.add_listener(tensorboard_monitor)

sess.run(tf.global_variables_initializer())



def run():
	cur_episode = 0
	num_frames = 0
	while counter.count < 50000000:
		score = agi.learn_episode()

		tensorboard_monitor.record({'score': score})

		elapsed_frames = counter.count - num_frames
		num_frames = counter.count
		print "Episode %d:  Total Score = %d\t# Frames = %d\tTotal Frames = %d\tEpsilon: %f" % (cur_episode, score, elapsed_frames, num_frames, agent.epsilon)
		cur_episode += 1

	print
	print "Done Training.  Playing..."

#	for i in range(25):
#		print "  Game #" + str(i), "- Score:", agi.play()

if __name__ == '__main__':
	run()

