##
##
##

# Import libraries to simulate Atari and display results
#from ale_python_interface import ALEInterface
#import pygame
#from pygame.locals import *

import numpy as np
import os

#import scipy.ndimage as ndimage

from models.DeepQNetwork import *

from agents.epsilon_agent import EpsilonAgent
from memory.memory import ReplayMemory
from memory.priority_memory import PriorityReplayMemory

from environments.AtariEnvironment import AtariEnvironment

import tensorflow as tf

from listeners.checkpoint_recorder import *
from listeners.tensorboard_monitor import *


import config



class Counter:
	"""
	Simple class to maintain a shared counter between objects
	"""

	def __init__(self, initial_count=0):
		"""
		"""

		self.count = initial_count


	def step(self):
		"""
		Increment the counter
		"""

		self.count += 1



class AtariGameInterface:
	"""
	"""

	def __init__(self, environment, agent, counter, config, **kwargs):
		"""
		Load the game and create a display using pygame
		"""

		self.environment = environment
		
		# Hang on to the provided controller and replay memory
		self.agent = agent

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

		# Reset the game to start a new episode
		self.environment.reset_game()

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

game_filename='roms/Breakout.bin'

sess = tf.InteractiveSession()
counter = Counter()

environment = AtariEnvironment(game_filename)
num_actions = environment.num_actions()
replay_memory = config.Memory(config.memory_size, config.screen_width, config.screen_height)
dqn_agent = config.Agent((config.screen_width,config.screen_height,config.history_length), config.network_layers, num_actions, replay_memory, counter, config, tf_session=sess)
agent = EpsilonAgent(dqn_agent, num_actions, counter)
agi = AtariGameInterface(environment, agent, counter, config)

# Create a Tensorboard monitor and populate with the desired summaries
tensorboard_monitor = TensorboardMonitor('./log/breakout/dueling-dqn/', sess, counter)
tensorboard_monitor.add_scalar_summary('score', 'per_game_summary')
tensorboard_monitor.add_scalar_summary('training_loss', 'training_summary')
for i in range(num_actions):
	tensorboard_monitor.add_histogram_summary('Q%d_training' % i, 'training_summary')

checkpoint_monitor = CheckpointRecorder(dqn_agent.dqn, replay_memory, counter, './checkpoints/breakout/dueling-dqn/', sess)
agi.add_listener(checkpoint_monitor)
agi.add_listener(tensorboard_monitor)
dqn_agent.add_listener(tensorboard_monitor)

sess.run(tf.global_variables_initializer())

# Restore things
#dqn_agent.dqn.restore('./checkpoints/dqn/4000000')
#dqn_agent.update_target_network()
#replay_memory.load('./checkpoints/replay_memory/4000000')

#dqn_agent.dqn.restore('./old_stuff/old_checkpoints/dqn/1800000//dqn_model-1800000')
#dqn_agent.update_target_network()


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

