import tensorflow as tf

from environments.AtariEnvironment import *
from utils.builders.network_builders import *
from utils.counter import *

from models.networks import NATURE

from agents.dqn_agent import DQN_Agent
from agents.epsilon_agent import EpsilonAgent

from memory.memory import ReplayMemory

import threading


class TrainingThread(threading.Thread):
	"""
	"""

	lock = threading.Lock()

	def __init__(self, environment, name, counter):
		"""
		"""

		threading.Thread.__init__(self)

		self.environment = environment
		self.name = name
		self.counter = counter

		print "Created Thread %s" % self.name

	def run(self):
		"""
		"""

		print "Starting Thread %s" % self.name

		self.environment.reset_game()

		while self.counter.count < 50000:
			if self.environment.terminal():
				self.environment.reset_game()

			screen = self.environment.get_state()
			reward = self.environment.act(0)

			TrainingThread.lock.acquire()
			self.counter.step()
			if self.counter.count % 500 == 0:
				print "%s - %d" % (self.name, self.counter.count)
			TrainingThread.lock.release()


# Make the network builder, memory and agent
counter = Counter()
mem = ReplayMemory(36)
dqn_builder = create_dqn_builder(NATURE)
dqn_agent = DQN_Agent((84,84), 4, 4, dqn_builder, mem, counter, replay_start_size=32)

threads = []
for i in range(4):
	environment = AtariEnvironment('roms/Breakout.bin', seed = i)
	threads.append(TrainingThread(environment, "Thread-%d"%i, counter))

print "MAIN THREAD: Starting threads"
for t in threads:
	t.start()

print "MAIN THREAD: Waiting until complete"
for t in threads:
	t.join()

print "MAIN THREAD: Terminating"