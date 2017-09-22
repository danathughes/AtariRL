## config_loader.py			Dana Hughes				21-Sept-2017
##
## Utility for loading a configuration from a config file


# Environments
from environments.AtariEnvironment import AtariEnvironment
from environments.OpenAIGymEnvironment import OpenAIGymEnvironment

# Networks
from models.networks import NIPS, NATURE, DUELING
from utils.builders.network_builders import *

# Memory
from memory.memory import ReplayMemory
from memory.priority_memory import PriorityReplayMemory, RankedPriorityReplayMemory
from memory.bootstrapped_memory import BootstrappedReplayMemory

# Agents
from agents.dqn_agent import DQN_Agent, DoubleDQN_Agent
from agents.bootstrapped_dqn_agent import Bootstrapped_DQN_Agent
from agents.epsilon_agent import EpsilonAgent

from utils.counter import Counter

import ConfigParser


def load(config_filename, sess):
	"""
	Loads a configuration of compoents from a file

	config_filename - Name of the file to load components from
	components      - A list of components to load from the file
	sess				 - The tensorflow session to run everything in
	"""

	# Does the filename exist?

	# Load the configuration file
	config = ConfigParser.RawConfigParser()
	config.read(config_filename)

	# Parse and construct each part
	environment = load_environment(config)
	network_builder, num_heads = load_network(config)
	memory = load_memory(config, environment, num_heads)

	# Make a counter
	counter = Counter()

	agent, eval_agent = load_agent(config, environment, network_builder, memory, counter, sess, num_heads)

	return environment, agent, eval_agent, counter



def load_agent(config, environment, network_builder, memory, counter, sess, num_heads):
	"""
	Load an agent from the ConfigParser
	"""

	# Which type of base agent is this?  Default to DQN
	try:
		agent_type = config.get('Agent', 'type')
	except:
		agent_type = "DQN"

	# Get all the agent parameters
	try:
		replay_start_size = config.getint('Agent', 'replay_start_size')
	except:
		replay_start_size = 50000

	try:
		target_update_frequency = config.getint('Agent', 'target_update_frequency')
	except:
		target_update_frequency = 10000

	try:
		update_frequency = config.getint('Agent', 'update_frequency')
	except:
		update_frequency = 4

	try:
		minibatch_size = config.getint('Agent', 'minibatch_size')
	except:
		minibatch_size = 32

	try:
		discount_factor = config.getfloat('Agent','discount_factor')
	except:
		discount_factor = 0.99

	try:
		history_size = config.getint('Agent', 'history_size')
	except:
		history_size = 4

	frame_shape = environment.screen_size
	num_actions = environment.num_actions

	# Build the agent!
	if agent_type == "DQN":
		dqn_agent = DQN_Agent(frame_shape, num_actions, history_size, network_builder, memory, counter,
									 replay_start_size=replay_start_size, target_update_frequency=target_update_frequency,
									 update_frequency=update_frequency, minibatch_size=minibatch_size, 
									 discount_factor=discount_factor, tf_session=sess)
	elif agent_type == "DoubleDQN":
		dqn_agent = DoubleDQN_Agent(frame_shape, num_actions, history_size, network_builder, memory, counter,
									 replay_start_size=replay_start_size, target_update_frequency=target_update_frequency,
									 update_frequency=update_frequency, minibatch_size=minibatch_size, 
									 discount_factor=discount_factor, tf_session=sess)
	elif agent_type == "BootstrappedDQN":
		dqn_agent = Bootstrapped_DQN_Agent(frame_shape, num_actions, history_size, network_builder, memory, num_heads, counter,
									 	replay_start_size=replay_start_size, target_update_frequency=target_update_frequency,
									 	update_frequency=update_frequency, minibatch_size=minibatch_size, 
									 	discount_factor=discount_factor, tf_session=sess)		

	# Create epsilon agents
	try:
		initial_epsilon = config.getfloat('Agent', 'initial_epsilon')
	except:
		initial_epsilon = 1.0

	try:
		final_epsilon = config.getfloat('Agent', 'final_epsilon')
	except:
		final_epsilon = 0.1

	try:
		initial_frame = config.getint('Agent', 'initial_epsilon_frame')
	except:
		initial_frame = 0

	try:
		final_frame = config.getint('Agent', 'final_epsilon_frame')
	except:
		final_frame = 1000000

	try:
		eval_epsilon = config.getint('Agent', 'evaluate_epsilon')
	except:
		eval_epsilon = 0.05

	# Make two agents -- a training epsilon agent, and an evaluation agent
	agent = EpsilonAgent(dqn_agent, initial_epsilon, final_epsilon, initial_frame, final_frame)
	eval_agent = EpsilonAgent(dqn_agent, eval_epsilon, eval_epsilon, 1, 1)

	return agent, eval_agent


def load_environment(config):
	"""
	Load an environment from the ConfigParser
	"""

	# Try to get values from the config file

	# Which game to load
	try:
		game_path = config.get('Environment', 'game_path')
	except:
		print "game_path not defined in Environment!"
		return None

	# Which environment class should be used?  Defaults to AtariEnvironment
	try:
		env_class = config.get('Environment', 'class')
	except:
		env_class = "AtariEnvironment"

	# Get the scaled screen dimensions - default is (84,84)
	try:
		width = config.getint('Environment', 'width')
	except:
		width = 84
	try:
		height = config.getint('Environment', 'height')
	except:
		height = 84

	# Build the environment
	if env_class == "AtariEnvironment":
		return AtariEnvironment(game_path, screen_size=(width, height))
	elif env_class == "OpenAIGym":
		return OpenAIGymEnvironment(game_path)
	else:
		print "Unknown environment class: %s" % env_class
		return None


def load_memory(config, environment, num_heads):
	"""
	Load memory from the ConfigParser
	"""

	# Which type of memory to use?  Default to ReplayMemory
	try:
		memory_type = config.get('Memory', 'type')
	except:
		memory_type = "ReplayMemory"

	try:
		base_memory_type = config.get('Memory', 'base_type')
	except:
		base_memory_type = "ReplayMemory"

	# Load all the parameters - use defaults as needed
	try:
		size = config.getint('Memory', 'size')
	except:
		size = 1000000

	try:
		alpha = config.getfloat('Memory', 'alpha')
	except:
		alpha = 0.6

	try:
		beta = config.getfloat('Memory', 'beta')
	except:
		beta = 0.4

	try:
		epsilon = config.getfloat('Memory', 'epsilon')
	except:
		epsilon = 1e-6

	try:
		mask_function = config.get('Memory', 'mask_function')
	except:
		mask_function = 'binomial'

	# Create the memory
	if memory_type == "ReplayMemory":
		memory = ReplayMemory(size, environment.screen_size)

	elif memory_type == "PriorityReplayMemory":
		memory = PriorityReplayMemory(size, environment.screen_size, alpha, beta, epsilon)

	elif memory_type == "RankedPriorityReplayMemory":
		memory = RankedPriorityReplayMemory(size, environment.screen_size, alpha, beta)

	elif memory_type == "BootstrappedReplayMemory":

		# Create the base memory, default to ReplayMemory
		if base_memory_type == "PriorityReplayMemory":
			base_memory = PriorityReplayMemory(size, environment.screen_size, alpha, beta, epsilon)
		elif base_memory_type == "RankedPriorityReplayMemory":
			base_memory = RankedPriorityReplayMemory(size, environment.screen_size, alpha, beta)
		else:
			base_memory = ReplayMemory(size, environment.screen_size)

		memory = BootstrappedReplayMemory(size, base_memory, num_heads)

	else:
		print "Unknown memory type: %s" % memory_type
		return None

	return memory



def load_optimizer(config, section):
	"""
	Load an optimizer from the ConfigParser
	"""

	pass


def load_network(config):
	"""
	Load a neural network builder
	"""

	# Which type of network agent is this?  Default to DQN
	try:
		network_type = config.get('Network', 'type')
	except:
		network_type = "DQN"

	# Which architecture should be used?  Default to NATURE
	try:
		architecture = config.get('Network', 'architecture')
	except:
		architecture = "NATURE"

	# Bootstrapped DQN requires knowledge of the number of heads.
	# Default to 10
	try:
		num_heads = config.getint('Network', 'num_heads')
	except:
		num_heads = 10


	# Use the appropriate architecture
	if architecture == "NIPS":
		layers = NIPS
	elif architecture == "NATURE":
		layers = NATURE
	elif architecture == "DUELING":
		layers = DUELING
	else:
		print "Unknown network architecture: %s" % architecture
		return None

	# Grab the appropriate network builder
	if network_type == "DQN":
		builder = create_dqn_builder(layers)
	elif network_type == "DuelingDQN":
		builder = create_dueling_dqn_builder(layers)
	elif network_type == "BootstrappedDQN":
		# Assume there is only one final layer to use as the head streams
		shared_layers = layers[:-1]
		head_layers = layers[-1:]
		builder = create_bootstrapped_dqn_builder(shared_layers, head_layers, num_heads)
	else:
		print "Unknown network type: %s" % network_type
		return None

	return builder, num_heads



