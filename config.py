## config.py             Dana Hughes									19-Sept-2017
##
## Configuration file storing all the parameters used for Atari RL training
##

from models.DeepQNetwork import DeepQNetwork
from models.DuelingDeepQNetwork import DuelingDeepQNetwork

from agents.dqn_agent import DQN_Agent, DoubleDQN_Agent
from agents.epsilon_agent import EpsilonAgent

from memory.memory import ReplayMemory
from memory.priority_memory import PriorityReplayMemory, RankedPriorityReplayMemory

from models import networks



##
## Environment Configuration
##
## screen_width - how many pixels wide the scaled down version of the screen to use as input
## screen_height - how many pixels high the scaled down version of the screen to use as input
## history_length - number of screens stacked and used as input for the agent
## action_repeat - how many times to repeat the same action / screens to skip
screen_width, screen_height = 84, 84
history_length = 4
action_repeat = 4

##
## Agent Configuration
##
Agent = DQN_Agent


# Which type of network to use
#
# DeepQNetwork - Network used in the Nature paper
# DuelingDeepQNetwork - Network used in Dueling DQN paper - 
Network = DeepQNetwork
network_layers = networks.NATURE


# Optimization parameters



# Which type of memory to use,
#
# ReplayMemory - Simple Replay Memory from Nature and NIPS paper
# PriorityReplayMemory - Priority Replay Memory from Priority Replay Memory paper
# RankedPriorityReplayMemory - Rank-based Priority Replay Memory
Memory = ReplayMemory
memory_size = 1000000

## 
## Training Configuration
##
## num_training_steps - Total number of training steps to perform
## step_num - Starting step number
## 
