from models.parts import *

import models.DeepQNetwork as DQN
import models.DuelingDeepQNetwork as DDQN
#import models.BootstrappedDeepQNetwork as BDQN

# Network used in NIPS workshop paper
NIPS  = [Convolutional((8,8), 16, name='conv1', stride=4),
          Convolutional((4,4), 32, name='conv2', stride=2),
          Flatten(name='flatten'),
          FullConnection(256, name='full1')
        ]

# Network used in Nature paper
NATURE = [Convolutional((8,8), 32, name='conv1', stride=4),
           Convolutional((4,4), 64, name='conv2', stride=2),
           Convolutional((3,3), 64, name='conv3', stride=1),
           Flatten(name='flatten'),
           FullConnection(512, name='full1')
         ]

# Network used for deuling DQN.  3-tuple, 2nd & 3rd elements are value and
# advantage streams
DUELING = ([Convolutional((8,8), 32, name='conv1', stride=4),
            Convolutional((4,4), 64, name='conv2', stride=2),
            Convolutional((3,3), 64, name='conv3', stride=1),
            Flatten(name='flatten')],
           [FullConnection(512, name='full_value')],
           [FullConnection(512, name='full_advantage')]
          )



def nips_dqn(input_shape, num_actions, network_name='dqn', trainable=True):
  """
  Create a version of the DQN network from NIPS
  """

  # Add the output layer to the end of the network layers
  layers = NIPS + [FullConnection(num_actions, name='Q', activation_function=None)]

  # Create a network
  return DQN.DeepQNetwork(input_shape, laypaperers, num_actions, network_name=network_name, trainable=trainable)
  


def nature_dqn(input_shape, num_actions, network_name='dqn', trainable=True):
  """
  Create a version of the DQN network from the Nature 
  """

  # Add the output layer to the end of the network layers
  layers = NATURE + [FullConnection(num_actions, name='Q', activation_function=None)]

  # Create a network
  return DQN.DeepQNetwork(input_shape, layers, num_actions, network_name=network_name, trainable=trainable)


def deuling_dqn(input_shape, num_actions, network_name='dqn', trainable=True):
  """
  Create a version of the Deuling DQN network from the ### paper
  """

  # Add value and advantage layers
  value_layers = DEULING[1] + [FullConnection(1, name='value', activation_function=None)]
  advantage_layers = DEULING[2] + [FullConnection(num_actions, name='advantage', activation_function=None)]

  layers = (DEULING[0], value_layers, advantage_layers)

  # Create a network
  return DDQN.DuelingDeepQNetwork(input_shape, layers, num_actions, network_name=network_name, trainable=trainable)

## FIGURE THIS OUT
#def bootstrapped_dqn(input_shape, num_actions, num_heads, network_name='dqn', trainable=True):
#  """
#  Create a version of the Bootstrapped DQN from the Bootstrapped paper
#  """
#
#  # Add the output layer to the end of the head layers
#  head = head_layers + [FullConnection(num_actions, name='Q', activation_function=None)]
#
#  # Create a network
#  return BDQN.BootstrappedDeepQNetwork(input_shape, shared_layers, head, num_actions, num_heads, network_name=network_name, trainable=trainable)