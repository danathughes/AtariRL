import models.DeepQNetwork as DQN
import models.DuelingDeepQNetwork as DDQN
import models.BootstrappedDeepQNetwork as BDQN

from models.parts import FullConnection

def create_dqn_builder(network_layers):
	"""
	"""

	def builder(input_shape, num_actions, sess, network_name='dqn', trainable=True):
		"""
		"""

		# Add the output layer to the end of the network layers
		layers = network_layers + [FullConnection(num_actions, name='Q', activation_function=None)]

		# Create a network
		return DQN.DeepQNetwork(input_shape, layers, num_actions, sess, network_name=network_name, trainable=trainable)
	
	return builder



def create_dueling_dqn_builder(network_layers):
	"""
	"""

	def builder(input_shape, num_actions, sess, network_name='dqn', trainable=True):
		"""
		"""

		# Add value and advantage layers
		value_layers = network_layers[1] + [FullConnection(1, name='value', activation_function=None)]
		advantage_layers = network_layers[2] + [FullConnection(num_actions, name='advantage', activation_function=None)]

		layers = (network_layers[0], value_layers, advantage_layers)

		# Create a network
		return DDQN.DuelingDeepQNetwork(input_shape, layers, num_actions, sess, network_name=network_name, trainable=trainable)

	return builder


def create_bootstrapped_dqn_builder(shared_layers, head_layers, num_heads):
	"""
	"""

	def builder(input_shape, num_actions, sess, network_name='dqn', trainable=True):
		"""
		"""

		# Add the output layer to the end of the head layers
		head = head_layers + [FullConnection(num_actions, name='Q', activation_function=None)]

		# Create a network
		return BDQN.BootstrappedDeepQNetwork(input_shape, shared_layers, head, num_actions, num_heads, sess, network_name=network_name, trainable=trainable)

	return builder