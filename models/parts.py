## parts.py
##
## Objects which represent parts (e.g., activation function, weights, etc.) of a neural network.  For
## constructing more complex neural networks

import tensorflow as tf

# Constants for type of pooling layer to use
MAX_POOL = "MAX"
AVG_POOL = "AVG"


def weight_variable(shape, name=None, trainable=True, is_conv=True):
   """
   Create a weight matrix
   """

   if is_conv:
      initializer = tf.contrib.layers.xavier_initializer_conv2d()
   else:
      initializer = tf.contrib.layers.xavier_initializer()

   initializer = tf.truncated_normal_initializer(0, 0.02)
   weights = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable)

   return weights


def bias_variable(shape, name=None, trainable=True):
   """
   Create a bias variable
   """

   initializer = tf.constant_initializer(0.0)
   bias = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer, trainable=trainable)

   return bias


class Convolutional:
   """
   A Convolutional Layer
   """

   def __init__(self, kernel_shape, num_kernels, **kwargs):
      """
      Create a holder for the convolutional layer

      Arguments:
        kernel_size - Size of each kernel
        num_kernels  - Number of kernels (feature maps) to use

      Optional Arguments:
        name    - A name for the layer.  Default is None
        stride  - Stride of the kernel.  Default is 1
        padding - Padding type of the kernel, default is "VALID"
                  One of "SAME" or "VALID"
         activation_function - Activation function to use, default is ReLU
      """

      # Simply hold on to the parameters for now
      self.kernel_shape = kernel_shape
      self.num_kernels = num_kernels

      self.name = kwargs.get("name", None)
      self.stride = kwargs.get("stride", 1)
      self.padding = kwargs.get("padding", "VALID")
      self.activation_function = kwargs.get("activation_function", tf.nn.relu)

      # Placeholder for the weight variable and this layer
      self.weights = None
      self.bias = None
      self.layer = None

      
   def build(self, input_layer, trainable=True):
      """
      Construct the layer in tensorflow
      """

      with tf.variable_scope(self.name):
         # Get the number of input channels
         input_shape = input_layer.get_shape()
         num_input_channels = input_shape[-1].value

         # Create the weights and convolutional layer
         weight_shape = [self.kernel_shape[0], self.kernel_shape[1], num_input_channels, self.num_kernels]

#         if self.name:
#            self.weights = weight_variable(weight_shape, 'W_'+self.name)
#         else:
#            self.weights = weight_variable(weight_shape)

         self.weights = weight_variable(weight_shape, 'weights', trainable)
         self.bias = bias_variable([self.num_kernels], 'bias', trainable)

         self.layer = tf.nn.conv2d(input_layer, self.weights, strides=[1, self.stride, self.stride, 1], padding=self.padding) + self.bias

         if self.activation_function:
            self.layer = self.activation_function(self.layer)

      return self.layer, self.weights, self.bias
      

class FullConnection:
   """
   A Fully Connected Layer
   """

   def __init__(self, output_size, **kwargs):
      """
      Create a fully connected weight matrix

      Arguments:
        output_size - The output size of the weight matrix

      Optional Arguments:
        name      - A name for the layer.  Default is None
      """

      # Simply hold on to the parameters for now
      self.output_size = output_size
      self.name = kwargs.get("name", None)
      self.activation_function = kwargs.get("activation_function", tf.nn.relu)

      # Placeholder for the resulting layer
      self.weights = None
      self.bias = None
      self.layer = None

      
   def build(self, input_layer, trainable=True):
      """
      Construct the layer in tensorflow
      """

      with tf.variable_scope(self.name):

         # Create a weight matrix
         input_size = input_layer.get_shape()[-1].value

         self.weights = weight_variable([input_size, self.output_size], 'weights', trainable, False)
         self.bias = bias_variable([self.output_size], 'bias', trainable)

         # Create the ReLU layer
         self.layer = tf.matmul(input_layer, self.weights) + self.bias

         if self.activation_function:
            self.layer = self.activation_function(self.layer)

      return self.layer, self.weights, self.bias


class Flatten:
   """
   A Flattening Layer
   """

   def __init__(self, **kwargs):
      """
      Create a layer which flattens the input

      Optional Arguments:
        name      - A name for the layer.  Default is None
      """

      # Simply hold on to the parameters for now
      self.name = kwargs.get("name", None)

      # Placeholder for the resulting layer
      self.layer = None

      
   def build(self, input_layer, trainable=True):
      """
      Construct the layer in tensorflow
      """

      with tf.variable_scope(self.name):
         # Determine the size of the input when flattened
         input_layer_shape = input_layer.get_shape()[1:].dims
         flattened_dimension = reduce(lambda x,y: x*y, input_layer_shape, tf.Dimension(1))

         # Create the layer
         self.layer = tf.reshape(input_layer, [-1, flattened_dimension.value])

         return self.layer, None, None
