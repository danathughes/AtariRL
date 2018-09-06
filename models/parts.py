## parts.py
##
## Objects which represent parts (e.g., activation function, weights, etc.) of a neural network.  For
## constructing more complex neural networks

import tensorflow as tf
import numpy as np

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



class PrimaryCaps_EM:
   """
   A Primary Capsule Layer.  See jhui.github.io blog for details.  uses EM routing.
   """

   def __init__(self, kernel_shape, pose_shape, num_output_capsules, **kwargs):
      """
      Create a holder for the primary capsule layer

      Arguments:
         kernel_shape - shape of the kernels
         pose_shape - the shape of the pose of each capsule
         num_output_capsules - how many capsules to generate

      Optional Arguments:
         name  - A name for the layer
         strides - Stride of the kernel
         padding - "SAME" or "VALID"
         activation_function - ReLU
      """

      # Simply hold on to the parameters for now
      self.kernel_shape = kernel_shape
      self.num_capsules = num_output_capsules
      self.pose_shape = pose_shape

      self.name = kwargs.get("name", None)
      self.stride = kwargs.get("stride", 1)
      self.padding = kwargs.get("padding", "SAME")
      self.activation_function = kwargs.get("activation_function", tf.nn.relu)

      # Placeholder for the weights for this layer
      self.pose_weights = None
      self.activation_weights = None


   def build(self, input_layer, trainable=True):
      """
      Construct the layer in tensorflow
      """

      with tf.variable_scope(self.name):
         # Get the number of input channels
         input_shape = input_layer.get_shape()
         num_input_channels = input_shape[-1].value

         # Create the weights for the pose and activation
         pose_weight_shape = [self.kernel_shape[0], self.kernel_shape[1], num_input_channels, self.pose_shape[0]*self.pose_shape[1]*self.num_capsules]
         activation_weight_shape = [self.kernel_shape[0], self.kernel_shape[1], num_input_channels, self.num_capsules]

         self.pose_weights = weight_variable(pose_weight_shape, 'w_pose_'+self.name, trainable)
         self.activation_weights = weight_variable(activation_weight_shape, 'w_activation_'+self.name, trainable)

         # Calculate the poses and activations - reshape pose to (-1, W, H, POSE_W, POSE_H, NUM_CAPSULES)
         self.poses = tf.nn.conv2d(input_layer, self.pose_weights, strides=[1, self.stride, self.stride, 1], padding=self.padding)
         self.poses = tf.reshape(self.poses, shape=[-1, input_shape[-3], input_shape[-2], self.num_capsules, self.pose_shape[0], self.pose_shape[1]])

         self.activations = tf.nn.conv2d(input_layer, self.activation_weights, strides=[1, self.stride, self.stride, 1], padding=self.padding)
         self.activations = tf.sigmoid(self.activations)

         print self.poses.get_shape()
         print self.activations.get_shape()

      return self.poses, self.activations, self.pose_weights, self.activation_weights


class ConvCaps_EM:
   """
   A Convolutional Capsule layer, using EM routing
   """

   def __init__(self, kernel_shape, num_output_capsules, batch_size, **kwargs):
      """
      """

      # Simply hold the parameters for now
      self.kernel_shape = kernel_shape
      self.num_capsules = num_output_capsules
      self.batch_size = batch_size

      self.name = kwargs.get("name", None)
      self.stride = kwargs.get("stride", 1)
      self.padding = kwargs.get("padding", "SAME")
      self.num_em_steps = kwargs.get("num_em_steps", 3)
      self.epsilon = kwargs.get("epsilon", 1e-8)

      # Placeholder for the weights for this layer
      self.pose_weights = None
      self.beta_v = None
      self.beta_a = None

      # Placeholder for activations of this layer
      self.votes = None
      self.routing = None


   # BLACK BOX helper functions, modified from jhui.github.io
   def _tile(self, input_layer):
      """
      Perform tiling and convolution to prepare the input pose and activation to the
      correct spatial dimension for voting and EM-routing.

      input_layer: a pose layer with shape (N, W, H, C, POSE_W, POSE_H) or
                   an activation layer with shape (N, W, H, C)
      return:  a tensor whose dimensions are (N, W, H, K, O)
               K = the flattened kernel shape (kernel_width x kernel_height)
               O = the flattened pose and/or activation (pose_width x pose_height x num_input_capsules)
      """

      # Extract relevent sizes from the input
      input_shape = input_layer.get_shape()

      input_width = input_shape[1].value
      input_height = input_shape[2].value
      num_input_capsules = input_shape[3].value
      kernel_width, kernel_height = self.kernel_shape

      if len(input_shape) > 5:   # Is this a pose tensor?
         output_channel_size = num_input_capsules*input_shape[4].value*input_shape[5].value
      else:                      # An activation tensor
         output_channel_size = num_input_capsules

      # Flatten the input so that it is (?, W, H, OUT_C)
      input_flat = tf.reshape(input_layer, shape=[-1, input_width, input_height, output_channel_size])

      # Create the tile filter operation
      tile_filter = np.zeros(shape=[kernel_width, kernel_height, output_channel_size, kernel_width*kernel_height], dtype=np.float32)

      for i in range(kernel_width):
         for j in range(kernel_height):
            tile_filter[i,j,:,i*kernel_height + j] = 1.0

      tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

      # Perform the tiling
      output = tf.nn.depthwise_conv2d(input_flat, tile_filter_op, strides=[1, self.stride, self.stride, 1], padding='VALID')

      # Get the width and height of the output
      output_shape = output.get_shape()
      output_width = output_shape[1].value
      output_height = output_shape[2].value

      # Put the right numbers in the right places
      output = tf.reshape(output, shape=[-1, output_width, output_height, num_input_capsules, kernel_width*kernel_height])
      output = tf.transpose(output, perm=[0,1,2,4,3])

      return output


   def _e_step(self, mean_h, stdev_h, activations, votes):
      """
      Perform an expectation step, i.e., the routing assignment

      mean_h:  (N, OW, OH, 1, OC, PW*PH)
      stdev_h: (N, OW, OH, 1, OC, PW*PH)
      activations:   (N, OW, OH, 1, OC, 1)
      votes:   (N, OW, OH, KW x KH x IC, OC, PW*PH)

      return:  routing
      """

      # We are calculating the log probability for P
      o_p0 = -tf.reduce_sum(tf.square(votes - mean_h) / (2*tf.square(stdev_h)), axis=-1, keep_dims=True)
      o_p1 = -tf.reduce_sum(tf.log(stdev_h + self.epsilon), axis=-1, keep_dims=True)

      # o_p is the probability density of the h-th component of the vote from i to j
      # (N, OW, OH, 1, OC, PWxPH)
      o_p = o_p0 + o_p1


      # The routing is the softmax of the probability distributions
      zz = tf.log(activations + self.epsilon) + o_p
      routing_assignments = tf.nn.softmax(zz, dim=len(zz.get_shape().as_list())-2)

      return routing_assignments


   def _m_step(self, routing_assignments, votes, activations, beta_v, beta_a, temperature):
      """
      routing_assignments: (KW x KH x IC, OC, 1)
      votes:               (N, OH, OW, KW x KH x IC, OC, PW x PH)
      activations:         (N, OH, OW, KW x KH x IC, 1, 1)
      beta_v:              (1, 1, 1, 1, OC, 1)
      beta_a:              (1, 1, 1, 1, OC, 1)
      temperature:         lambda

      return               out_mean, out_stdev, out_activation
      """

      routing_prime = routing_assignments * activations

      # Sum over all input capulse
      routing_prime_sum = tf.reduce_sum(routing_prime, axis=-3, keep_dims=True, name='routing_prime_sum')

      # Calculate mean and std_dev for all h
      mean_h = tf.reduce_sum(routing_prime * votes, axis=-3, keep_dims=True) / routing_prime_sum
      stdev_h = tf.sqrt(tf.reduce_sum(routing_prime * tf.square(votes - mean_h), axis=-3, keep_dims=True) / routing_prime_sum)

      # Calculate cost
      cost_h = (beta_v + tf.log(stdev_h + self.epsilon)) * routing_prime_sum

      # The relative variance between each channel determines which one should activate
      cost_sum = tf.reduce_sum(cost_h, axis=-1, keep_dims=True)
      cost_mean = tf.reduce_mean(cost_sum, axis=-2, keep_dims=True)
      cost_stdev = tf.sqrt(tf.reduce_sum(tf.square(cost_sum - cost_mean), axis=-2, keep_dims=True)/ cost_sum.get_shape().as_list()[-2])

      cost_h = beta_a + (cost_mean - cost_sum) / (cost_stdev + self.epsilon)

      # Activation - sigmoid(lambda * (beta_a - sum(cost)))
      out_activation = tf.sigmoid(temperature * cost_h)

      return mean_h, stdev_h, out_activation


   def _routing(self, votes, activations):
      """
      votes:         (N, OW, OH, KW x KH x IC, OC, PW x PH)
      activations:   (N, OW, OH, KW x KH x IC)

      return:        pose, activation
      """

      votes_shape = votes.get_shape().as_list()

      with tf.variable_scope('em_routing'):
         # Create the initial routing assignment as evenly distributed

         routing = tf.constant(1.0 / votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32)

         # Expand the dimensions of the activations
         activations = activations[..., tf.newaxis, tf.newaxis]

         # Similarly for beta_v and beta_a
         beta_v = self.beta_v[..., tf.newaxis, :, tf.newaxis]
         beta_a = self.beta_a[..., tf.newaxis, :, tf.newaxis]

         # Temperature schedule
         temp_min = 1.0
         temp_max = min(self.num_em_steps, 3.0)

         for step in range(self.num_em_steps):
            with tf.variable_scope("iteration_%d" % step):

               temp = temp_min + (temp_max - temp_min) * step / max(1.0, self.num_em_steps - 1.0)

               mean_h, stdev_h, out_activations = self._m_step(routing, votes, activations, beta_v, beta_a, temperature=temp)

               if step < self.num_em_steps - 1:
                  routing = self._e_step(mean_h, stdev_h, out_activations, votes)

      # Now that the EM routing is done, calculate the output pose and activations
      out_poses = tf.squeeze(mean_h, axis=-3)
      out_activations = tf.squeeze(out_activations, axis=[-3,-1])

      return out_poses, out_activations, routing


   def _transform(self, _input, output_capsule_size, size, pose_width, pose_height, trainable=True):
      """
      """

      print _input.get_shape()
      print output_capsule_size
      print size
      print pose_width
      print pose_height

      num_input_capsules = _input.get_shape()[1].value
      output = tf.reshape(_input, shape=[size, num_input_capsules, 1, pose_width, pose_height])

      weight_shape = [1, num_input_capsules, output_capsule_size, pose_width, pose_height]
      self.pose_weights = weight_variable(weight_shape, 'W_'+self.name, trainable=trainable, is_conv=False)

      w = tf.tile(self.pose_weights, [size, 1, 1, 1, 1])
      output = tf.tile(output, [1,1,output_capsule_size,1,1])

      votes = tf.matmul(output, w)
      votes = tf.reshape(votes, [size, num_input_capsules, output_capsule_size, pose_width*pose_height])

      return votes


   def build(self, pose_layer, activation_layer, trainable=True):
      """
      Construct the convolution capsule layer

      pose_layer: a primary or convolution capsule layer with shape (N, W, H, C, POSE_W, POSE_H)
      activation_layer: (N, W, H, C)
      """

      # Some useful numbers in a more legible format
      pose_shape = pose_layer.get_shape()
      kernel_width, kernel_height = self.kernel_shape
      num_input_capsules = pose_shape[3].value
      pose_width = pose_shape[4].value
      pose_height = pose_shape[5].value

      print pose_layer.get_shape()


      with tf.variable_scope(self.name):

         # Tile the activations and input poses
         # The input capsules' pose matrices are tiled to the spatial dimension of the output, allowing multiplication
         # later with the transformation matricies to generate votes
         # The input capsules' activation matricies are tiled for EM routing

         # Tile the pose matrix so that it can be multiplied with the transformation weights to generate the votes
         input_poses = self._tile(pose_layer)

         # Tile the activations for use with EM routing
         input_activations = self._tile(activation_layer)
         spatial_width = input_activations.get_shape()[1].value
         spatial_height = input_activations.get_shape()[2].value

         # Reshape the tensors for later operations
         input_poses = tf.reshape(input_poses, shape=[-1, kernel_width*kernel_height*num_input_capsules, pose_width*pose_height])
         input_activations = tf.reshape(input_activations, shape=[-1, spatial_width, spatial_height, kernel_width*kernel_height*num_input_capsules])

         # Compute the votes
         with tf.variable_scope('votes'):
            # Create the transformation matrix (weights)
#            weight_shape = [1, num_input_capsules, self.num_capsules, pose_width, pose_height]
#            self.pose_weights = weight_variable(weight_shape, 'W_'+self.name, trainable=trainable, is_conv=False)

            # size of the multiplication
            vote_size = self.batch_size*spatial_width*spatial_height

            # Tile the weight matrix and poses by the batch size
#            w = tf.tile(self.pose_weights, [vote_size, 1,1,1,1])

#            reshaped_poses = tf.reshape(input_poses, shape=[vote_size, kernel_width*kernel_height*num_input_capsules, 1, pose_width, pose_height])
#            tiled_poses = tf.tile(reshaped_poses, [1, 1, self.num_capsules, 1, 1])

            # Calculate the votes
#            votes = tf.matmul(tiled_poses, w)
#            self.votes = tf.reshape(votes, shape=[self.batch_size, spatial_width, spatial_height, num_input_capsules, self.num_capsules, pose_width*pose_height])

            votes = self._transform(input_poses, self.num_capsules, vote_size, pose_width, pose_height)
            vote_shape = votes.get_shape()
            self.votes = tf.reshape(votes, shape=[self.batch_size, spatial_width, spatial_height, vote_shape[-3].value, vote_shape[-2].value, vote_shape[-1].value]) #self.num_capsules, pose_width*pose_height])


            print self.name + " votes shape: " + str(self.votes.get_shape())

         # Compute the routing
         with tf.variable_scope('routing'):
            # Create beta variables for each capsule
            self.beta_v = weight_variable([1,1,1,self.num_capsules], 'beta_v_'+self.name, trainable=trainable, is_conv=False)
            self.beta_a = weight_variable([1,1,1,self.num_capsules], 'beta_a_'+self.name, trainable=trainable, is_conv=False)

            # Use EM routing to compute the pose and activations
            poses, self.activations, routing = self._routing(self.votes, input_activations)

            self.routing = routing

         # Reshape the pose matrix
         pose_shape = poses.get_shape()
         self.poses = tf.reshape(poses, [pose_shape[0], pose_shape[1], pose_shape[2], pose_shape[3], pose_width, pose_height])

         print self.name + " pose shape: " + str(self.poses.get_shape())
         print self.name + " activations shape: " + str(self.activations.get_shape())
         print self.name + " routing shape: " + str(self.routing.get_shape())

         return self.poses, self.activations, self.routing, self.pose_weights, self.beta_v, self.beta_a


class ClassCaps_EM:
   """
   A Convolutional Capsule layer, using EM routing
   """

   def __init__(self, num_classes, batch_size, **kwargs):
      """
      num_classes
      batch_size
      """

      # Simply hold the parameters for now
      self.num_classes = num_classes
      self.batch_size = batch_size

      self.name = kwargs.get("name", None)
      self.num_em_steps = kwargs.get("num_em_steps", 3)
      self.epsilon = kwargs.get("epsilon", 1e-8)

      # Placeholder for the weights for this layer
      self.pose_weights = None
      self.beta_v = None
      self.beta_a = None

      # Placeholder for activations of this layer
      self.votes = None
      self.routing = None


   # BLACK BOX helper functions, modified from jhui.github.io
   def _tile(self, input_layer):
      """
      Perform tiling and convolution to prepare the input pose and activation to the
      correct spatial dimension for voting and EM-routing.

      input_layer: a pose layer with shape (N, W, H, C, POSE_W, POSE_H) or
                   an activation layer with shape (N, W, H, C)
      return:  a tensor whose dimensions are (N, W, H, K, O)
               K = the flattened kernel shape (kernel_width x kernel_height)
               O = the flattened pose and/or activation (pose_width x pose_height x num_input_capsules)
      """

      # Extract relevent sizes from the input
      input_shape = input_layer.get_shape()

      input_width = input_shape[1].value
      input_height = input_shape[2].value
      num_input_capsules = input_shape[3].value
      kernel_width, kernel_height = self.kernel_shape

      if len(input_shape) > 5:   # Is this a pose tensor?
         output_channel_size = num_input_capsules*input_shape[4].value*input_shape[5].value
      else:                      # An activation tensor
         output_channel_size = num_input_capsules

      # Flatten the input so that it is (?, W, H, OUT_C)
      input_flat = tf.reshape(input_layer, shape=[-1, input_width, input_height, output_channel_size])

      # Create the tile filter operation
      tile_filter = np.zeros(shape=[kernel_width, kernel_height, num_input_capsules, kernel_width*kernel_height], dtype=np.float32)

      for i in range(kernel_width):
         for j in range(kernel_height):
            tile_filter[i,j,:,i*kernel_height + j] = 1.0

      tile_filter_op = tf.constant(tile_filter, dtype=tf.float32)

      # Perform the tiling
      output = tf.nn.depthwise_conv2d(input_flat, tile_filter_op, strides=[1, self.stride, self.stride, 1], padding='VALID')

      # Get the width and height of the output
      output_shape = output.get_shape()
      output_width = output_shape[1].value
      output_height = output_shape[2].value

      # Put the right numbers in the right places
      output = tf.reshape(output, shape=[-1, output_width, output_height, num_input_capsules, kernel_width*kernel_height])
      output = tf.transpose(output, perm=[0,1,2,4,3])

      return output


   def _e_step(self, mean_h, stdev_h, activations, votes):
      """
      Perform an expectation step, i.e., the routing assignment

      mean_h:  (N, OW, OH, 1, OC, PW*PH)
      stdev_h: (N, OW, OH, 1, OC, PW*PH)
      activations:   (N, OW, OH, 1, OC, 1)
      votes:   (N, OW, OH, KW x KH x IC, OC, PW*PH)

      return:  routing
      """

      # We are calculating the log probability for P
      o_p0 = -tf.reduce_sum(tf.square(votes - mean_h) / (2*tf.square(stdev_h)), axis=-1, keep_dims=True)
      o_p1 = -tf.reduce_sum(tf.log(stdev_h + self.epsilon), axis=-1, keep_dims=True)

      # o_p is the probability density of the h-th component of the vote from i to j
      # (N, OW, OH, 1, OC, PWxPH)
      o_p = o_p0 + o_p1


      # The routing is the softmax of the probability distributions
      zz = tf.log(activations + self.epsilon) + o_p
      routing_assignments = tf.nn.softmax(zz, dim=len(zz.get_shape().as_list())-2)

      return routing_assignments


   def _m_step(self, routing_assignments, votes, activations, beta_v, beta_a, temperature):
      """
      routing_assignments: (KW x KH x IC, OC, 1)
      votes:               (N, OH, OW, KW x KH x IC, OC, PW x PH)
      activations:         (N, OH, OW, KW x KH x IC, 1, 1)
      beta_v:              (1, 1, 1, 1, OC, 1)
      beta_a:              (1, 1, 1, 1, OC, 1)
      temperature:         lambda

      return               out_mean, out_stdev, out_activation
      """

      routing_prime = routing_assignments * activations

      # Sum over all input capulse
      routing_prime_sum = tf.reduce_sum(routing_prime, axis=-3, keep_dims=True, name='routing_prime_sum')

      # Calculate mean and std_dev for all h
      mean_h = tf.reduce_sum(routing_prime * votes, axis=-3, keep_dims=True) / routing_prime_sum
      stdev_h = tf.sqrt(tf.reduce_sum(routing_prime * tf.square(votes - mean_h), axis=-3, keep_dims=True) / routing_prime_sum)

      # Calculate cost
      cost_h = (beta_v + tf.log(stdev_h + self.epsilon)) * routing_prime_sum

      # The relative variance between each channel determines which one should activate
      cost_sum = tf.reduce_sum(cost_h, axis=-1, keep_dims=True)
      cost_mean = tf.reduce_mean(cost_sum, axis=-2, keep_dims=True)
      cost_stdev = tf.sqrt(tf.reduce_sum(tf.square(cost_sum - cost_mean), axis=-2, keep_dims=True)/ cost_sum.get_shape().as_list()[-2])

      cost_h = beta_a + (cost_mean - cost_sum) / (cost_stdev + self.epsilon)

      # Activation - sigmoid(lambda * (beta_a - sum(cost)))
      out_activation = tf.sigmoid(temperature * cost_h)

      return mean_h, stdev_h, out_activation


   def _routing(self, votes, activations):
      """
      votes:         (N, OW, OH, KW x KH x IC, OC, PW x PH)
      activations:   (N, OW, OH, KW x KH x IC)

      return:        pose, activation
      """

      votes_shape = votes.get_shape().as_list()

      with tf.variable_scope('em_routing'):
         # Create the initial routing assignment as evenly distributed

         routing = tf.constant(1.0 / votes_shape[-2], shape=votes_shape[-3:-1] + [1], dtype=tf.float32)

         # Expand the dimensions of the activations
         activations = activations[..., tf.newaxis, tf.newaxis]

         # Similarly for beta_v and beta_a
         beta_v = self.beta_v[..., tf.newaxis, :, tf.newaxis]
         beta_a = self.beta_a[..., tf.newaxis, :, tf.newaxis]

         # Temperature schedule
         temp_min = 1.0
         temp_max = min(self.num_em_steps, 3.0)

         for step in range(self.num_em_steps):
            with tf.variable_scope("iteration_%d" % step):

               temp = temp_min + (temp_max - temp_min) * step / max(1.0, self.num_em_steps - 1.0)

               mean_h, stdev_h, out_activations = self._m_step(routing, votes, activations, beta_v, beta_a, temperature=temp)

               if step < self.num_em_steps - 1:
                  routing = self._e_step(mean_h, stdev_h, out_activations, votes)

      # Now that the EM routing is done, calculate the output pose and activations
      out_poses = tf.squeeze(mean_h, axis=-3)
      out_activations = tf.squeeze(out_activations, axis=[-3,-1])

      return out_poses, out_activations, routing


   def _transform(self, _input, output_capsule_size, size, pose_width, pose_height, trainable=True):
      """
      """

      print _input.get_shape()
      print output_capsule_size
      print size
      print pose_width
      print pose_height

      num_input_capsules = _input.get_shape()[1].value
      output = tf.reshape(_input, shape=[size, num_input_capsules, 1, pose_width, pose_height])

      weight_shape = [1, num_input_capsules, output_capsule_size, pose_width, pose_height]
      self.pose_weights = weight_variable(weight_shape, 'W_'+self.name, trainable=trainable, is_conv=False)

      w = tf.tile(self.pose_weights, [size, 1, 1, 1, 1])
      output = tf.tile(output, [1,1,output_capsule_size,1,1])

      votes = tf.matmul(output, w)
      votes = tf.reshape(votes, [size, num_input_capsules, output_capsule_size, pose_width*pose_height])

      return votes


   def _coord_addition(self, votes, width, height):
      """
      """

      pose_size = votes.get_shape()[-1].value

      coordinate_offset_hh = tf.reshape((tf.range(height, dtype=tf.float32) + 0.5) / height, [1, 1, height, 1, 1])
      coordinate_offset_h0 = tf.constant(0.0, shape=[1, 1, height, 1, 1], dtype=tf.float32)
      coordinate_offset_h = tf.stack([coordinate_offset_h0, coordinate_offset_hh] + [coordinate_offset_h0 for _ in range(pose_size-2)], axis=-1)

      coordinate_offset_ww = tf.reshape((tf.range(width, dtype=tf.float32) + 0.5) / width, [1, width, 1, 1, 1])
      coordinate_offset_w0 = tf.constant(0.0, shape=[1,width,1,1,1], dtype=tf.float32)
      coordinate_offset_w = tf.stack([coordinate_offset_ww, coordinate_offset_w0] + [coordinate_offset_w0 for _ in range(pose_size-2)], axis=-1)

      return votes + coordinate_offset_h + coordinate_offset_w


   def build(self, pose_layer, activation_layer, trainable=True):
      """
      Construct the convolution capsule layer

      pose_layer: a primary or convolution capsule layer with shape (N, W, H, C, POSE_W, POSE_H)
      activation_layer: (N, W, H, C)
      """

      # Some useful numbers in a more legible format
      pose_shape = pose_layer.get_shape()
      spatial_width = pose_shape[1].value
      spatial_height = pose_shape[2].value
      num_input_capsules = pose_shape[3].value
      pose_width = pose_shape[4].value
      pose_height = pose_shape[5].value

      with tf.variable_scope(self.name):

         # Reshape the tensors for later operations
         input_poses = tf.reshape(pose_layer, shape=[self.batch_size*spatial_width*spatial_height, num_input_capsules, pose_width*pose_height])

         # Compute the votes
         with tf.variable_scope('votes'):
            # Create the transformation matrix (weights)
#            weight_shape = [1, num_input_capsules, self.num_classes, pose_width, pose_height]
#            self.pose_weights = weight_variable(weight_shape, 'W_'+self.name, trainable=trainable, is_conv=False)

            # size of the multiplication
            vote_size = self.batch_size*spatial_width*spatial_height

            # Tile the weight matrix and poses by the batch size
#            w = tf.tile(self.pose_weights, [self.batch_size, 1,1,1,1])
#            reshaped_poses = tf.reshape(input_poses, shape=[vote_size, num_input_capsules, 1, pose_width, pose_height])
#            tiled_poses = tf.tile(reshaped_poses, [1, 1, self.num_classes, 1, 1])

            # Calculate the votes
#            votes = tf.matmul(tiled_poses, w)
#            votes = tf.reshape(votes, shape=[self.batch_size, spatial_width, spatial_height, num_input_capsules, self.num_classes, pose_width*pose_height])

            votes = self._transform(input_poses, self.num_classes, vote_size, pose_width, pose_height)
            self.votes = tf.reshape(votes, shape=[self.batch_size, spatial_width, spatial_height, num_input_capsules, self.num_classes, pose_width*pose_height])

            self.votes = self._coord_addition(self.votes, spatial_width, spatial_height)

            print self.name + " votes shape: " + str(self.votes.get_shape())


         # Compute the routing
         with tf.variable_scope('routing'):
            # Create beta variables for each capsule
            self.beta_v = weight_variable([1,1,1,self.num_classes], 'beta_v_'+self.name, trainable=trainable, is_conv=False)
            self.beta_a = weight_variable([1,1,1,self.num_classes], 'beta_a_'+self.name, trainable=trainable, is_conv=False)

            votes_shape = self.votes.get_shape()
            votes = tf.reshape(self.votes, shape = [self.batch_size, votes_shape[1]*votes_shape[2]*votes_shape[3], votes_shape[4], votes_shape[5]])

            input_activations = tf.reshape(activation_layer, shape=[self.batch_size, votes_shape[1]*votes_shape[2]*votes_shape[3]])

            # Use EM routing to compute the pose and activations
            poses, self.activations, self.routing = self._routing(votes, input_activations)

         # Reshape the pose matrix
         pose_shape = poses.get_shape()
         self.poses = tf.reshape(poses, [self.batch_size, self.num_classes, pose_width, pose_height])

         self.activations = tf.squeeze(self.activations)
         self.routing = tf.squeeze(self.routing)

         print self.name + " pose shape: " + str(self.poses.get_shape())
         print self.name + " activations shape: " + str(self.activations.get_shape())
         print self.name + " routing shape: " + str(self.routing.get_shape())

         return self.poses, self.activations, self.routing, self.pose_weights, self.beta_v, self.beta_a


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
