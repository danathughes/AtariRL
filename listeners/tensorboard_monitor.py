import tensorflow as tf

class TensorboardMonitor:
	"""
	"""

	def __init__(self, logdir, sess, counter, **kwargs):
		"""
		"""

		self.name = kwargs.get('name', 'summary')

		self.sess = sess
		self.logdir = logdir
		self.counter = counter

		with tf.variable_scope(self.name):
			self.writer = tf.summary.FileWriter(logdir, self.sess.graph)

		self.summaries = {}


	def add_scalar_summary(self, summary_name):
		"""
		Create a placeholder to accept a 
		"""

		with tf.variable_scope(self.name):
			summary_placeholder = tf.placeholder('float32', None, name=summary_name)
			summary_op = tf.summary.scalar(summary_name, summary_placeholder)
			self.summaries[summary_name] = (summary_placeholder, summary_op)


	def add_histogram_summary(self, summary_name):
		"""
		"""

		with tf.variable_scope(self.name):
			summary_placeholder = tf.placeholder('float32', [None,], name=summary_name)
			summary_op = tf.summary.histogram(summary_name, summary_placeholder)
			self.summaries[summary_name] = (summary_placeholder, summary_op)


	def summarize(self, data, step):
		"""
		Push the provided data into Tensorboard

		data - a dictionary mapping summary_names to appropriate values
		"""

		# Create a feed dictionary and list of operations to run
		data_feed_dict = {}
		tensorboard_ops = []

		with tf.variable_scope(self.name):

			for name in data.keys():
				# Make sure the name is in the list of summaries
				if name in self.summaries.keys():
					placeholder, op = self.summaries[name]
					data_feed_dict[placeholder] = data[name]
					tensorboard_ops.append(op)

			# Now run the summary operations and push to the writer
			summary_results = self.sess.run(tensorboard_ops, feed_dict=data_feed_dict)
			for result in summary_results:
				self.writer.add_summary(result, step)


	def record(self, data, **kwargs):
		"""
		"""

		if data:
			self.summarize(data, self.counter.count)