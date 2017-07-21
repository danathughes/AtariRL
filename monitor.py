import tensorflow as tf

class TensorboardMonitor:
	"""
	"""

	def __init__(self, logdir, sess, **kwargs):
		"""
		"""

		self.name = kwargs.get('name', 'summary')

		self.sess = sess
		self.logdir = logdir

		with tf.variable_scope(self.name):
			self.writer = tf.summary.FileWriter(logdir, self.sess.graph)

		self.summaries = {}


	def add_dqn_summary(self, dqn, **kwargs):
		"""
		"""

		q_summary_name = kwargs.get('q_summary_name', 'Q%d')

		num_actions = dqn.num_actions

		q_averages = tf.reduce_mean(dqn.output, 0)
		q_averages_summary = []

		# Add a histogram summary for each Q action
		for i in range(num_actions):
			q_averages_summary.append(tf.summary.histogram(q_summary_name % i, q_averages[i]))
		self.summaries['q_summary'] = tf.summary.merge(q_averages_summary)


	def summarize(self, summary_names, step, feed_dict):
		"""
		"""

#		summaries = [self.summaries[name] for name in summary_names]

		# Collect all the summaries to summarize
#		for name in summary_names:
		summary = self.sess.run(self.summaries['q_summary'], feed_dict = feed_dict)

		self.writer.add_summary(summary, step)