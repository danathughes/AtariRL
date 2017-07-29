import tensorflow as tf
import numpy as np

class TensorboardMonitor:
	"""
	"""

	def __init__(self, logdir, sess, counter, **kwargs):
		"""
		"""

		self.name = kwargs.get('name', 'summary')
		self.num_actions = kwargs.get('num_actions', 4)
		self.report_frequency = kwargs.get('report_frequency', 10000)

		self.sess = sess
		self.logdir = logdir
		self.counter = counter

		with tf.variable_scope(self.name):
			self.writer = tf.summary.FileWriter(logdir, self.sess.graph)

		self.summaries = {}

		# Add summaries this observer will be responsible for
		self.add_scalar_summary('average_loss', 'episode_summary')
		self.add_scalar_summary('average_Q', 'episode_summary')
		self.add_scalar_summary('average_reward', 'episode_summary')
		self.add_scalar_summary('average_score', 'episode_summary')
		self.add_scalar_summary('max_score', 'episode_summary')
		self.add_scalar_summary('min_score', 'episode_summary')
		self.add_scalar_summary('number_of_games', 'episode_summary')

		self.add_histogram_summary('action_distribution', 'per_game_summary')

		self.reset_stats()


	def reset_stats(self):
		"""
		Reset the statistics collected
		"""

		self.total_loss = 0.0
		self.total_Q = 0.0
		self.total_reward = 0.0
		self.total_score = 0.0
		self.max_score = 0.0
		self.min_score = 1000000.0
		self.num_games = 0.0


	def create_report(self):
		"""
		"""

		self.num_games = max(self.num_games, 1.0)

		report_data = {'average_loss': self.total_loss / self.report_frequency,
		               'average_Q': self.total_Q / self.report_frequency,
		               'average_reward': self.total_reward / self.report_frequency,
		               'average_score': self.total_score / self.num_games,
		               'max_score': self.max_score,
		               'min_score': self.min_score,
		               'number_of_games': self.num_games}

		self.summarize(report_data, self.counter.count)


	def add_scalar_summary(self, summary_name, group=None):
		"""
		Create a placeholder to accept a 
		"""

		with tf.name_scope(group):
			summary_placeholder = tf.placeholder('float32', None, name=summary_name)
			summary_op = tf.summary.scalar(summary_name, summary_placeholder)
			self.summaries[summary_name] = (summary_placeholder, summary_op, group)


	def add_histogram_summary(self, summary_name, group=None):
		"""
		"""

		with tf.name_scope(group):
			summary_placeholder = tf.placeholder('float32', [None,], name=summary_name)
			summary_op = tf.summary.histogram(summary_name, summary_placeholder)
			self.summaries[summary_name] = (summary_placeholder, summary_op, group)


	def summarize(self, data, step):
		"""
		Push the provided data into Tensorboard

		data - a dictionary mapping summary_names to appropriate values
		"""

		# Create a feed dictionary and list of operations to run
		data_feed_dict = {}
		tensorboard_ops = []


		for name in data.keys():
			# Make sure the name is in the list of summaries
			if name in self.summaries.keys():
				placeholder, op, group = self.summaries[name]
				data_feed_dict[placeholder] = data[name]
				tensorboard_ops.append(op)

		# Now run the summary operations and push to the writer
		summary_results = self.sess.run(tensorboard_ops, feed_dict=data_feed_dict)
		for result in summary_results:
			self.writer.add_summary(result, step)


	def start_episode(self, data, **kwargs):
		"""
		"""

		self.actions = []


	def end_episode(self, data, **kwargs):
		"""
		"""

		self.num_games += 1

		if 'score' in data:
			score = data['score']

			self.total_score += score
			self.max_score = max(self.max_score, score)
			self.min_score = min(self.min_score, score)

		self.summarize({'action_distribution': np.array(self.actions)}, self.counter.count)


	def record(self, data, **kwargs):
		"""
		"""

		if data:
			self.summarize(data, self.counter.count)

		if 'training_loss' in data:
			self.total_loss += data['training_loss']
		if 'Q' in data:
			self.total_Q += data['Q']
		if 'reward' in data:
			self.total_reward += data['reward']
		if 'action' in data:
			self.actions.append(data['action'])

		if self.counter.count % self.report_frequency == self.report_frequency - 1:
			self.create_report()
			self.reset_stats()