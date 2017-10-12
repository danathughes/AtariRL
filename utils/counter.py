## counter.py
##
## Simple counter to allow univeral maintainance of a common value


class Counter:
	"""
	Simple class to maintain a shared counter between objects
	"""

	def __init__(self, initial_count=0):
		"""
		"""

		self.count = initial_count

		# Timing at which to perform particular actions
		self.hooks = {}


	def add_hook(self, action, timing, start_time = 0):
		"""
		Perform the action at a certain timing

		action 		- callback function, takes no arguments
		timing 		- the number of frames to skip between each call
		start_time  - an optional time which to perform the first call
		"""

		# Add the hook and start time to the list of actions at this timing
		self.hooks[timing] = self.hooks.get(timing, []) + [(action, start_time)]


	def step(self):
		"""
		Increment the counter
		"""

		self.count += 1

		# Are there callbacks to be made?
		for time in self.hooks.keys():
			if self.count % time == 0:
				for actions in self.hooks[time]:
					action, start_time = actions
					if self.count >= start_time:
						action()


