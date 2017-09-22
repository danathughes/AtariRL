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


	def step(self):
		"""
		Increment the counter
		"""

		self.count += 1
