##
##
##

import tensorflow as tf

from utils.config_loader import load as load_config
from AtariTrainer import AtariTrainer

sess = tf.InteractiveSession()

environment, agent, eval_agent, counter, checkpoint, tensorboard = load_config('example.cfg')
dqn_agent = agent.base_agent

# Put it all together!
trainer = AtariTrainer(environment, agent, counter, eval_agent=eval_agent)
trainer.add_listener(tensorboard)

# Start Tensorflow
sess.run(tf.global_variables_initializer())

# Can restore things from the checkpoint, if desired
if counter.count > 0:
	checkpoint.restore_memory(3000000)
	checkpoint.restore_tensorflow(counter.count)

def run():
	cur_episode = 0
	num_frames = counter.count
	while counter.count <= 50000000:
		score = trainer.learn_episode()

		tensorboard.record({'score': score})

		elapsed_frames = counter.count - num_frames
		num_frames = counter.count
		print "Episode %d:  Total Score = %d\t# Frames = %d\tTotal Frames = %d\tEpsilon: %f" % (cur_episode, score, elapsed_frames, num_frames, agent.epsilon)
		cur_episode += 1

	print
	print "Done Training.  Playing..."

if __name__ == '__main__':
	try:
		run()
	except KeyboardInterrupt:
		# User pressed Control-C, save the current model and memory
		checkpoint.save_memory()
		checkpoint.save_tensorflow()
		checkpoint.save_dqn()
