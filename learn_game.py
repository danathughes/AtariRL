##
##
##

import tensorflow as tf

from utils.config_loader import load as load_config
from AtariTrainer import AtariTrainer

sess = tf.InteractiveSession()

environment, agent, eval_agent, counter, checkpoint, tensorboard = load_config('example.cfg', sess)
dqn_agent = agent.base_agent

# Put it all together!
trainer = AtariTrainer(environment, agent, counter, eval_agent=eval_agent)
trainer.add_listener(tensorboard_monitor)

# Start Tensorflow
sess.run(tf.global_variables_initializer())

def run():
	cur_episode = 0
	num_frames = counter.count
	while counter.count < 50000000:
		score = trainer.learn_episode()

		tensorboard.record({'score': score})

		elapsed_frames = counter.count - num_frames
		num_frames = counter.count
		print "Episode %d:  Total Score = %d\t# Frames = %d\tTotal Frames = %d\tEpsilon: %f" % (cur_episode, score, elapsed_frames, num_frames, agent.epsilon)
		cur_episode += 1

	print
	print "Done Training.  Playing..."

if __name__ == '__main__':
	run()

