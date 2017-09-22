##
##
##

import tensorflow as tf

from utils.config_loader import load as load_config
from AtariTrainer import AtariTrainer

# Listeners to perform various bookkeeping tasks
from listeners.checkpoint_recorder import *
from listeners.tensorboard_monitor import *

sess = tf.InteractiveSession()

environment, agent, eval_agent, counter = load_config('example.cfg', sess)
dqn_agent = agent.base_agent

# Create a Tensorboard monitor and populate with the desired summaries
tensorboard_monitor = TensorboardMonitor('./log/breakout/dueling-dqn/', sess, counter)
tensorboard_monitor.add_scalar_summary('score', 'per_game_summary')
tensorboard_monitor.add_scalar_summary('training_loss', 'training_summary')
for i in range(agent.num_actions):
	tensorboard_monitor.add_histogram_summary('Q%d_training' % i, 'training_summary')

checkpoint_monitor = CheckpointRecorder(dqn_agent.dqn, dqn_agent.replay_memory, counter, './checkpoints/breakout/dueling-dqn/', sess)


# Put it all together!
trainer = AtariTrainer(environment, agent, counter, eval_agent=eval_agent)

trainer.add_listener(checkpoint_monitor)
trainer.add_listener(tensorboard_monitor)
dqn_agent.add_listener(tensorboard_monitor)

sess.run(tf.global_variables_initializer())


def run():
	cur_episode = 0
	num_frames = 0
	while counter.count < 50000000:
		score = trainer.learn_episode()

		tensorboard_monitor.record({'score': score})

		elapsed_frames = counter.count - num_frames
		num_frames = counter.count
		print "Episode %d:  Total Score = %d\t# Frames = %d\tTotal Frames = %d\tEpsilon: %f" % (cur_episode, score, elapsed_frames, num_frames, agent.epsilon)
		cur_episode += 1

	print
	print "Done Training.  Playing..."

if __name__ == '__main__':
	run()

