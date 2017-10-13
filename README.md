# AtariRL

Implementations of various Deep-Q Networks for learning Atari games.  The repository contains the following:

1.  Implementation of Deep-Q Networks, Dueling Deep-Q Networks and Bootstrapped Deep-Q Networks.  Networks are implemented using TensorFlow.

2.  Learning agents implementing the DQN algorithm (using a target DQN), the double DQN algorithm, and the bootstrapped DQN algorithm.

3.  Experience Replay Memory, Priority Replay Memory, Rank-based Priority Replay Memory, and Bootstrapped Replay Memory.

4.  Environments based on Atari Learning Environment (ALE) and OpenAI Gym.

5.  Utilities for loading from a configuration file, storing checkpoints, and visualizing using TensorBoard.

## Achieved Implementations

Currently, the approaches used in the following papers can be implemented:

* Human-level Control through Deep Reinforcement Learning [2].

* Deep Reinforcement Learning with Double Q-Learning [3].

* Dueling Network Architectures for Deep Reinforcement Learning [4].

* Prioritized Experience Replay [5].

* Deep Exploration via Bootstrapped DQN [6].

## Current Work

Current papers / architectures being implemented:

* Asynchronous version of DQN, from [7].  This requires some fundamental additions / changes to the software architecture, to allow for multi-threaded rollouts and agents that do not implement replay memory.

* An agent which does not implement a target DQN, to mimic the original NIPS workshop paper [1].

* An agent which only uses Neural Fitted Q Iteration (i.e., no replay memory or target DQN).  This would be for comparison purposes, and would also allow for an implementation of the original Riedmiller paper on this topic [9].

## Future Plans

Other implementations which still need to be implemented (not sure when, though):

* DRQL and DARQL - this primarily should only require slight modifications to a DQN agent, and implementing recurrent and attention networks.

* Other asynchronous algorithms (e.g., A3C!) from [7].

* The implementations from the C51 paper.

* The UCT approached used by Honglak Lee's group.

* All the approaches used to handle Montezuma's revenge.


## Prerequisites

The software and versions used to implement the code are

1.  Python 2.7

2.  Tensorflow 1.2

3.  Numpy 1.13

4.  pyGame 1.9.1

## References

[1].  V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra and M. Riedmiller, "Playing Atari with Deep Reinforcement Learning," NIPS Deep Learning Workshop 2013, arXiv preprint arXiv:1312.5602, 2013.  

[2].  V. Mnih, K. Kavukcuoglu, D. Silver, A.A. Rusu, J. Veness, M.G. Bellemare, A. Graves, M. Riedmiller, A.K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg and D. Hassabis, "Human-level Control through Deep Reinforcement Learning," Nature, vol. 518, pp. 529--533, 2015.  

[3].  H. van Hasselt, A. Guez and D. Silver, "Deep Reinforcement Learning with Double Q-Learning," arXiv preprint arXiv:1509.06461, 2015.

[4]. Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot and N. de Freitas, "Deuling Network Architectures for Deep Reinforcement Learning," arXiv preprint arXiv:1511.06581, 2016.

[5]. T. Schaul, J. Quan, I. Antonoglou and D. Silver, "Prioritized Experience Replay," arXiv preprint arXiv:1511.05952, 2016.

[6]. I. Osband, C. Blundell, A. Pritzel and B. van Roy, "Deep Exploration via Bootstrapped DQN," arXiv preprint arXiv:1602.04621, 2016.

[7]. V. Mnih, A. Puigdomenech, M. Mizra, A. Graves, T.P. Lillicrap, T. Harley, D. Liver, K. Kavukcuoglu, "Asynchronous Methods for Deep Reinforcement Learning," arXiv prepreint arXiv:1602.01783, 2016.

[8]. S. Gu, T. Lillicrap, I. Sutskever, S. Levine, "Continuous Deep Q-Learning with Model-Based Acceleration," arXiv preprint arXiv:1603.00748, 2016.


[1]: https://arxiv.org/abs/1312.5602

[2]: http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html

[3]: https://arxiv.org/abs/1509.06461

[4]: https://arxiv.org/abs/1511.06581

[5]: https://arxiv.org/abs/1511.05952

[6]: https://arxiv.org/abs/1602.04621

[7]: https://arxiv.org/abs/1602.01783

[8]: https://arxiv.org/abs/1603.00748

## Other Implementations

