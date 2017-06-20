from models.parts import *

NATURE = ([Convolutional((8,8), 32, name='conv1', stride=4),
           ReLU(name='relu1'),
           Convolutional((4,4), 64, name='conv2', stride=2),
           ReLU(name='relu2'),
           Convolutional((3,3), 64, name='conv3', stride=1),
           ReLU(name='relu3'),
           Flatten(name='flatten'),
           FullConnection(512, name='full1'),
           ReLU(name='relu4'),
           FullConnection(4, name='full2'),
           Linear(name='linear')
          ],
          [Convolutional((8,8), 32, name='conv1_tgt', stride=4),
           ReLU(name='relu1_tgt'),
           Convolutional((4,4), 64, name='conv2_tgt', stride=2),
           ReLU(name='relu2_tgt'),
           Convolutional((3,3), 64, name='conv3_tgt', stride=1),
           ReLU(name='relu3_tgt'),
           Flatten(name='flatten_tgt'),
           FullConnection(512, name='full1_tgt'),
           ReLU(name='relu4_tgt'),
           FullConnection(4, name='full2_tgt'),
           Linear(name='linear_tgt')
          ])

ARXIV  = ([Convolutional((8,8), 16, name='conv1', stride=4),
           ReLU(name='relu1'),
           Convolutional((4,4), 32, name='conv2', stride=2),
           ReLU(name='relu2'),
           Flatten(name='flatten'),
           FullConnection(256, name='full1'),
           ReLU(name='relu4'),
           FullConnection(4, name='full2'),
           Linear(name='linear')
          ],
          [Convolutional((8,8), 16, name='conv1_tgt', stride=4),
           ReLU(name='relu1_tgt'),
           Convolutional((4,4), 32, name='conv2_tgt', stride=2),
           ReLU(name='relu2_tgt'),
           Flatten(name='flatten_tgt'),
           FullConnection(256, name='full1_tgt'),
           ReLU(name='relu4_tgt'),
           FullConnection(4, name='full2_tgt'),
           Linear(name='linear_tgt')
          ])      