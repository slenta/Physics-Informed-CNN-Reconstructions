#file to look at and analyze network logs

import numpy as np
from tensorboard.backend.event_processing import event_multiplexer

LOGDIR = '../Asi_maskiert/logs/default/'
RUN_NAME = 'events.out.tfevents.1652028193.mg207'

def main():
  multiplexer = event_multiplexer.EventMultiplexer()
  multiplexer.AddRunsFromDirectory(LOGDIR)
  multiplexer.Reload()

  graph = multiplexer.Graph(RUN_NAME)
  print(len(graph.node))
  print(graph.node[0])

if __name__ == '__main__':
  main()