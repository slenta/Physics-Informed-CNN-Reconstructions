#file to look at and analyze network logs

import numpy as np
from tensorboard.backend.event_processing import event_multiplexer

LOGDIR = '/home/wchargin/data/scalars_demo/'
RUN_NAME = 'temperature:t0=270,tA=270,kH=0.001'

def main():
  multiplexer = event_multiplexer.EventMultiplexer()
  multiplexer.AddRunsFromDirectory(LOGDIR)
  multiplexer.Reload()

  graph = multiplexer.Graph(RUN_NAME)
  print(len(graph.node))
  print(graph.node[0])

if __name__ == '__main__':
  main()