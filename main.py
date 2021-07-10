import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 

import argparse
import logging


DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

class Trace(object):
	
	def __init__(self, window, time, peer1, peer2, monitor1, monitor2):
		self.window = window
		self.time = time
		self.peer1 = peer1
		self.peer2 = peer2
		self.monitor1 = monitor1
		self.monitor2 = monitor2

	def printTrace(self):
		print(self.window, self.time, self.peer1, self.peer2, self.monitor1, self.monitor2)

def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', type=str)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt=TIME_FORMAT, level=args.log)

	traces = []

	nodes_peers = []
	nodes_monitors = [] 

	with open('data/data', 'r') as file:
		
		file.readline() #ignora cabeçalho 
		
		# for line in file:
		for i in range(0, 1000000):
			
			# window, time, peer1, peer2, monitor1, monitor2 = line.split(' ')			
			window, time, peer1, peer2, monitor1, monitor2 = file.readline().split(' ')
			
			nodes_peers.append(peer1)
			
			nodes_monitors.append(monitor1)
			
			# print(window, time, peer1, peer2, monitor1, monitor2)

			# traces.append(Trace(window, time, peer1, peer2, monitor1, monitor2))
			# traces[i].printTrace()


	graph = nx.Graph()
	pos = nx.spring_layout(graph)

	graph.add_nodes_from(nodes_peers, nodetype='red')
	graph.add_nodes_from(nodes_monitors, nodetype='blue')

	print(graph.nodes())

	graph.add_edges_from(list(zip(nodes_peers, nodes_monitors)))

	print(graph.edges())

	colors = [u[1] for u in graph.nodes(data="nodetype")]
	nx.draw(graph, with_labels=True, node_color=colors)

	plt.show()
	
if __name__ == '__main__':
	main()