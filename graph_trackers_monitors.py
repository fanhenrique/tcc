import matplotlib
import networkx as nx
import networkx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

from collections import Counter


import argparse
import logging


DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'


def readFile(file, n):

	windows = []
	trakers = []
	monitors = []

	with open(file, 'r') as file:

		file.readline() #ignora cabeçalho 

		if n == 0:
			for line in file:

				line_split = line.split()			
				
				windows.append(float(line_split[0]))

				trakers.append(line_split[1].split("'")[1])

				monitors.append(line_split[16].split("'")[1])	
				
		else:
			for i in range(0, n):

				line_split = file.readline().split()			
					
				windows.append(float(line_split[0]))

				trakers.append(line_split[1].split("'")[1])

				monitors.append(line_split[16].split("'")[1])

	
	return windows, trakers, monitors

def create_graph(nodes):

	graph = nx.Graph()


	for n in nodes:
		graph.add_nodes_from(n[0], type_nodes=n[1], color_nodes=n[2])
		

	edges = list(zip(nodes[0][0], nodes[1][0]))
	# print(edges)
	
	weight = dict(Counter(edges))
	# print(weight)

	weighted_edges = []

	for e in edges:
		weighted_edges.append((e[0], e[1], weight[(e[0], e[1])]))

	# print(weighted_edges)

	graph.add_weighted_edges_from(weighted_edges)

	return graph

def show_graph(graph):

	colors = [u[1] for u in graph.nodes(data='color_nodes')]
	# nx.draw(graph, with_labels=True, node_color=colors)
	
	pos = nx.spring_layout(graph)
	weights = nx.get_edge_attributes(graph, "weight")

	nx.draw_networkx(graph, pos, with_labels=True, node_color=colors)
	nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)

	plt.show()


def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)
	help_msg='Escolha o algoritmo 0 - degree_centrality 1 - degree 2 - eigenvector_centrality 3 - betweenness_centrality'
	parser.add_argument('--algorithm', '-a', choices=[0,1,2,3], help=help_msg, required=True, type=int)
	parser.add_argument('--sizeshow', '-s', help='head e tail monitores no arquivo de saida', default=0, type=int)
	
	# REMOVER DEPOIS
	parser.add_argument('--numberlines', '-n', help='number lines', default=0, type=int) 

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt=TIME_FORMAT, level=args.log)
	

	windows, trakers, monitors =  readFile(args.file, args.numberlines)

	nodes = []
	nodes.append((trakers, 'traker', 'red'))
	nodes.append((monitors, 'monitor', 'blue'))

	graph = create_graph(nodes)


	# monitor_list = node_list(monitor_nodes)
	# peer_list = node_list(peer_nodes)
	

	# algorithm(args.algorithm, args.sizeshow, graph, monitor_list)

	show_graph(graph)


if __name__ == '__main__':
	main()