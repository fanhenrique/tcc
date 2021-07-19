import matplotlib
import networkx as nx
import networkx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

import argparse
import logging


DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

# class Trace(object):
	
# 	def __init__(self, window, time, peer1, peer2, monitor1, monitor2):
# 		self.window = window
# 		self.time = time
# 		self.peer1 = peer1
# 		self.peer2 = peer2
# 		self.monitor1 = monitor1
# 		self.monitor2 = monitor2

# 	def printTrace(self):
# 		print(self.window, self.time, self.peer1, self.peer2, self.monitor1, self.monitor2)

def readFile(file, n):

	# traces = []
	peer_nodes = []
	monitor_nodes = [] 

	with open(file, 'r') as file:
		
		file.readline() #ignora cabeçalho 
		
		# for line in file:
		for i in range(0, n):
			
			# window, time, peer1, peer2, monitor1, monitor2 = line.split(' ')			
			window, time, peer1, peer2, monitor1, monitor2 = file.readline().split(' ')
			
			peer_nodes.append('p'+peer1)
			
			monitor_nodes.append('m'+monitor1)
			
			# print(window, time, peer1, peer2, monitor1, monitor2)

			# traces.append(Trace(window, time, peer1, peer2, monitor1, monitor2))
			# traces[i].printTrace()

	return peer_nodes, monitor_nodes

def create_graph(nodes1, type_nodes1, color_nodes1, nodes2, type_nodes2, color_nodes2):

	graph = nx.Graph()

	graph.add_nodes_from(nodes1, type_nodes=type_nodes1, color_nodes=color_nodes1)
	graph.add_nodes_from(nodes2, type_nodes=type_nodes2, color_nodes=color_nodes2)

	graph.add_edges_from(list(zip(nodes1, nodes2)))

	return graph

def show_graph(graph):

	colors = [u[1] for u in graph.nodes(data='color_nodes')]
	nx.draw(graph, with_labels=True, node_color=colors)

	plt.show()

def node_list(nodes):
	return list(dict.fromkeys(nodes))

def degree(graph, nodes):

	all_degrees = nx.degree(graph)
	
	degrees = []

	for d in all_degrees:
		if d[0] in nodes:
			degrees.append(d)

	degrees.sort(key=lambda x: x[1], reverse=True)
	
	return degrees


def degree_centrality(graph, nodes):
	
	all_centralities = nx.degree_centrality(graph)

	centralities = {}
		
	for c in all_centralities:
		if c in nodes:
			centralities[c] = all_centralities[c]

	centralities = list(centralities.items())
	centralities.sort(key=lambda tup: tup[1], reverse=True)

	return centralities

def eigenvector_centrality(graph, nodes):

	all_eigenvectors = nx.eigenvector_centrality(graph)

	eigenvectors = {}

	for e in all_eigenvectors:
		if e in nodes:
			eigenvectors[e] = all_eigenvectors[e]

	eigenvectors = list(eigenvectors.items())
	eigenvectors.sort(key=lambda tup: tup[1], reverse=True)

	return eigenvectors

def adjacent_nodes(graph, node):
	nodes = []
	for i in nx.edge_boundary(graph, node):
		nodes.append(i[1])
	return nodes

def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt=TIME_FORMAT, level=args.log)
	

	peer_nodes, monitor_nodes = readFile(args.file, 200)

	monitor_list = node_list(monitor_nodes)
	peer_list = node_list(peer_nodes)
	
	graph = create_graph(peer_nodes, 'peer', 'red',  monitor_nodes, 'monitor', 'blue')

	monitors_degrees_centralities = degree_centrality(graph, monitor_list)
	print('grau de centralidade dos monitores\n\n', monitors_degrees_centralities)

	monitors_degrees = degree(graph, monitor_list)
	print('grau dos monitores\n\n', monitors_degrees)

	monitors_vector_contralities = eigenvector_centrality(graph, monitor_list)
	print('vector centralities\n\n', monitors_vector_contralities)



	print('-------------------')
	
	print(graph.edges())

	print('-------------------')


	print(adjacent_nodes(graph, 'p9'))

	# print(bipartite.degrees(graph, monitor_list))
	# print(bipartite.density(graph, peer_list))

	

	


	



	show_graph(graph)
	
if __name__ == '__main__':
	main()