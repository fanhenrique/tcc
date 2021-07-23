import matplotlib
import networkx as nx
import networkx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

import argparse
import logging


DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'


def readFile(file, n):

	peer_nodes = []
	monitor_nodes = [] 

	with open(file, 'r') as file:
		
		file.readline() #ignora cabeçalho 
		
		# for line in file:
		for i in range(0, n):
			
			# window, time, ip_port, peer_id, monitor_id, monitor = line.split(' ')			
			window, time, ip_port, peer_id, monitor_id, monitor = file.readline().split(' ')
			
			peer_nodes.append('p'+peer_id)
			
			monitor_nodes.append('m'+monitor_id)

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

def betweenness_centrality(graph, nodes):
	
	all_betweenness = nx.betweenness_centrality(graph)

	betweenness = {}

	for b in all_betweenness:
		if b in nodes:
			betweenness[b] = all_betweenness[b]

	betweenness = list(betweenness.items())
	betweenness.sort(key=lambda tup: tup[1], reverse=True)

	return betweenness

def show_rating(rating, size):

	if size <= 0:
		for r in rating:
			print(r[0], r[1])
	else:
		for r in rating[0:size]:
			print(r[0], r[1])
		
		print('.\n.\n.')

		for r in rating[len(rating)-size:len(rating)]:
			print(r[0], r[1])

def adjacent_nodes(graph, node):
	nodes = []
	for i in nx.edge_boundary(graph, node):
		nodes.append(i[1])
	return nodes

def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)
	help_msg='Escolha o algoritmo 0 - degree_centrality 1 - degree 2 - eigenvector_centrality 3 - betweenness_centrality'
	parser.add_argument('--algorithm', '-a', choices=[0,1,2,3], help=help_msg, required=True, type=int)
	parser.add_argument('--sizeshow', '-s', help='size show', default=0, type=int)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt=TIME_FORMAT, level=args.log)
	

	peer_nodes, monitor_nodes = readFile(args.file, 10000)

	monitor_list = node_list(monitor_nodes)
	peer_list = node_list(peer_nodes)
	
	graph = create_graph(peer_nodes, 'peer', 'red',  monitor_nodes, 'monitor', 'blue')


	if args.algorithm == 0:
		monitors_degree_centrality = degree_centrality(graph, monitor_list)
		print('degree_centrality\n')
		show_rating(monitors_degree_centrality, args.sizeshow)

	elif args.algorithm == 1:
		monitors_degree = degree(graph, monitor_list)
		print('\ndegree\n')
		show_rating(monitors_degree, args.sizeshow)

	elif args.algorithm == 2:
		monitors_eigenvector_centrality = eigenvector_centrality(graph, monitor_list)
		print('\neigenvector_centrality\n')
		show_rating(monitors_eigenvector_centrality, args.sizeshow)

	elif args.algorithm == 3:
		monitors_betweenness_centrality = betweenness_centrality(graph, monitor_list)
		print('\nbetweenness_centrality\n')
		show_rating(monitors_betweenness_centrality, args.sizeshow)


	# print(bipartite.degrees(graph, monitor_list))
	# print(bipartite.density(graph, peer_list))

	show_graph(graph)
	
if __name__ == '__main__':
	main()