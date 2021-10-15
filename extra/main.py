'''
cria os graficos e tira algumas metricas
'''

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

	peer_nodes = []
	monitor_nodes = [] 

	with open(file, 'r') as file:
		
		file.readline() #ignora cabeçalho 
			
		## IF ELSE temporario pra rodar grafo menor
		if n == 0:
			for line in file:

				window, time, ip_port, peer_id, monitor_id, monitor = line.split(' ')			
				
				peer_nodes.append('p'+peer_id)
				
				monitor_nodes.append('m'+monitor_id)
		else:
			for i in range(0, n):
			
				window, time, ip_port, peer_id, monitor_id, monitor = file.readline().split(' ')
				
				peer_nodes.append('p'+peer_id)
				
				monitor_nodes.append('m'+monitor_id)

	return peer_nodes, monitor_nodes

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

### Cria grafo sem pesso nas arestas
def create_graph2(nodes1, type_nodes1, color_nodes1, nodes2, type_nodes2, color_nodes2):

	graph = nx.Graph()

	graph.add_nodes_from(nodes1, type_nodes=type_nodes1, color_nodes=color_nodes1)
	graph.add_nodes_from(nodes2, type_nodes=type_nodes2, color_nodes=color_nodes2)

	edges = list(zip(nodes1, nodes2))
	# print(edges)

	graph.add_edges_from(edges)

	return graph


def show_graph(graph):

	colors = [u[1] for u in graph.nodes(data='color_nodes')]
	# nx.draw(graph, with_labels=True, node_color=colors)
	
	pos = nx.spring_layout(graph)
	weights = nx.get_edge_attributes(graph, "weight")

	nx.draw_networkx(graph, pos, with_labels=True, node_color=colors)
	nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)

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
			print('{} {:.4f}'.format(r[0], r[1]))
	else:
		for r in rating[0:size]:
			print('{} {:.4f}'.format(r[0], r[1]))
		
		print('.\n.\n.')

		for r in rating[len(rating)-size:len(rating)]:
			print('{} {:.4f}'.format(r[0], r[1]))

def adjacent_nodes(graph, node):
	nodes = []
	for i in nx.edge_boundary(graph, node):
		nodes.append(i[1])
	return nodes


def algorithm(algorithm, sizeshow,graph, monitor_list):

	if algorithm == 0:
		monitors_degree_centrality = degree_centrality(graph, monitor_list)
		print('degree_centrality\n')
		show_rating(monitors_degree_centrality, sizeshow)

	elif algorithm == 1:
		monitors_degree = degree(graph, monitor_list)
		print('\ndegree\n')
		show_rating(monitors_degree, sizeshow)

	elif algorithm == 2:
		monitors_eigenvector_centrality = eigenvector_centrality(graph, monitor_list)
		print('\neigenvector_centrality\n')
		show_rating(monitors_eigenvector_centrality, sizeshow)

	elif algorithm == 3:
		monitors_betweenness_centrality = betweenness_centrality(graph, monitor_list)
		print('\nbetweenness_centrality\n')
		show_rating(monitors_betweenness_centrality, sizeshow)

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
	

	peer_nodes, monitor_nodes = readFile(args.file, args.numberlines)

	monitor_list = node_list(monitor_nodes)
	peer_list = node_list(peer_nodes)
	
	nodes = []
	nodes.append((peer_nodes, 'peer', 'red'))
	nodes.append((monitor_nodes, 'monitor', 'blue'))

	graph = create_graph(nodes)	

	algorithm(args.algorithm, args.sizeshow, graph, monitor_list)

	show_graph(graph)

	


if __name__ == '__main__':
	main()