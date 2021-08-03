import networkx as nx
import math
import matplotlib.pyplot as plt
import stellargraph as sg

from collections import Counter

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

hash_count_tracker = 1
hash_table_tracker = {}

hash_count_monitor = 1
hash_table_monitor = {}

def readFile(file, n):

	epoch = []
	trakers = []
	monitors = []

	with open(file, 'r') as file:

		file.readline() #ignora cabeçalho 

		if n == 0:
			
			for line in file:

				line_split = line.split()			
				
				try:
					epoch.append(float(line_split[0]))
				except:
					continue
				try:
					trakers.append(line_split[1].split("'")[1])
				except:
					epoch.pop()
					continue
				try:
					monitors.append(line_split[16].split("'")[1])	
				except:
					epoch.pop()
					trakers.pop()
					continue
		
		## REMOVER ELSE DEPOIS (apenas pra rodar com um arquivo menor)
		else:
			for i in range(0, n):

				line_split = file.readline().split()			
				
				try:	
					epoch.append(float(line_split[0]))
				except:
					continue
				try:
					trakers.append(line_split[1].split("'")[1])
				except:
					epoch.pop()
					continue
				try:
					monitors.append(line_split[16].split("'")[1])
				except:
					epoch.pop()
					trakers.pop()
					# print(i, line_split)
					continue

	
	return epoch, trakers, monitors

def create_graph(nodes_list):

	graph = nx.Graph()

	# Vertices
	for nodes in nodes_list:
		graph.add_nodes_from(nodes[0], color_nodes=nodes[2])

	# Tipo de vertices 	
	dict_nodes = {}	
	for nodes in nodes_list:
		for n in nodes[0]:
			dict_nodes[n] = nodes[1]
	nx.set_node_attributes(graph, dict_nodes, 'label')	

	# Arestas
	edges = list(zip(nodes_list[0][0], nodes_list[1][0]))
	
	# Peso das arestas
	weight = dict(Counter(edges))
	
	# Arestas com peso
	weighted_edges = []
	for e in edges:
		weighted_edges.append((e[0], e[1], weight[(e[0], e[1])]))

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

def my_hash_tracker(tracker_str):

	global hash_table_tracker
	global hash_count_tracker

	my_hash_tracker_value = hash_table_tracker.get(tracker_str)

	if my_hash_tracker_value is None:
		my_hash_tracker_value = hash_count_tracker
		hash_table_tracker[tracker_str] = my_hash_tracker_value
		hash_count_tracker += 1

	return my_hash_tracker_value


def my_hash_monitor(monitor_str):

	global hash_table_monitor
	global hash_count_monitor

	my_hash_monitor_value = hash_table_monitor.get(monitor_str)

	if my_hash_monitor_value is None:
		my_hash_monitor_value = hash_count_monitor
		hash_table_monitor[monitor_str] = my_hash_monitor_value
		hash_count_monitor += 1

	return my_hash_monitor_value


def cal_windows(epoch):
	
	time_min = []
	windows = []

	for e in epoch:
		time_min.append(((e - epoch[0]) / 60))

	for t in time_min:
		windows.append(math.trunc(t / 15))

	# for i in range(0, len(windows)):	
	# 	print(windows[i], format(time_min[i], '.2f'))

	return time_min, windows

def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)
	# help_msg='Escolha o algoritmo 0 - degree_centrality 1 - degree 2 - eigenvector_centrality 3 - betweenness_centrality'
	# parser.add_argument('--algorithm', '-a', choices=[0,1,2,3], help=help_msg, required=True, type=int)
	parser.add_argument('--sizeshow', '-s', help='head e tail monitores no arquivo de saida', default=0, type=int)
	
	# REMOVER DEPOIS (apenas pra rodar com um arquivo menor)
	parser.add_argument('--numberlines', '-n', help='number lines', default=0, type=int) 

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt=TIME_FORMAT, level=args.log)
	
	global hash_table_tracker
	global hash_count_tracker

	global hash_table_monitor
	global hash_count_monitor

	epochs, trakers, monitors =  readFile(args.file, args.numberlines)


	print(len(epochs), len(trakers), len(monitors))
	print(epochs[len(epochs)-1], trakers[len(trakers)-1], monitors[len(monitors)-1])


	time_min, windows = cal_windows(epochs) 	

	
	# Label pra os vertices
	traker_nodes_labels = []
	for t in trakers:
		traker_nodes_labels.append('t' + str(my_hash_tracker(t)))
	monitor_nodes_labels = []
	for m in monitors:
		monitor_nodes_labels.append('m' + str(my_hash_monitor(m)))


	# Label, tipo, cor dos vertices	
	nodes_list = []
	nodes_list.append((traker_nodes_labels, 'traker', 'red'))
	nodes_list.append((monitor_nodes_labels, 'monitor', 'blue'))


	graph = create_graph(nodes_list)

	graph_stellar = sg.StellarGraph.from_networkx(graph)

	print(graph_stellar.info())

	show_graph(graph)


if __name__ == '__main__':
	main()