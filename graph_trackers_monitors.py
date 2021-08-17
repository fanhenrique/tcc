import math
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt

# import stellargraph as sg

from collections import Counter

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'
WINDOWS_LEN = 15

TRACKER = 'TRACKER'
MONITOR = 'MONITOR'

hash_count_tracker = 1
hash_table_tracker = {}

hash_count_monitor = 1
hash_table_monitor = {}

def readFile(file, n):
	epochs, trakers, monitors = [], [], []
	with open(file, 'r') as file:
		file.readline() #ignora cabeçalho 
		for line in file:
			line_split = line.split()			
			try:
				epochs.append(float(line_split[0]))
			except:
				print(line)
				continue
			try:
				trakers.append(line_split[1].split("'")[1])
			except:
				print(line)
				epochs.pop()
				continue
			try:
				monitors.append(line_split[16].split("'")[1])	
			except:
				print(line)
				epochs.pop()
				trakers.pop()
				continue
	return epochs, trakers, monitors

def create_graph(nodes_list):

	graph = nx.Graph()

	# vertices
	for nodes in nodes_list:
		graph.add_nodes_from(nodes[0], color_nodes=nodes[2])

	# label dos vertices 	
	dict_nodes = {}	
	for nodes in nodes_list:
		for n in nodes[0]:
			dict_nodes[n] = nodes[1]
	nx.set_node_attributes(graph, dict_nodes, 'label')	

	# cria as restas
	edges = list(zip(nodes_list[0][0], nodes_list[1][0]))
	
	# conta os pesos das arestas
	weight = dict(Counter(edges))
	
	# arestas com peso
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


def cal_windows(epoch, number_windows):
	
	time_min = []
	windows = []


	w_previous = 0
	counter_windows = 0

	for e in epoch:		
		if counter_windows < number_windows:

			tm = (e - epoch[0]) / 60	
			w = math.trunc(tm / WINDOWS_LEN)

			if w_previous != w:
				counter_windows+=1	
		
			time_min.append(tm)
			windows.append(w)
			w_previous = w
		else:
			break
	
	windows_index_range = []
	break0 = 0
	for i in range(len(windows)-1):
		if windows[i] != windows[i+1]:
			break1 = i
			windows_index_range.append((break0, break1))
			break0 = break1+1


	return time_min, windows, windows_index_range	

def save_graph(graph, g):
	# print(graph.nodes.data())
	with open('out/graph_'+str(g)+'.txt', 'w') as file:
		for edge in graph.edges.data():
			file.write(edge[0] + ' ' + str(edge[2]['weight']) + ' ' + edge[1] + '\n')

def init():

	try:
		shutil.rmtree('./out')
	except FileNotFoundError:
		pass
	try:
		os.mkdir('./out')
	except FileExistsError:
		pass


def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)
	# help_msg='Escolha o algoritmo 0 - degree_centrality 1 - degree 2 - eigenvector_centrality 3 - betweenness_centrality'
	# parser.add_argument('--algorithm', '-a', choices=[0,1,2,3], help=help_msg, required=True, type=int)
	# parser.add_argument('--sizeshow', '-s', help='head e tail monitores no arquivo de saida', default=0, type=int)
	
	# REMOVER DEPOIS (apenas pra rodar com um arquivo menor)
	parser.add_argument('--numberlines', '-n', help='number lines', default=0, type=int) 
	parser.add_argument('--numberwindows', '-w', help='number windows', default=0, type=int) 

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)
	
	global hash_table_tracker
	global hash_count_tracker

	global hash_table_monitor
	global hash_count_monitor

	init()

	logging.info('reading file ...')
	epochs, trakers, monitors =  readFile(args.file, args.numberlines)

	logging.info('calculating windows ...')
	time_min, windows, windows_index_range = cal_windows(epochs, args.numberwindows)


	# Label pra os vertices
	traker_labels = []
	for t in trakers:
		traker_labels.append(TRACKER+'_'+str(my_hash_tracker(t)))
	monitor_labels = []
	for m in monitors:
		monitor_labels.append(MONITOR+'_'+str(my_hash_monitor(m)))

	graphs = []
	graphs_stellar = []
	logging.info('creating graphs ...')
	for wir in windows_index_range:
		# Label, tipo, cor dos vertices	
		traker_nodes = traker_labels[wir[0]:wir[1]]
		monitor_nodes = monitor_labels[wir[0]:wir[1]]
	
		nodes_list = []
		nodes_list.append((traker_nodes, TRACKER, 'red'))
		nodes_list.append((monitor_nodes, MONITOR, 'blue'))

		graph = create_graph(nodes_list)

		# graph_stellar = sg.StellarGraph.from_networkx(graph)
		# print(graph_stellar.info())
		# graphs_stellar.append(graph_stellar)

		graphs.append(graph)

		save_graph(graph, len(graphs))

		show_graph(graph)

	logging.info(str(len(graphs)) + ' graphs in directory: out/')

# linha com problema 
# 531964 ['1546539658.000000', "['udp://exodus.desync.com:6969/ann#planetlab1.pop-pa.rnp.br',", "'UTC_time',", "'UTC_epoch',", "'infohash',", "'tracker',", "'interval_sec',", "'minInterval_sec',", "'downloads',", "'leechers',", "'seeders',", "'size_of_peerlist',", "'monitor_name',", "'peerlist\\n']"]

if __name__ == '__main__':
	main()



# PALAVRAS CHAVES:
# predict 
# distributed systems 
# fails 
# monitoring
# P2P
# Bittorrent