import math
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt

# import stellargraph as sg

import utils



from collections import Counter

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

TRACKER = 'TRACKER'
MONITOR = 'MONITOR'
PEER = 'PEER'


# def readFile(file):
# 	epochs, trakers, monitors = [], [], []
# 	with open(file, 'r') as file:
# 		file.readline() #ignora cabeçalho 
# 		for line in file:
# 			line_split = line.split()			
# 			try:
# 				epochs.append(float(line_split[0]))
# 			except:
# 				print(line)
# 				continue
# 			try:
# 				trakers.append(line_split[1].split("'")[1])
# 			except:
# 				print(line)
# 				epochs.pop()
# 				continue
# 			try:
# 				monitors.append(line_split[16].split("'")[1])	
# 			except:
# 				print(line)
# 				epochs.pop()
# 				trakers.pop()
# 				continue
# 	return epochs, trakers, monitors


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

def create_graph_peer_weights(nodes_list, peer_lists):

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

	# # cria as restas
	edges = list(zip(nodes_list[0][0], nodes_list[1][0]))
	print(edges)

	# # conta os pesos das arestas
	ed= Counter(edges)
	print(ed)
	weight = dict(ed)
	print(weight)

	# # arestas com peso
	weighted_edges = []
	# for e in edges:
	# 	weighted_edges.append((e[0], e[1], weight[(e[0], e[1])]))


	# weight = dict(edges, peer_lists)

	# for i in range(len(nodes_list[0])):
	# 	print((nodes_list[0][i], nodes_list[1][i], len(peer_lists[i])))	
	# 	weighted_edges.append((nodes_list[0][i], nodes_list[1][i], len(peer_lists[i])))

	# graph.add_weighted_edges_from(weighted_edges)

	return graph



def show_graph(graph):

	colors = [u[1] for u in graph.nodes(data='color_nodes')]
	# nx.draw(graph, with_labels=True, node_color=colors)
	
	pos = nx.spring_layout(graph)
	weights = nx.get_edge_attributes(graph, "weight")

	nx.draw_networkx(graph, pos, with_labels=True, node_color=colors)
	nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)

	plt.show()

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
	
	parser.add_argument('--numberwindows', '-w', help='number windows', default=1, type=int) 
	parser.add_argument('--numberedges', '-e', help='number edges', default=0, type=int) 

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)

	init()

	logging.info('reading file ...')
	epochs, trakers, monitors, peer_lists =  utils.read_file(args.file)

	logging.info('calculating windows ...')
	time_min, windows, windows_index_range = utils.cal_windows(epochs, args.numberwindows)


	# Label pra os vertices
	traker_labels = []
	for t in trakers:
		traker_labels.append('T'+str(utils.my_hash_tracker(t)))
		# traker_labels.append(str(utils.my_hash_tracker(t)))
	monitor_labels = []
	for m in monitors:
		monitor_labels.append('M'+str(utils.my_hash_monitor(m)))
		# monitor_labels.append(str(utils.my_hash_monitor(m)))
	peer_lists_labels = []
	for l in peer_lists:
		peer_list_labels = []
		for p in l:
			peer_list_labels.append('P'+str(utils.my_hash_peer(p)))
			# peer_list_labels.append(str(utils.my_hash_peer(p)))
		peer_lists_labels.append(peer_list_labels)

	graphs = []
	# graphs_stellar = []
	logging.info('creating graphs ...')
	for wir in windows_index_range:
		
		traker_nodes = traker_labels[wir[0]:wir[1]]
		monitor_nodes = monitor_labels[wir[0]:wir[1]]
		peer_list_nodes = peer_lists_labels[wir[0]:wir[1]]
	
		if args.numberedges == 0:
			num_edges = len(traker_nodes)
		else:
			num_edges = args.numberedges


		nodes_list = []
		# Label, tipo, cor dos vertices	
		nodes_list.append((traker_nodes[0:num_edges], TRACKER, 'red'))
		nodes_list.append((monitor_nodes[0:num_edges], MONITOR, 'blue'))
		# nodes_list.append((peer_list_nodes[0:num_edges], PEER, 'black'))

		# graph = create_graph(nodes_list)
		graph = create_graph_peer_weights(nodes_list, peer_list_nodes)

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