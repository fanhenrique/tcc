import math
import os
import shutil
import matplotlib.pyplot as plt

import networkx as nx
# import stellargraph as sg

from collections import Counter

import argparse
import logging

#my imports
import utils

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

TRACKER = 'TRACKER'
MONITOR = 'MONITOR'
PEER = 'PEER'

global SHOWPEERS

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
	weights = dict(Counter(edges))
	
	# arestas com peso
	weighted_edges = []
	for e in edges:
		weighted_edges.append((e[0], e[1], weights[(e[0], e[1])]))

	graph.add_weighted_edges_from(weighted_edges)

	return graph

def create_graph_peer_weights2(nodes_list, peer_lists):

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
	# print(edges, len(edges))


	# conta pesos das arestas
	weights = []
	for peer in peer_lists:
		weights.append(len(peer))
	# print(weights, len(weights))


	ew = zip(edges, weights)

	c = Counter()
	for k, v in ew:
		c[k] += v
	

	weighted_edges = []
	for e in edges:
		weighted_edges.append((e[0], e[1], c[(e[0], e[1])]))

	graph.add_weighted_edges_from(weighted_edges)

	return graph

def create_graph_peer_weights(monitors, trackers, peer_lists):

	graph = nx.Graph()

	# vertices
	graph.add_nodes_from(monitors[0], color_nodes=monitors[2])
	graph.add_nodes_from(trackers[0], color_nodes=trackers[2])
	
	if SHOWPEERS: 
		for nodes in peer_lists[0]:
			graph.add_nodes_from(nodes, color_nodes=peer_lists[2])

		
	# USADO NO STELLARGRAPH
	# label dos vertices 	
	# dict_nodes = {}	
	# for nodes in nodes_list[0:-1]:
	# 	for n in nodes[0]:
	# 		dict_nodes[n] = nodes[1]
	# nx.set_node_attributes(graph, dict_nodes, 'label')


	# arestas trackers peers
	if SHOWPEERS:
		edges_tp = []	
		for i in range(len(peer_lists[0])):
			for peer in peer_lists[0][i]:
				edges_tp.append((trackers[0][i], peer))
		# print(edges_tp)
		
		graph.add_edges_from(edges_tp)



	# arestas monitors trackers
	edges_mt = list(zip(monitors[0], trackers[0]))
	# print(edges_mt, len(edges_mt))

	# conta pesos das arestas monitors trackers
	weights_mt = []
	for peer_list in peer_lists[0]:
		weights_mt.append(len(peer_list))
	# print(weights_mt, len(weights_mt))
	
	c = Counter()
	for k, v in zip(edges_mt, weights_mt):
		c[k] += v

	weighted_edges = []
	for e in edges_mt:
		weighted_edges.append((e[0], e[1], c[(e[0], e[1])]))

	graph.add_weighted_edges_from(weighted_edges)


	return graph



def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)
	parser.add_argument('--showpeers', '-p', help='show peers', action='store_true')
	

	parser.add_argument('--numberwindows', '-w', help='number windows', default=1, type=int)
	# parser.add_argument('--numberedges', '-e', help='number edges', default=0, type=int) 

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)

	global SHOWPEERS
	SHOWPEERS = args.showpeers

	utils.init()

	logging.info('reading file ...')
	epochs, trackers, monitors, peer_lists =  utils.read_file(args.file)

	logging.info('calculating windows ...')
	time_min, windows, windows_index_range = utils.cal_windows(epochs, args.numberwindows)

	logging.info('renaming entities ...')
	# Label pra os vertices
	tracker_labels = []
	for t in trackers:
		tracker_labels.append('T'+str(utils.my_hash_tracker(t)))
		# traker_labels.append(str(utils.my_hash_tracker(t)))
	monitor_labels = []
	for m in monitors:
		monitor_labels.append('M'+str(utils.my_hash_monitor(m)))
		# monitor_labels.append(str(utils.my_hash_monitor(m)))
	peer_lists_labels = []
	for l in peer_lists:
		pl_labels = []
		for p in l:
			pl_labels.append('P'+str(utils.my_hash_peer(p)))
			# pl_labels.append(str(utils.my_hash_peer(p)))
		peer_lists_labels.append(pl_labels)

	graphs = []
	# graphs_stellar = []
	logging.info('creating graphs ...')
	for wir in windows_index_range:
		
		m = (monitor_labels[wir[0]:wir[1]], MONITOR, 'blue')
		t = (tracker_labels[wir[0]:wir[1]], TRACKER, 'red')		
		pl = (peer_lists_labels[wir[0]:wir[1]], PEER, 'green')

		graph = create_graph_peer_weights(m, t, pl)

		graphs.append(graph)

		utils.save_graph_txt(graph, len(graphs))

		# utils.show_graph(graph)

		utils.save_graph_fig(graph, len(graphs))


		# traker_nodes = traker_labels[wir[0]:wir[1]]
		# monitor_nodes = monitor_labels[wir[0]:wir[1]]
		# peer_list_nodes = peer_lists_labels[wir[0]:wir[1]]
	
		# if args.numberedges <= 0 or args.numberedges >= len(traker_nodes):
		# 	num_edges = len(traker_nodes)
		# else:
		# 	num_edges = args.numberedges	

		# nodes_list = []
		# # Label, tipo, cor dos vertices	
		# nodes_list.append((traker_nodes[0:num_edges], TRACKER, 'red'))
		# nodes_list.append((monitor_nodes[0:num_edges], MONITOR, 'blue'))
		# # nodes_list.append((peer_list_nodes[0:num_edges], PEER, 'black'))

		# # graph = create_graph(nodes_list)
		# graph = create_graph_peer_weights(nodes_list, peer_list_nodes[0:num_edges])


		# # graph_stellar = sg.StellarGraph.from_networkx(graph)
		# # print(graph_stellar.info())
		# # graphs_stellar.append(graph_stellar)

		# graphs.append(graph)

		# save_graph_txt(graph, len(graphs))

		# save_graph_fig(graph, len(graphs))

	logging.info(str(len(graphs)) + ' graphs in directory: out/')
	logging.info(str(len(graphs)) + ' images graphs in directory fig/')

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