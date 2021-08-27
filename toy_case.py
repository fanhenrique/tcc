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

def read_file(file):

	windows, monitors, trackers, peer_lists = [], [], [], []

	with open(file, 'r') as file:
		
		for line in file:
			l = line.split()

			windows.append(l[0])
			monitors.append(l[1])
			trackers.append(l[2])
			pl = []
			for x in l[3:]:
				pl.append(x)
			peer_lists.append(pl)

	return windows, monitors, trackers, peer_lists

def create_graph_peer_weights(monitors, trackers, peer_lists):

	graph = nx.Graph()

	# vertices
	graph.add_nodes_from(monitors[0], color_nodes=monitors[2])
	graph.add_nodes_from(trackers[0], color_nodes=trackers[2])
	graph.add_node('G', color_nodes='yellow')
	
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
				edges_tp.append((trackers[0][i], peer, 1))
		# print(edges_tp)

		graph.add_weighted_edges_from(edges_tp)



	# arestas monitors trackers
	edges_mt = list(zip(monitors[0], trackers[0]))
	# print(edges_mt, len(edges_mt))

	# conta pesos das arestas monitors trackers
	weights_mt = []
	for peer_list in peer_lists[0]:
		weights_mt.append(len(peer_list))
	# print(weights_mt, len(weights_mt))
	
	c_mt = Counter()
	for k, v in zip(edges_mt, weights_mt):
		c_mt[k] += v

	weighted_edges = []
	for e in edges_mt:
		weighted_edges.append((e[0], e[1], c_mt[(e[0], e[1])]))

	graph.add_weighted_edges_from(weighted_edges)


	edges_gm = []

	for i in range(len(monitors[0])):
				



	return graph

def main():

	parser = argparse.ArgumentParser(description='Create toy case')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)
	parser.add_argument('--showpeers', '-p', help='show peers', action='store_true')
	
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
	windows, monitors, trackers, peer_lists = read_file(args.file)


	logging.info('range windows ...')
	windows_index_range = utils.windows_range(windows)

	graphs = []
	logging.info('creating graphs ...')
	for wir in windows_index_range:

		m = (monitors[wir[0]:wir[1]], MONITOR, 'blue')
		t = (trackers[wir[0]:wir[1]], TRACKER, 'red')		
		pl = (peer_lists[wir[0]:wir[1]], PEER, 'green')

		graph = create_graph_peer_weights(m, t, pl)

		graphs.append(graph)

		utils.save_graph_txt(graph, len(graphs))

		utils.show_graph(graph)

		utils.save_graph_fig(graph, len(graphs))	
	
	logging.info(str(len(graphs)) + ' graphs in directory: out/')
	logging.info(str(len(graphs)) + ' images graphs in directory fig/')


if __name__ == '__main__':
	main()