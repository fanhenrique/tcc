import networkx as nx
# import stellargraph as sg

from collections import Counter

import argparse
import logging

#my imports
import utils
import saves

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'



# def create_graph(nodes_list):

# 	graph = nx.Graph()

# 	# vertices
# 	for nodes in nodes_list:
# 		graph.add_nodes_from(nodes[0], color_nodes=nodes[2])

# 	# label dos vertices 	
# 	dict_nodes = {}	
# 	for nodes in nodes_list:
# 		for n in nodes[0]:
# 			dict_nodes[n] = nodes[1]
# 	nx.set_node_attributes(graph, dict_nodes, 'label')	

# 	# cria as restas
# 	edges = list(zip(nodes_list[0][0], nodes_list[1][0]))
	
# 	# conta os pesos das arestas
# 	weights = dict(Counter(edges))
	
# 	# arestas com peso
# 	weighted_edges = []
# 	for e in edges:
# 		weighted_edges.append((e[0], e[1], weights[(e[0], e[1])]))

# 	graph.add_weighted_edges_from(weighted_edges)

# 	return graph

# def create_graph_peer_weights2(nodes_list, peer_lists):

# 	graph = nx.Graph()

# 	# vertices
# 	for nodes in nodes_list:
# 		graph.add_nodes_from(nodes[0], color_nodes=nodes[2])

# 	# label dos vertices 	
# 	dict_nodes = {}	
# 	for nodes in nodes_list:
# 		for n in nodes[0]:
# 			dict_nodes[n] = nodes[1]
# 	nx.set_node_attributes(graph, dict_nodes, 'label')

# 	# cria as restas
# 	edges = list(zip(nodes_list[0][0], nodes_list[1][0]))
# 	# print(edges, len(edges))


# 	# conta pesos das arestas
# 	weights = []
# 	for peer in peer_lists:
# 		weights.append(len(peer))
# 	# print(weights, len(weights))


# 	ew = zip(edges, weights)

# 	c = Counter()
# 	for k, v in ew:
# 		c[k] += v
	

# 	weighted_edges = []
# 	for e in edges:
# 		weighted_edges.append((e[0], e[1], c[(e[0], e[1])]))

# 	graph.add_weighted_edges_from(weighted_edges)

# 	return graph

# def create_graph_peer_weights3(monitors, trackers, peer_lists):

# 	graph = nx.Graph()

# 	# vertices
# 	graph.add_nodes_from(monitors[0], color_nodes=monitors[2])
# 	graph.add_nodes_from(trackers[0], color_nodes=trackers[2])
	
# 	if SHOWPEERS: 
# 		for nodes in peer_lists[0]:
# 			graph.add_nodes_from(nodes, color_nodes=peer_lists[2])

		
# 	# USADO NO STELLARGRAPH
# 	# label dos vertices 	
# 	# dict_nodes = {}	
# 	# for nodes in nodes_list[0:-1]:
# 	# 	for n in nodes[0]:
# 	# 		dict_nodes[n] = nodes[1]
# 	# nx.set_node_attributes(graph, dict_nodes, 'label')


# 	# arestas trackers peers
# 	if SHOWPEERS:
# 		edges_tp = []	
# 		for i in range(len(peer_lists[0])):
# 			for peer in peer_lists[0][i]:
# 				edges_tp.append((trackers[0][i], peer))
# 		# print(edges_tp)
		
# 		graph.add_edges_from(edges_tp)

# 	# arestas monitors trackers
# 	edges_mt = list(zip(monitors[0], trackers[0]))
# 	# print(edges_mt, len(edges_mt))

# 	# conta pesos das arestas monitors trackers
# 	weights_mt = []
# 	for peer_list in peer_lists[0]:
# 		weights_mt.append(len(peer_list))
# 	# print(weights_mt, len(weights_mt))
	
# 	c = Counter()
# 	for k, v in zip(edges_mt, weights_mt):
# 		c[k] += v

# 	weighted_edges = []
# 	for e in edges_mt:
# 		weighted_edges.append((e[0], e[1], c[(e[0], e[1])]))

# 	graph.add_weighted_edges_from(weighted_edges)


# 	return graph



def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='file input', required=True, type=str)

	parser.add_argument('--showpeers', '-p', help='peers in graph', action='store_true')
	parser.add_argument('--showmaster', '-g', help='master in graph', action='store_true')
	parser.add_argument('--showtrackers', '-t', help='trackers in graph', action='store_true')
	parser.add_argument('--showmonitors', '-m', help='monitors in graph', action='store_true')

	# parser.add_argument('--numberwindows', '-w', help='number windows', default=1, type=int)
	# parser.add_argument('--numberedges', '-e', help='number edges', default=0, type=int) 

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)

	utils.init(showmaster = args.showmaster, showpeers = args.showpeers, showtrackers = args.showtrackers, showmonitors = args.showmonitors)

	logging.info('reading file ...')
	epochs, trackers, monitors, peer_lists, leechers, seeders, ns =  utils.read_file(args.file)

	ls = []
	for a, b in zip(leechers, seeders):
		ls.append(a+b)

	logging.info('calculating windows ...')
	time_min, windows, windows_index_range = utils.cal_windows(epochs)

	print('epochs:', len(epochs))
	print('trackers:', len(trackers))
	print('monitors:', len(monitors))
	print('peer_lists:', len(peer_lists))
	print('ls:', len(ls))
	print('n:', len(ns))
	print('windows:', len(windows))
	
	# print(windows_index_range, len(windows_index_range))

	for i in range(len(windows_index_range)):
		print(i, windows_index_range[i])
	# exit()

	logging.info('renaming entities ...')
	# Label pra os vertices
	monitor_labels = []
	for m in monitors:
		monitor_labels.append('M'+str(utils.my_hash_monitor(m)))
		# monitor_labels.append(str(utils.my_hash_monitor(m)))
		# monitor_labels.append(m)
	tracker_labels = []
	for t in trackers:
		tracker_labels.append('T'+str(utils.my_hash_tracker(t)))
		# tracker_labels.append(str(utils.my_hash_tracker(t)))
		# tracker_labels.append(t)
	peer_lists_labels = []
	for l in peer_lists:
		pl_labels = []
		for p in l:
			pl_labels.append('P'+str(utils.my_hash_peer(p)))
			# pl_labels.append(str(utils.my_hash_peer(p)))
			# pl_labels.append(p)
		peer_lists_labels.append(pl_labels)


	graphs = []
	logging.info('creating graphs ...')
	for wir in windows_index_range:
		
		ms = 'MS'
		m = monitor_labels[wir[0]:wir[1]]
		t = tracker_labels[wir[0]:wir[1]]	
		pl = peer_lists_labels[wir[0]:wir[1]]
		wt = ls[wir[0]:wir[1]]
		
		graph = utils.create_graph_master_tracker(ms, m, t, pl)

		graphs.append(graph)

		# saves.save_graph_txt(graph, len(graphs))

		# saves.show_graph(graph)

		# saves.save_graph_fig(graph, len(graphs))	


	# for g in graphs:
	# 	for e in g.edges.data():
	# 		print(e)
	# 	print('-------------')


	logging.info(str(len(graphs)) + ' graphs in directory: out_graphs/')
	logging.info(str(len(graphs)) + ' images graphs in directory figs_graphs/')

	saves.save_graph_adj_csv(graphs)
	saves.save_graph_weigths_csv(graphs)
	logging.info('adjacency and weight matrices are directory: out_matrices/')

	
if __name__ == '__main__':
	main()
