import networkx as nx

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
MASTERSERVER = 'MASTER SERVER'

COLOR_TRACKER = 'red'
COLOR_MONITOR = 'blue'
COLOR_PEER = 'green'
COLOR_MASTERSERVER = 'yellow'


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

def create_graph_peer_weights(master, monitors, trackers, peer_lists):

	graph = nx.Graph()

	# vertices

	graph.add_node(master, color_nodes=COLOR_MASTERSERVER)
	graph.add_nodes_from(monitors, color_nodes=COLOR_MONITOR)
	graph.add_nodes_from(trackers, color_nodes=COLOR_TRACKER)
	for peer_list in peer_lists:
		graph.add_nodes_from(peer_list, color_nodes=COLOR_PEER)



	# arestas trackers peers
	edges_tp_weighted = []	
	for i in range(len(trackers)):
		for peer in peer_lists[i]:
			edges_tp_weighted.append((trackers[i], peer, 1))

	print(edges_tp_weighted, len(edges_tp_weighted))

	edges_tp_weighted = list(dict.fromkeys(edges_tp_weighted)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)
	
	print(edges_tp_weighted, len(edges_tp_weighted))
	graph.add_weighted_edges_from(edges_tp_weighted)


	print('----------------------------------')

	# pessos vindos dos trackers
	weights_t = Counter() 
	for k, _, w in edges_tp_weighted:
		weights_t[k] += w

	print(weights_t, len(weights_t))

	# arestas monitors trackers
	edges_mt = list(zip(monitors, trackers))

	print(edges_mt, len(edges_mt))

	edges_mt_weighted = []
	for e in edges_mt:
		edges_mt_weighted.append((e[0], e[1], weights_t[e[1]]))

	print(edges_mt_weighted, len(edges_mt_weighted))		

	edges_mt_weighted = list(dict.fromkeys(edges_mt_weighted)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)

	print(edges_mt_weighted, len(edges_mt_weighted))
	graph.add_weighted_edges_from(edges_mt_weighted)


	print('---------------------------------')

	weights_m = Counter() 
	for k, _, w in edges_mt_weighted:
		weights_m[k] += w

	print(weights_m, len(weights_m))

	edges_gm = []
	for m in monitors:
		edges_gm.append((master, m))
	
	print(edges_gm, len(edges_gm))

	edges_gm = list(dict.fromkeys(edges_gm)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)

	print(edges_gm, len(edges_gm))

	edges_gm_weighted = []
	for e in edges_gm:
		edges_gm_weighted.append((e[0], e[1], weights_m[e[1]]))

	print(edges_gm_weighted, len(edges_gm_weighted))	

	graph.add_weighted_edges_from(edges_gm_weighted)

	print('------------------------------')

	return graph

def save_graph_adj_csv(graphs):

	with open('monitoring_adj.csv', 'w') as file:
		for g in graphs:
			matrix_adj = nx.adjacency_matrix(g)
			for i in range(matrix_adj.shape[0]):
				for j in range(matrix_adj.shape[1]):
					
					if matrix_adj[i,j] != 0:
						file.write('1\n') if i == matrix_adj.shape[0]-1 and j == matrix_adj.shape[1]-1 else file.write('1,')
					else:
						file.write('0\n') if i == matrix_adj.shape[0]-1 and j == matrix_adj.shape[1]-1 else file.write('0,')

def save_graph_weigths_csv(graphs):

	with open('monitoring_weigths.csv', 'w') as file:
		for g in graphs:
			matrix_adj = nx.adjacency_matrix(g)
			for i in range(matrix_adj.shape[0]):
				for j in range(matrix_adj.shape[1]):				
					file.write(str(matrix_adj[i,j])+'\n') if i == matrix_adj.shape[0]-1 and j == matrix_adj.shape[1]-1 else file.write(str(matrix_adj[i,j])+',')
			

def main():

	parser = argparse.ArgumentParser(description='Create toy case')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', required=True, type=str)
	
	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)


	utils.init()

	logging.info('reading file ...')
	windows, monitors, trackers, peer_lists = read_file(args.file)


	logging.info('range windows ...')
	windows_index_range = utils.windows_range(windows)

	graphs = []
	logging.info('creating graphs ...')
	for wir in windows_index_range:

		ms = 'MS'
		m = monitors[wir[0]:wir[1]]
		t = trackers[wir[0]:wir[1]]	
		pl = peer_lists[wir[0]:wir[1]]
		
		graph = create_graph_peer_weights(ms, m, t, pl)

		graphs.append(graph)

		# utils.save_graph_txt(graph, len(graphs))

		# utils.show_graph(graph)

		# utils.save_graph_fig(graph, len(graphs))

	save_graph_adj_csv(graphs)
	save_graph_weigths_csv(graphs)		
	
	logging.info(str(len(graphs)) + ' graphs in directory: out/')
	logging.info(str(len(graphs)) + ' images graphs in directory fig/')


if __name__ == '__main__':
	main()