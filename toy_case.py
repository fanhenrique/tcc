import numpy as np
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


def save_graph_adj_csv(graphs, monitors, trackers, peer_lists):


	print('#################################')
	
	
	m = list(dict.fromkeys(monitors))
	t = list(dict.fromkeys(trackers))
	pls = []
	for pl in peer_lists:
		for p in pl:
			pls.append(p)
	ps = list(dict.fromkeys(pls))

	
	vector = ['MS']
	vector += m + t + ps
	print(vector, len(vector))
	
	
	# for g in graphs:
	# 	matrix = np.zeros((len(vector), len(vector)), dtype=int)
	# 	print(g.edges(), len(g.edges()))
	# 	for e in g.edges():
	# 		matrix[vector.index(e[0]), vector.index(e[1])] = 1
	# 		matrix[vector.index(e[1]), vector.index(e[0])] = 1

	# 	print(matrix.shape)	
	# 	print(matrix)

	print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


	with open('out_tgcn/monitoring_adj.csv', 'w') as file:

		matrixt = np.zeros((len(vector)*len(graphs), len(vector)*len(graphs)), dtype=int)

		for g in graphs:
			print(graphs.index(g), g.edges(), len(g.edges()))
			for e in g.edges():
				print((vector.index(e[0])+(graphs.index(g)*len(vector)), vector.index(e[1])+(graphs.index(g)*len(vector))), end=' ')
				print((vector.index(e[1])+(graphs.index(g)*len(vector)), vector.index(e[0])+(graphs.index(g)*len(vector))), end=' ')
				print((e[0], e[1]))
				
				matrixt[vector.index(e[0])+(graphs.index(g)*len(vector)), vector.index(e[1])+(graphs.index(g)*len(vector))] = 1
				matrixt[vector.index(e[1])+(graphs.index(g)*len(vector)), vector.index(e[0])+(graphs.index(g)*len(vector))] = 1
				

		for i in range(matrixt.shape[0]):
			for j in range(matrixt.shape[1]):
				file.write(str(matrixt[i,j])+'\n') if j == matrixt.shape[1]-1 else file.write(str(matrixt[i,j])+',')


	print('#################################')

def save_graph_weigths_csv(graphs, monitors, trackers, peer_lists):

	m = list(dict.fromkeys(monitors))
	t = list(dict.fromkeys(trackers))
	pls = []
	for pl in peer_lists:
		for p in pl:
			pls.append(p)
	ps = list(dict.fromkeys(pls))

	
	vector = ['MS']
	vector += m + t + ps
	
	
	with open('out_tgcn/monitoring_weigths.csv', 'w') as file:
		
		matrixt = np.zeros((len(vector)*len(graphs), len(vector)*len(graphs)), dtype=int)

		for g in graphs:
			print(graphs.index(g), g.edges(), len(g.edges()))
			for e in g.edges().data():

				print((vector.index(e[0])+(graphs.index(g)*len(vector)), vector.index(e[1])+(graphs.index(g)*len(vector))), end=' ')
				print((vector.index(e[1])+(graphs.index(g)*len(vector)), vector.index(e[0])+(graphs.index(g)*len(vector))), end=' ')
				print((e[0], e[1]), e[2]['weight'])

				
				matrixt[vector.index(e[0])+(graphs.index(g)*len(vector)), vector.index(e[1])+(graphs.index(g)*len(vector))] = e[2]['weight']
				matrixt[vector.index(e[1])+(graphs.index(g)*len(vector)), vector.index(e[0])+(graphs.index(g)*len(vector))] = e[2]['weight']
				

		for i in range(matrixt.shape[0]):
			for j in range(matrixt.shape[1]):
				file.write(str(matrixt[i,j])+'\n') if j == matrixt.shape[1]-1 else file.write(str(matrixt[i,j])+',')

			

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
		
		graph = utils.create_graph_peer_weights(ms, m, t, pl)
	

		graphs.append(graph)

		utils.save_graph_txt(graph, len(graphs))

		utils.show_graph(graph)

		utils.save_graph_fig(graph, len(graphs))

	save_graph_adj_csv(graphs, monitors, trackers, peer_lists)
	save_graph_weigths_csv(graphs, monitors, trackers, peer_lists)
			
	
	logging.info(str(len(graphs)) + ' graphs in directory: out/')
	logging.info(str(len(graphs)) + ' images graphs in directory fig/')


if __name__ == '__main__':
	main()