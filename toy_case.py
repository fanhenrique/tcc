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

		# utils.save_graph_txt(graph, len(graphs))

		# utils.show_graph(graph)

		# utils.save_graph_fig(graph, len(graphs))

	utils.save_graph_adj_csv(graphs)
	utils.save_graph_weigths_csv(graphs)		
	
	logging.info(str(len(graphs)) + ' graphs in directory: out/')
	logging.info(str(len(graphs)) + ' images graphs in directory fig/')


if __name__ == '__main__':
	main()