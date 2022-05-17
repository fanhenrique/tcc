import math
import os
import shutil

import networkx as nx
from collections import Counter

#my imports
import vars_paths as vp

WINDOWS_LEN = 15.0

hash_count_tracker = 1
hash_table_tracker = {}

hash_count_monitor = 1
hash_table_monitor = {}

hash_count_peer = 1
hash_table_peer = {}

def init(showmaster=False, showpeers=False, showtrackers=False, showmonitors=False):

	vp.SHOWMASTER = showmaster
	vp.SHOWPEERS = showpeers
	vp.SHOWTRACKERS = showtrackers
	vp.SHOWMONITORS = showmonitors
	
	try:
		shutil.rmtree(vp.PATH)
		shutil.rmtree(vp.PATH_FIGS)
		shutil.rmtree(vp.PATH_GRAPHS)
		shutil.rmtree(vp.PATH_MATRICES)
	except FileNotFoundError:
		pass
	try:
		os.mkdir(vp.PATH)
		os.mkdir(vp.PATH_FIGS)
		os.mkdir(vp.PATH_GRAPHS)
		os.mkdir(vp.PATH_MATRICES)
	except FileExistsError:
		pass

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

def my_hash_peer(peer_str):

	global hash_table_peer
	global hash_count_peer

	my_hash_peer_value = hash_table_peer.get(peer_str)

	if my_hash_peer_value is None:
		my_hash_peer_value = hash_count_peer
		hash_table_peer[peer_str] = my_hash_peer_value
		hash_count_peer += 1

	return my_hash_peer_value

def cal_windows(epochs):
	
	time_min, windows = [], []

	# w_previous = 0 #remover depois q remover o -w
	# counter_windows = 0 #remover depois q remover o -w
	
	for e in epochs:		
		
		tm = (e - epochs[0]) / 60.0	
		w = math.trunc(tm / WINDOWS_LEN)

		# #remover depois q remover o -w
		# if w_previous != w:
		# 	counter_windows+=1	

		# #remover depois q remover o -w	
		# if counter_windows >= number_windows:
		# 	break
	
		time_min.append(tm)
		windows.append(w)
		
		# w_previous = w #remover depois q remover o -w

	windows_index_range = windows_range(windows) 
	
	return time_min, windows, windows_index_range

def windows_range(windows):

	windows_index_range = []
	break0 = 0
	for i in range(0, len(windows)):
		if i+1 == len(windows):
			break1 = i+1
			windows_index_range.append((break0, break1))
			break0 = break1
		else:
			if windows[i] != windows[i+1]:
				break1 = i+1
				windows_index_range.append((break0, break1))
				break0 = break1

	return windows_index_range


def read_file(file):
	problem_lines = 0
	epochs, trackers, monitors, peer_lists, leechers, seeders, ns = [], [], [], [], [], [], []
	with open(file, 'r') as file:
		for line in file:

			line_split = line.split()
			
			try:
				epoch = float(line_split[0])
				tracker = line_split[1].split("'")[1]	
				monitor = line_split[16].split("'")[1]

				l = int(line_split[9].split("'")[1])
				s = int(line_split[11].split("'")[1])
				n = int(line_split[13].split("'")[1])


				peer_list = []
				peer_list.append(line_split[18].split('{')[1].split(':')[0])
				i = 20
				while True:
					try:
						peer_list.append(line_split[i].split("'")[1].split(':')[0])
					except IndexError:
						break
					i+=2
			except:
				problem_lines+=1
				print(line)
				continue

			epochs.append(epoch)
			trackers.append(tracker)
			monitors.append(monitor)
			leechers.append(l)
			seeders.append(s)
			ns.append(n)
			peer_lists.append(peer_list)

	print('Problem lines: ', problem_lines)

	return epochs, trackers, monitors, peer_lists, leechers, seeders, ns


def create_graph_peer_weights(master, monitors, trackers, peer_lists):

	graph = nx.Graph()

	# vertices

	if vp.SHOWMASTER:
		graph.add_node(master, color_nodes=vp.COLOR_MASTERSERVER)
	graph.add_nodes_from(monitors, color_nodes=vp.COLOR_MONITOR)
	graph.add_nodes_from(trackers, color_nodes=vp.COLOR_TRACKER)
	if vp.SHOWPEERS:
		for peer_list in peer_lists:
			graph.add_nodes_from(peer_list, color_nodes=vp.COLOR_PEER)


	# arestas trackers peers
	edges_tp_weighted = []	
	for i in range(len(trackers)):
		for peer in peer_lists[i]:
			edges_tp_weighted.append((trackers[i], peer, 1))

	# print('EDGES TP WEIGHTED', edges_tp_weighted, len(edges_tp_weighted))

	edges_tp_weighted = list(dict.fromkeys(edges_tp_weighted)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)
	
	# print('EDGES TP WEIGHTED (RD)', edges_tp_weighted, len(edges_tp_weighted))
	if vp.SHOWPEERS:
		graph.add_weighted_edges_from(edges_tp_weighted)


	# print('----------------------------------')

	# pessos vindos dos trackers
	weights_t = Counter() 
	for k, _, w in edges_tp_weighted:
		weights_t[k] += w

	# print('WEIGHT T', weights_t, len(weights_t))

	# arestas monitors trackers
	edges_mt = list(zip(monitors, trackers))

	# print('EDGES MT', edges_mt, len(edges_mt))

	edges_mt = list(dict.fromkeys(edges_mt)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)

	# print('EDGES MT (RD))', edges_mt, len(edges_mt))

	edges_mt_weighted = []
	for e in edges_mt:
		edges_mt_weighted.append((e[0], e[1], weights_t[e[1]]))


	# print('EDGES MT WEIGHTED', edges_mt_weighted, len(edges_mt_weighted))
	graph.add_weighted_edges_from(edges_mt_weighted)


	# print('---------------------------------')

	if vp.SHOWMASTER:
		weights_m = Counter() 
		for k, _, w in edges_mt_weighted:
			weights_m[k] += w

		# print('WEIGHT M', weights_m, len(weights_m))

		edges_gm = []
		for m in monitors:
			edges_gm.append((master, m))
		
		# print('EDGES GM', edges_gm, len(edges_gm))

		edges_gm = list(dict.fromkeys(edges_gm)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)

		# print('EDGES GM (RD)', edges_gm, len(edges_gm))

		edges_gm_weighted = []
		for e in edges_gm:
			edges_gm_weighted.append((e[0], e[1], weights_m[e[1]]))

		# print('EDGES GM WEIGHTED', edges_gm_weighted, len(edges_gm_weighted))	

		graph.add_weighted_edges_from(edges_gm_weighted)

	# print('********************************************')

	return graph











def create_graph_master_tracker(master, monitors, trackers, peer_lists):


	graph = nx.Graph()

	# vertices
	
	graph.add_node(master, color_nodes=vp.COLOR_MASTERSERVER)
	
	graph.add_nodes_from(trackers, color_nodes=vp.COLOR_TRACKER)
	
	if vp.SHOWPEERS:
		for peer_list in peer_lists:
			graph.add_nodes_from(peer_list, color_nodes=vp.COLOR_PEER)

			
	# arestas trackers peers
	edges_tp_weighted = []	
	for i in range(len(trackers)):
		for peer in peer_lists[i]:
			edges_tp_weighted.append((trackers[i], peer, 1))

	# print('EDGES TP WEIGHTED', edges_tp_weighted, len(edges_tp_weighted))

	edges_tp_weighted = list(dict.fromkeys(edges_tp_weighted)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)
	
	# print('EDGES TP WEIGHTED (RD)', edges_tp_weighted, len(edges_tp_weighted))
	if vp.SHOWPEERS:
		graph.add_weighted_edges_from(edges_tp_weighted)
		

	# print('----------------------------------')

	# pessos vindos dos trackers
	weights_t = Counter() 
	for k, _, w in edges_tp_weighted:
		weights_t[k] += w


	edges_gt = []
	for t in trackers:
		edges_gt.append((master, t))
		

	edges_gt = list(dict.fromkeys(edges_gt))

	
	edges_gt_weighted = []
	for e in edges_gt:
		edges_gt_weighted.append((e[0], e[1], weights_t[e[1]]))

	

	graph.add_weighted_edges_from(edges_gt_weighted)



	return graph








# def create_graph_wt(master, monitors, trackers, peer_lists, wt):

# 	graph = nx.Graph()

# 	# vertices

# 	if vp.SHOWMASTER:
# 		graph.add_node(master, color_nodes=vp.COLOR_MASTERSERVER)
# 	graph.add_nodes_from(monitors, color_nodes=vp.COLOR_MONITOR)
# 	graph.add_nodes_from(trackers, color_nodes=vp.COLOR_TRACKER)
# 	if vp.SHOWPEERS:
# 		for peer_list in peer_lists:
# 			graph.add_nodes_from(peer_list, color_nodes=vp.COLOR_PEER)


		
# 	# arestas trackers peers
# 	edges_tp_weighted = []	
# 	for i in range(len(trackers)):
# 		for peer in peer_lists[i]:
# 			edges_tp_weighted.append((trackers[i], peer, 1))

# 	# print('EDGES TP WEIGHTED', edges_tp_weighted, len(edges_tp_weighted))

# 	edges_tp_weighted = list(dict.fromkeys(edges_tp_weighted)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)
	
# 	# print('EDGES TP WEIGHTED (RD)', edges_tp_weighted, len(edges_tp_weighted))
	
# 	if vp.SHOWPEERS:
# 		graph.add_weighted_edges_from(edges_tp_weighted)




# 	# print('----------------------------------')

# 	weights_t =  list(zip(trackers, wt))

# 	print(weights_t, len(weights_t))

# 	a = list(dict.fromkeys(weights_t))

# 	print(a, len(a))


# 	exit()


# 	# pessos vindos dos trackers
# 	weights_t = Counter() 
# 	for k, _, w in edges_tp_weighted:
# 		weights_t[k] += w

# 	# print('WEIGHT T', weights_t, len(weights_t))

# 	# arestas monitors trackers
# 	edges_mt = list(zip(monitors, trackers))

# 	# print('EDGES MT', edges_mt, len(edges_mt))


# 	edges_mt = list(dict.fromkeys(edges_mt)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)

# 	# print('EDGES MT (RD))', edges_mt, len(edges_mt))

# 	edges_mt_weighted = []
# 	for e in edges_mt:
# 		edges_mt_weighted.append((e[0], e[1], weights_t[e[1]]))


# 	# print('EDGES MT WEIGHTED', edges_mt_weighted, len(edges_mt_weighted))
# 	graph.add_weighted_edges_from(edges_mt_weighted)


# 	# print('---------------------------------')

# 	if vp.SHOWMASTER:
# 		weights_m = Counter() 
# 		for k, _, w in edges_mt_weighted:
# 			weights_m[k] += w

# 		# print('WEIGHT M', weights_m, len(weights_m))

# 		edges_gm = []
# 		for m in monitors:
# 			edges_gm.append((master, m))
		
# 		# print('EDGES GM', edges_gm, len(edges_gm))

# 		edges_gm = list(dict.fromkeys(edges_gm)) #remove duplicados (caso venha duas mensagens de um numa mesma janela)

# 		# print('EDGES GM (RD)', edges_gm, len(edges_gm))

# 		edges_gm_weighted = []
# 		for e in edges_gm:
# 			edges_gm_weighted.append((e[0], e[1], weights_m[e[1]]))

# 		# print('EDGES GM WEIGHTED', edges_gm_weighted, len(edges_gm_weighted))	

# 		graph.add_weighted_edges_from(edges_gm_weighted)

# 	# print('********************************************')

# 	return graph



# LABEL USADO NO STELLARGRAPH
	# label dos vertices 	
	# dict_nodes = {}	
	# for nodes in nodes_list[0:-1]:
	# 	for n in nodes[0]:
	# 		dict_nodes[n] = nodes[1]
	# nx.set_node_attributes(graph, dict_nodes, 'label')


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
