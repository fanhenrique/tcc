import math
import os
import shutil
import matplotlib.pyplot as plt

import networkx as nx


WINDOWS_LEN = 15.0


TRACKER = 'TRACKER'
MONITOR = 'MONITOR'
PEER = 'PEER'
MASTERSERVER = 'MASTER SERVER'

COLOR_TRACKER = 'red'
COLOR_MONITOR = 'blue'
COLOR_PEER = 'green'
COLOR_MASTERSERVER = 'yellow'


hash_count_tracker = 1
hash_table_tracker = {}

hash_count_monitor = 1
hash_table_monitor = {}

hash_count_peer = 1
hash_table_peer = {}

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

def cal_windows(epochs, number_windows):
	
	time_min, windows = [], []

	w_previous = 0 #remover depois q remover o -w
	counter_windows = 0 #remover depois q remover o -w
	
	for e in epochs:		
		
		tm = (e - epochs[0]) / 60.0	
		w = math.trunc(tm / WINDOWS_LEN)

		#remover depois q remover o -w
		if w_previous != w:
			counter_windows+=1	

		#remover depois q remover o -w	
		if counter_windows >= number_windows:
			break
	
		time_min.append(tm)
		windows.append(w)
		
		w_previous = w #remover depois q remover o -w

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
	epochs, trakers, monitors, peer_lists = [], [], [], []
	with open(file, 'r') as file:
		file.readline() #ignora cabeçalho 
		for line in file:

			line_split = line.split()
			
			try:
				epochs.append(float(line_split[0]))
			except:
				# print(line)
				continue
			try:
				trakers.append(line_split[1].split("'")[1])
			except:
				# print(line)
				epochs.pop()
				continue
			try:
				monitors.append(line_split[16].split("'")[1])	
			except:
				# print(line)
				epochs.pop()
				trakers.pop()
				continue
			try:
				peer_list = []
				peer_list.append(line_split[18].split('{')[1].split(':')[0])
				i = 20
				while True:
					try:
						peer_list.append(line_split[i].split("'")[1].split(':')[0])
					except IndexError:
						break
					i+=2
				peer_lists.append(peer_list)
			except:
				# print(line)
				epochs.pop()
				trakers.pop()
				monitors.pop()	

	return epochs, trakers, monitors, peer_lists


def draw_graph(graph):

	colors = [u[1] for u in graph.nodes(data='color_nodes')]
	# nx.draw(graph, with_labels=True, node_color=colors)
	
	# pos = nx.spring_layout(graph)
	pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
	weights = nx.get_edge_attributes(graph, "weight")

	nx.draw_networkx(graph, pos, with_labels=True, node_color=colors)
	nx.draw_networkx_edge_labels(graph, pos, edge_labels=weights)

def show_graph(graph):

	draw_graph(graph)
	
	plt.show()

def save_graph_fig(graph, g):

	draw_graph(graph)
		
	plt.savefig('fig/graph'+str(g)+'.png')
	plt.clf()

def save_graph_txt(graph, g):
	# print(graph.nodes.data())
	with open('out/graph'+str(g)+'.txt', 'w') as file:
		for edge in graph.edges.data():
			if edge[2]:
				file.write(edge[0] + ' ' + str(edge[2]['weight']) + ' ' + edge[1] + '\n')
			else:
				file.write(edge[0] + ' ' + edge[1] + '\n')

def init():
	try:
		shutil.rmtree('./out')
		shutil.rmtree('./fig')
	except FileNotFoundError:
		pass
	try:
		os.mkdir('./out')
		os.mkdir('./fig')
	except FileExistsError:
		pass

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
