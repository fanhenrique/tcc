import math
import os
import shutil
import matplotlib.pyplot as plt

import networkx as nx


WINDOWS_LEN = 15

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

	w_previous = 0
	counter_windows = 0

	print(len(epochs))
		
	for e in epochs:		
		
		tm = (e - epochs[0]) / 60	
		w = math.trunc(tm / WINDOWS_LEN)

		if w_previous != w:
			counter_windows+=1	
	
		time_min.append(tm)
		windows.append(w)
		w_previous = w

		if counter_windows >= number_windows:
			break
	
				

	print(list(zip(time_min, windows)))	
	print(number_windows)

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
	
	pos = nx.spring_layout(graph)
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
