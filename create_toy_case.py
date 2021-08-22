import math

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'
WINDOWS_LEN = 15

TRACKER = 'TRACKER'
MONITOR = 'MONITOR'
PEER = 'PEER'

hash_count_tracker = 1
hash_table_tracker = {}

hash_count_monitor = 1
hash_table_monitor = {}

hash_count_peer = 1
hash_table_peer = {}


def readFile(file):
	epochs, trakers, monitors, peer_lists = [], [], [], []
	with open(file, 'r') as file:
		file.readline() #ignora cabeçalho 
		for line in file:

			line_split = line.split()
			
			try:
				epochs.append(float(line_split[0]))
			except:
				print(line)
				continue
			try:
				trakers.append(line_split[1].split("'")[1])
			except:
				print(line)
				epochs.pop()
				continue
			try:
				monitors.append(line_split[16].split("'")[1])	
			except:
				print(line)
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
				print(line)
				epochs.pop()
				trakers.pop()
				monitors.pop()	

	return epochs, trakers, monitors, peer_lists


def cal_windows(epoch, number_windows):
	
	time_min = []
	windows = []


	w_previous = 0
	counter_windows = 0

	for e in epoch:		
		if counter_windows < number_windows:

			tm = (e - epoch[0]) / 60	
			w = math.trunc(tm / WINDOWS_LEN)

			if w_previous != w:
				counter_windows+=1	
		
			time_min.append(tm)
			windows.append(w)
			w_previous = w
		else:
			break
	
	windows_index_range = []
	break0 = 0
	for i in range(len(windows)-1):
		if windows[i] != windows[i+1]:
			break1 = i
			windows_index_range.append((break0, break1))
			break0 = break1+1


	return time_min, windows, windows_index_range	



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


def main():

	parser = argparse.ArgumentParser(description='Create toy case')

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

	global hash_table_tracker
	global hash_count_tracker

	global hash_table_monitor
	global hash_count_monitor

	global hash_table_peer
	global hash_count_peer


	logging.info('reading file ...')
	epochs, trakers, monitors, peer_lists =  readFile(args.file)

	logging.info('calculating windows ...')
	time_min, windows, windows_index_range = cal_windows(epochs, args.numberwindows)


	# Label pra os vertices
	traker_labels = []
	for t in trakers:
		# traker_labels.append(TRACKER+'_'+str(my_hash_tracker(t)))
		traker_labels.append(my_hash_tracker(t))
	monitor_labels = []
	for m in monitors:
		# monitor_labels.append(MONITOR+'_'+str(my_hash_monitor(m)))
		monitor_labels.append(my_hash_monitor(m))
	peer_labels = []
	for l in peer_lists:
		for p in l:
			# peer_labels.append(PEER+'_'+str(my_hash_peer(p)))
			peer_labels.append(my_hash_peer(p))

	# print(windows_index_range)

	logging.info('save file ...')
	with open('toy_case.txt', 'w') as file:
		
		for wir in windows_index_range:
		
			traker_nodes = traker_labels[wir[0]:wir[1]]
			monitor_nodes = monitor_labels[wir[0]:wir[1]]
			peer_list_nodes = peer_labels[wir[0]:wir[1]]

			if args.numberedges == 0:
				num_edges = len(traker_nodes)
			else:
				num_edges = args.numberedges
			
			for i in range(num_edges):
				
				file.write(str(traker_nodes[i]) + ' ')
				file.write(str(monitor_nodes[i]) + ' ')
				for peer in peer_list_nodes:
					file.write(str(peer)+ ' ')
				file.write('\n')


	logging.info('file created toy_case.txt')



if __name__ == '__main__':
	main()