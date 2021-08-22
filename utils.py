import math

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
