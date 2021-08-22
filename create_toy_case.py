import math

import utils

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'
WINDOWS_LEN = 15

TRACKER = 'TRACKER'
MONITOR = 'MONITOR'
PEER = 'PEER'


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


	logging.info('reading file ...')
	epochs, trakers, monitors, peer_lists =  readFile(args.file)

	logging.info('calculating windows ...')
	time_min, windows, windows_index_range = utils.cal_windows(epochs, args.numberwindows)


	# Label pra os vertices
	traker_labels = []
	for t in trakers:
		# traker_labels.append(TRACKER+'_'+str(my_hash_tracker(t)))
		traker_labels.append(utils.my_hash_tracker(t))
	monitor_labels = []
	for m in monitors:
		# monitor_labels.append(MONITOR+'_'+str(my_hash_monitor(m)))
		monitor_labels.append(utils.my_hash_monitor(m))
	peer_labels = []
	for l in peer_lists:
		for p in l:
			# peer_labels.append(PEER+'_'+str(my_hash_peer(p)))
			peer_labels.append(utils.my_hash_peer(p))

	# print(windows_index_range)

	logging.info('save file ...')
	with open('toy_case.txt', 'w') as file:
		
		for i, wir in windows_index_range:
			print(i, wir)
		
			traker_nodes = traker_labels[wir[0]:wir[1]]
			monitor_nodes = monitor_labels[wir[0]:wir[1]]
			peer_list_nodes = peer_labels[wir[0]:wir[1]]

			if args.numberedges == 0:
				num_edges = len(traker_nodes)
			else:
				num_edges = args.numberedges
			
			for j in range(num_edges):

				file.write(str(i))
				file.write(str(traker_nodes[j]) + ' ')
				file.write(str(monitor_nodes[j]) + ' ')
				for peer in peer_list_nodes:
					file.write(str(peer)+ ' ')
				file.write('\n')


	logging.info('file created toy_case.txt')



if __name__ == '__main__':
	main()