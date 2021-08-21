import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def readFile(file):
	epochs, trakers, monitors = [], [], []
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
			except:
				print(line)
				epochs.pop()
				trakers.pop()
				monitors.pop()	

	return epochs, trakers, monitors, peer_list


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


	logging.info('reading file ...')
	epochs, trakers, monitors =  readFile(args.file)

	logging.info('calculating windows ...')
	time_min, windows, windows_index_range = cal_windows(epochs, args.numberwindows)

	



if __name__ == '__main__':
	main()