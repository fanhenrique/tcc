import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

class Trace(object):
	
	def __init__(self, window, time, peer1, peer2, monitor1, monitor2):
		self.window = window
		self.time = time
		self.peer1 = peer1
		self.peer2 = peer2
		self.monitor1 = monitor1
		self.monitor2 = monitor2

	def printTrace(self):
		print(self.window, self.time, self.peer1, self.peer2, self.monitor1, self.monitor2)

def main():

	parser = argparse.ArgumentParser(description='Traces')

	parser.add_argument('--file', '-f', help='Arquivo de entrada', type=str)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s', datefmt=TIME_FORMAT, level=args.log)


	file = open(args.file, 'r')
	line = file.readline()

	traces = []
	# with open('data/data', 'r') as file:
	for i in range(0, 50):
		window, time, peer1, peer2, monitor1, monitor2 = file.readline().split(' ')
		traces.append(Trace(window, time, peer1, peer2, monitor1, monitor2))

	traces[23].printTrace()

if __name__ == '__main__':
	main()