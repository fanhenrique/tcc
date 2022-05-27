import shlex
import subprocess
from datetime import datetime

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

TRAINRATE = [0.8]

# ARIMA parameters
AR = [2, 1, 0]
MA = [2, 1, 0]
DIFF = [2, 1, 0]

# RNA parameters
SEQLEN = [1, 2, 4, 8]
PREDLEN = [1, 2, 4, 8]
GCNSIZE = [8, 16, 32]
LSTMSIZE = [100, 200, 300]
BATCH = [16, 32, 64]
EPOCHS = [250, 500, 750]

def main():

	parser = argparse.ArgumentParser(description='Experiments')

	parser.add_argument('--file', '-f', help='input file', required=True, type=str)
	# parser.add_argument('--plot', '-p', help='plot mode', action='store_true')
	# parser.add_argument('--mean', '-m', help='main mode', action='store_true')
	# parser.add_argument('--trainrate', '-tr', help='Taxa de treinamento', default=DEFAULT_TRAIN_RATE, type=float)
	# parser.add_argument('--ar', '-p', help='Ordem do termo AR', default=DEFAULT_AR, type=int)
	# parser.add_argument('--ma', '-q', help='Ordem do termo MA', default=DEFAULT_MA, type=int)
	# parser.add_argument('--diff', '-d', help='Ordem de diferenciação', default=DEFAULT_DIFF, type=int)
	# parser.add_argument('--cutaxis', '-c', help='Corta eixo x', default=DEFALT_CUT_AXIS, type=int)


	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)

	# cmd_graph = 'python3 graph.py --file %s' % (args.file)
	# param = shlex.split(cmd_graph)
	# subprocess.call(param)

	date_path = '%s' % datetime.now().strftime('%m-%d_%H-%M-%S')

	c1 = {'seqlen': [1, 2, 4, 8, 16], 'predlen': [1], 'gcnsize': [16], 'lstmsize': [200], 'batch': [32], 'epochs': [500]}

	for trainrate in TRAINRATE:
		for ar in AR:
			for ma in MA:
				for diff in DIFF:
					cmd_arima = 'python3 arima_walk_forward.py'\
					' --weigths out/out-matrices/monitoring-weigths.csv'\
					' --trainrate %f'\
					' --ar %d'\
					' --ma %d'\
					' --diff %d'\
					' --path %s' % (trainrate, ar, ma, diff, date_path)
					param = shlex.split(cmd_arima)
					subprocess.call(param)

	for trainrate in TRAINRATE:
		for seqlen in c1['seqlen']:
			for predlen in c1['predlen']:
				for gcnsize in c1['gcnsize']:
					for lstmsize in c1['lstmsize']:
						for batch in c1['batch']:
							for epochs in c1['epochs']:
								cmd_gcn_lstm = 'python3 stellar_model.py'\
								' --adjs out/out-matrices/monitoring-adj.csv'\
								' --weigths out/out-matrices/monitoring-weigths.csv'\
								' --trainrate %f'\
								' --seqlen %d'\
								' --predlen %d'\
								' --gcnsize %d'\
								' --lstmsize %d'\
								' --batch %d'\
								' --epochs %d'\
								' --path %s' % (trainrate, seqlen, predlen, gcnsize, lstmsize, batch, epochs, date_path)
								param = shlex.split(cmd_gcn_lstm)
								subprocess.call(param)



if __name__ == '__main__':
	main()