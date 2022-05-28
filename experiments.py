import shlex
import subprocess
from datetime import datetime

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def runRNA(c):
	for trainrate in c['trainrate']:
		for seqlen in c['seqlen']:
			for predlen in c['predlen']:
				for gcnsize in c['gcnsize']:
					for lstmsize in c['lstmsize']:
						for batch in c['batch']:
							for epochs in c['epochs']:
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

def runARIMA(c):
for trainrate in c['trainrate']:
		for ar in c['ar']:
			for ma in c['ma']:
				for diff in c['diff']:
					cmd_arima = 'python3 arima_walk_forward.py'\
					' --weigths out/out-matrices/monitoring-weigths.csv'\
					' --trainrate %f'\
					' --ar %d'\
					' --ma %d'\
					' --diff %d'\
					' --path %s' % (trainrate, ar, ma, diff, date_path)
					param = shlex.split(cmd_arima)
					subprocess.call(param)

def main():

	parser = argparse.ArgumentParser(description='Experiments')

	parser.add_argument('--file', '-f', help='input file', required=True, type=str)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)

	cmd_graph = 'python3 graph.py --file %s' % (args.file)
	param = shlex.split(cmd_graph)
	subprocess.call(param)

	date_path = '%s' % datetime.now().strftime('%m-%d_%H-%M-%S')

	#ARIMA parameters
	a1 = {'ar':[3,2,1,0], 'ma':[3,2,1,0], 'diff':[3,2,1,0], 'trainrate':[0.8]}

	#RNA parameters
	r1 = {'seqlen':[1,2,4,8,16], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	r2 = {'seqlen':[8], 'predlen':[1,2,4,8,16], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	r3 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[8,16,32], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	r4 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[100,200,300], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	r5 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[16,32,64], 'epochs':[500], 'trainrate':[0.8]}
	r6 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[1000], 'trainrate':[0.8]}
	
	runARIMA(a)
	runRNA(r1)
	runRNA(r2)
	runRNA(r3)
	runRNA(r4)
	runRNA(r5)
	runRNA(r6)

if __name__ == '__main__':
	main()