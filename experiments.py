import shlex
import subprocess
from datetime import datetime
import os

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def runRNA(c, date_path, execution):
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
								' --number %d'\
								' --path %s' % (trainrate, seqlen, predlen, gcnsize, lstmsize, batch, epochs, execution, date_path)
								param = shlex.split(cmd_gcn_lstm)
								subprocess.call(param)

def runARIMA(c, date_path, execution):
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
						' --number %d'\
						' --path %s' % (trainrate, ar, ma, diff, execution, date_path)
						param = shlex.split(cmd_arima)
						subprocess.call(param)

def main():

	parser = argparse.ArgumentParser(description='Experiments')

	parser.add_argument('--file', '-f', help='input file', required=True, type=str)
	parser.add_argument('--number', '-n', help='number of executions', default=1, type=int)

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


	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	#ARIMA parameters
	# a = {'ar':[2,1,0], 'ma':[2,1,0], 'diff':[2,1,0], 'trainrate':[0.8]}

	# testes arima
	a2 = {'ar':[0], 'ma':[0], 'diff':[1], 'trainrate':[0.8]}
	a6 = {'ar':[0], 'ma':[1], 'diff':[2], 'trainrate':[0.8]}
	a8 = {'ar':[0], 'ma':[2], 'diff':[1], 'trainrate':[0.8]}
	a9 = {'ar':[0], 'ma':[2], 'diff':[2], 'trainrate':[0.8]}
	a10 = {'ar':[1], 'ma':[0], 'diff':[0], 'trainrate':[0.8]}
	a11 = {'ar':[1], 'ma':[0], 'diff':[1], 'trainrate':[0.8]}
	a12 = {'ar':[1], 'ma':[0], 'diff':[2], 'trainrate':[0.8]}
	a13 = {'ar':[1], 'ma':[1], 'diff':[0], 'trainrate':[0.8]}
	a15 = {'ar':[1], 'ma':[1], 'diff':[2], 'trainrate':[0.8]}
	a16 = {'ar':[1], 'ma':[2], 'diff':[0], 'trainrate':[0.8]}
	a17 = {'ar':[1], 'ma':[2], 'diff':[1], 'trainrate':[0.8]}
	a18 = {'ar':[1], 'ma':[2], 'diff':[2], 'trainrate':[0.8]}
	a19 = {'ar':[2], 'ma':[0], 'diff':[0], 'trainrate':[0.8]}
	a20 = {'ar':[2], 'ma':[0], 'diff':[1], 'trainrate':[0.8]}
	a21 = {'ar':[2], 'ma':[0], 'diff':[2], 'trainrate':[0.8]}
	a26 = {'ar':[2], 'ma':[2], 'diff':[1], 'trainrate':[0.8]}
	a27 = {'ar':[2], 'ma':[2], 'diff':[2], 'trainrate':[0.8]}
	
	#ruins
	a3 = {'ar':[0], 'ma':[0], 'diff':[2], 'trainrate':[0.8]}
	a4 = {'ar':[0], 'ma':[1], 'diff':[0], 'trainrate':[0.8]}
	a7 = {'ar':[0], 'ma':[2], 'diff':[0], 'trainrate':[0.8]}

	#não funcionam
	# a1 = {'ar':[0], 'ma':[0], 'diff':[0], 'trainrate':[0.8]} # gera resultado linear
	# a5 = {'ar':[0], 'ma':[1], 'diff':[1], 'trainrate':[0.8]}
	# a14 = {'ar':[1], 'ma':[1], 'diff':[1], 'trainrate':[0.8]}
	# a22 = {'ar':[2], 'ma':[1], 'diff':[0], 'trainrate':[0.8]}
	# a23 = {'ar':[2], 'ma':[1], 'diff':[1], 'trainrate':[0.8]}
	# a24 = {'ar':[2], 'ma':[1], 'diff':[2], 'trainrate':[0.8]}
	# a25 = {'ar':[2], 'ma':[2], 'diff':[0], 'trainrate':[0.8]}



	# a28 = {'ar':[4, 8], 'ma':[4, 8], 'diff':[0, 1], 'trainrate':[0.8]}

	# a29 = {'ar':[3], 'ma':[3, 2, 1, 0], 'diff':[0, 1], 'trainrate':[0.8]}

	# a30 = {'ar':[2, 1, 0], 'ma':[3], 'diff':[0, 1], 'trainrate':[0.8]}


	a28 = {'ar':[0], 'ma':[3], 'diff':[0, 1, 2], 'trainrate':[0.8]}
	a29 = {'ar':[1], 'ma':[3], 'diff':[2], 'trainrate':[0.8]}
	a30 = {'ar':[2], 'ma':[3], 'diff':[2], 'trainrate':[0.8]}
	a31 = {'ar':[3], 'ma':[3], 'diff':[2], 'trainrate':[0.8]}
	a32 = {'ar':[3], 'ma':[0], 'diff':[0, 1, 2], 'trainrate':[0.8]}


	#RNA parameters
	# r1 = {'seqlen':[1,2,4,8,16], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	# r2 = {'seqlen':[8], 'predlen':[2,4,8,16], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	# r3 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[8,32], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	# r4 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[100,300], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	# r5 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[16,64], 'epochs':[500], 'trainrate':[0.8]}
	# r6 = {'seqlen':[8], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[1000], 'trainrate':[0.8]}
	
	
	# r2 = {'seqlen':[4], 'predlen':[1,2,4,8,16], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	# r3 = {'seqlen':[4], 'predlen':[1], 'gcnsize':[8,32], 'lstmsize':[200], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	# r4 = {'seqlen':[4], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[100,300], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}
	# r5 = {'seqlen':[4], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[16,64], 'epochs':[500], 'trainrate':[0.8]}
	# r6 = {'seqlen':[4], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[200], 'batch':[32], 'epochs':[1000], 'trainrate':[0.8]}
	
	rm = {'seqlen':[4], 'predlen':[1], 'gcnsize':[16], 'lstmsize':[100], 'batch':[32], 'epochs':[500], 'trainrate':[0.8]}

	for e in range(args.number):
		# runARIMA(a2, date_path, e)
		# runARIMA(a6, date_path, e)
		# runARIMA(a8, date_path, e)
		# runARIMA(a9, date_path, e)
		# runARIMA(a10, date_path, e)
		# runARIMA(a11, date_path, e)
		# runARIMA(a12, date_path, e)
		# runARIMA(a13, date_path, e)
		# runARIMA(a15, date_path, e)
		# runARIMA(a16, date_path, e)
		# runARIMA(a17, date_path, e)
		# runARIMA(a18, date_path, e)
		# runARIMA(a19, date_path, e)
		# runARIMA(a20, date_path, e)
		# runARIMA(a21, date_path, e)
		# runARIMA(a26, date_path, e)
		# runARIMA(a27, date_path, e)
		# runARIMA(a1, date_path, e)
		# runARIMA(a3, date_path, e)
		# runARIMA(a4, date_path, e)
		# runARIMA(a7, date_path, e)

		runARIMA(a28, date_path, e)
		runARIMA(a29, date_path, e)
		runARIMA(a30, date_path, e)
		runARIMA(a31, date_path, e)
		runARIMA(a32, date_path, e)
		

	# for e in range(args.number):
		# runRNA(rm, date_path, e)
		# runRNA(r1, date_path, e)
		# runRNA(r2, date_path, e)
		# runRNA(r3, date_path, e)
		# runRNA(r4, date_path, e)
		# runRNA(r5, date_path, e)
		# runRNA(r6, date_path, e)

if __name__ == '__main__':
	main()