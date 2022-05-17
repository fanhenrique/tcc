import shlex
import subprocess

import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def main():

	parser = argparse.ArgumentParser(description='Experiments')

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


	cmd_arima = 'python3 arima_walk_forward.py'\
	' --weigths out/out-matrices/monitoring-weigths.csv'\
	' --trainrate 0.8'\
	' --ar 1'\
	' --ma 1'\
	' --diff 0'
	param = shlex.split(cmd_arima)
	subprocess.call(param)

	cmd_gcn_lstm = 'python3 stellar_model.py'\
	' --adjs out/out-matrices/monitoring-adj.csv'\
	' --weigths out/out-matrices/monitoring-weigths.csv'
	' --trainrate 0.8'\
	' --seqlen 4'\
	' --predlen 1'\
	' --epochs 500'\
	' --batch 32'\
	' --gcnsize 16'\
	' --lstmsize 200'
	param = shlex.split(cmd_gcn_lstm)
	subprocess.call(param)



if __name__ == '__main__':
	main()