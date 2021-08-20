def readFile(file, n):
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
	return epochs, trakers, monitors


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


	= readFile(args.file)


if __name__ == '__main__':
	main()