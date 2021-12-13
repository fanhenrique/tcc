import pandas as pd
import numpy as np
import sys

def main():

	los_adj = pd.read_csv('../out/out-matrices/monitoring-adj.csv', header=None)

	los_speed = pd.read_csv('../out/out-matrices/monitoring-weigths.csv', header=None)

	
	adj = [int(sys.argv[1])]

	weigths = los_speed[int(sys.argv[2])]


	np.savetxt('../out/out-matrices/monitoring-adj.csv', adj, fmt='%d')	
	np.savetxt('../out/out-matrices/monitoring-weigths.csv', weigths, fmt='%d')	




if __name__ == '__main__':
	main()