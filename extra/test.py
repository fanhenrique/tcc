import pandas as pd
import numpy as np
import sys

def main():

	los_adj = pd.read_csv('../out/out-matrices/monitoring-adj.csv', header=None)

	los_speed = pd.read_csv('../out/out-matrices/monitoring-weigths.csv', header=None)


	print(los_adj)
	print(los_speed)

	
	adj = np.ones(shape=(len(sys.argv)-1, len(sys.argv)-1), dtype=int)

	np.fill_diagonal(np.flipud(adj), 0)

	print(adj)


	weigths = pd.DataFrame()
	for i in range(1,len(sys.argv)):
		weigths[i] = los_speed[int(sys.argv[i])]

	print(weigths)


	np.savetxt('../out/out-matrices/monitoring-adj.csv', adj, delimiter=',', fmt='%d')	
	np.savetxt('../out/out-matrices/monitoring-weigths.csv', weigths, delimiter=',', fmt='%d')	




if __name__ == '__main__':
	main()