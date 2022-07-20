import inspect
import os
import shutil
import numpy as np
import pandas as pd


import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def main():

	parser = argparse.ArgumentParser(description='mean')
	parser.add_argument('--path', help='Date path', required=True, type=str)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)
	args = parser.parse_args()
	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)


	filename = inspect.getframeinfo(inspect.currentframe()).filename
	mypath = os.path.dirname(os.path.abspath(filename))


	dirs = os.listdir(args.path)
	for d1 in dirs:
		dirs2 = os.listdir(args.path+d1)
	exc = args.path.split('/')
	path = mypath+'/mean/'+exc[-2]+'/'


	# exc = args.path.split('/')
	# print()
	if os.path.exists(path):
		shutil.rmtree(path)
	# exit()

	allTimes = {}
	allMaxTimes = []
	for d2 in dirs2:

		path_times = path + 'times/'+d2
		if not os.path.exists(path_times):
			os.makedirs(path_times)
		else:
			shutil.rmtree(path+'times/')
			os.makedirs(path_times)

		times = pd.DataFrame()
		for d1 in dirs:
			# print(args.path+d1+'/'+d2+'/outs/times.csv')
			t = pd.read_csv(args.path+d1+'/'+d2+'/outs/times.csv', header=None)
			times = pd.concat([times, t]) 

		# print(times)
		# print('-------------------------')
		times_mean = times.mean(axis=0).to_numpy()
		times_std = times.std(axis=0).to_numpy()
		print(times)
		print(times_mean)
		print(times_std)
		
		# times = times.head(1)
		# t = {'mean': times_mean, 'std': times_std}
		# print(t)
		# np.savetxt(path_times+'/mean_times_executions.csv', [times.mean(axis=0), times.std(axis=0)] , fmt='%.8f')

		df = pd.DataFrame({'mean':times_mean, 'std':times_std})
		print(df)
		allTimes[d2] = df
		
		allMaxTimes.append(times_mean)

	print(allMaxTimes)	

	max_y_times = np.max(allMaxTimes)
	print(max_y_times)


	for d2 in dirs2:
		allTimes[d2]['max_y'] = max_y_times
		# print(allTimes[d2])
		path_times = path + 'times/'+d2
		allTimes[d2].to_csv(path_times+'/mean_times_executions.csv', sep=' ', header=None, columns=['mean', 'std', 'max_y'], float_format='%.8f', index=False)







	allMse = {}
	allMaxMse = []
	for d2 in dirs2:

		path_mse = path + 'mse/'+d2
		if not os.path.exists(path_mse):
			os.makedirs(path_mse)
		else:
			shutil.rmtree(path+'mse/')
			os.makedirs(path_mse)

		mse = pd.DataFrame()
		for d1 in dirs:
			# print(args.path+d1+'/'+d2+'/outs/mse.csv')
			t = pd.read_csv(args.path+d1+'/'+d2+'/outs/mean.csv', header=None)
			# mse = pd.concat([mse, t]) 
			mse[d1] = t
		
		print(d2)
		print(mse.iloc[10])
		mse_mean = mse.mean(axis=1).to_numpy()[-1]
		mse_std = mse.std(axis=1).to_numpy()[-1]
		# m = mse['mean', 'std']
		print(mse)
		print(mse_mean)
		print(mse_std)
		
		# np.savetxt(path_mse+'/mean_mse_executions.csv', np.transpose(mse['mean'], mse['std']), fmt='%.8f')	

		# mse = mse.tail(1)
		
		df = pd.DataFrame({'mean':[mse_mean], 'std':[mse_std]})
		print(df)

		allMse[d2] = df

		allMaxMse.append(mse_mean)
		print('---------------------------------------------------------')

	print(allMaxMse)

	max_y_mse = np.max(allMaxMse)
	print(max_y_mse)	


	for d2 in dirs2:
		allMse[d2]['max_y'] = max_y_mse
		print(allMse[d2])
		path_mse = path + 'mse/'+d2
		allMse[d2].to_csv(path_mse+'/mean_mse_executions.csv', sep=' ', header=None, columns=['mean', 'std', 'max_y'], float_format='%.8f', index=False)

if __name__ == '__main__':
	main()