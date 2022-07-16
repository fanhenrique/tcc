import inspect
import os
import pandas as pd
import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def plot_mean_mse(labels, mean, std, path_plots, name, ylabel, max_y):

    # max_y = 216.0

    ## PREDICTION TEST ##
    plt.figure(figsize=(15,8))

    plt.xlim([-(true.shape[0]*0.02), true.shape[0]+(true.shape[0]*0.02)])
    
    xticks = np.arange(0, true.shape[0], true.shape[0]*0.1)
    xticks = np.append(xticks, true.shape[0])
    plt.xticks(xticks, fontsize=27)


    plt.ylim([-(max_y*0.02), max_y+max_y*0.02])

    yticks = np.arange(0, max_y, max_y*0.1)
    yticks = np.append(yticks, max_y)       
    plt.yticks(yticks, fontsize=27)


    # plt.plot(true, "b-", label="verdadeiro")
    # plt.plot(prediction, "r-", label="predição")
    # plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    # plt.legend(loc="best", fontsize=30)
    # plt.title(title)
    plt.savefig(path_plots+'/'+name+'.svg', format='svg')
    plt.savefig(path_plots+'/png/'+name+'.png', format='png')
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')

def main():
	parser = argparse.ArgumentParser(description='mean')
	parser.add_argument('--name', help='name plot', required=True, type=str)
	parser.add_argument("--paths", nargs='+', help='list of paths', type=str)
	parser.add_argument("--exp", nargs='+', help='list of experiments', type=str)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)
	args = parser.parse_args()
	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)


	filename = inspect.getframeinfo(inspect.currentframe()).filename
	mypath = os.path.dirname(os.path.abspath(filename))

	path = mypath+'/mean/'

	df_mse = pd.DataFrame()
	df_times = pd.DataFrame()

	for p in args.paths:

		mse = pd.read_csv(path+'mse/'+p+'mean_mse_executions.csv')
		times = pd.read_csv(path+'times/'+p+'mean_times_executions.csv')
		
		df_mse = pd.concat([df_mse, mse], keys=['mean', 'std', 'max_y'], ignore_index=True)
		df_times = pd.concat([df_times, times], keys=['mean', 'std', 'max_y'], ignore_index=True)
	

	print(df_mse)
	print(df_times)

	print(df_mse['mean'])

	# path_plot_mse = mypath + '/plots/mse'
	# plot_mean_mse(args.exp,
	# 	df_mse['mean'],
	# 	df_mse['std'],
	# 	path_plot_mse,
	# 	args.name,
	# 	'Média do erro quadrático médio',
	# 	df_mse['max_y'][0]
	# 	)









if __name__ == '__main__':
	main()