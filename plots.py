import inspect
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def plot_mean(labels, mean, std, path_plots, name, ylabel, max_y, t):


    
    plt.figure(figsize=(15,8))

    plt.ylim([-(max_y*0.02), max_y*2*0.5])

    yticks = np.arange(0, max_y*2, max_y*0.2)
    yticks = np.append(yticks, max_y)       
    

    m = plt.bar(x=labels, width = 0.3, height=mean, yerr=std)

    l = []
    for e in std:
    	l.append('± '+f'{e:.{t}f}')	
    # print(l)

    # plt.bar_label(m, labels=['± '+(':.'+str(t)+'f'.format(e)) for e in std], label_type='edge', fontsize=16, padding=2)
    plt.bar_label(m, labels=l, label_type='edge', fontsize=16, padding=2)
    plt.bar_label(m, padding=0, fmt='%.'+str(t)+'f', fontsize=16, label_type='center')
    
    plt.tick_params(labelsize=27)

    plt.ylabel(ylabel, fontsize=27)
    
    plt.yticks(yticks, fontsize=18)
    plt.savefig(path_plots+'/'+name+'.svg', format='svg')
    plt.savefig(path_plots+'/png/'+name+'.png', format='png')

    # plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')

def main():
	parser = argparse.ArgumentParser(description='plots')
	parser.add_argument("--save", help='name plot', required=True, type=str)
	parser.add_argument("--paths", nargs='+', help='list of paths', type=str)
	parser.add_argument("--exp", nargs='+', help='list of experiments', type=str)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)
	args = parser.parse_args()
	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)


	print(args)

	filename = inspect.getframeinfo(inspect.currentframe()).filename
	mypath = os.path.dirname(os.path.abspath(filename))

	path = mypath+'/mean/'

	df_mse = pd.DataFrame()
	df_times = pd.DataFrame()

	for p in args.paths:

		mse = pd.read_csv(path+'mse/'+p+'mean_mse_executions.csv', sep=' ', header=None)
		times = pd.read_csv(path+'times/'+p+'mean_times_executions.csv', sep=' ', header=None)

		# print(mse.to_numpy())
		df_mse = pd.concat([df_mse, mse])
		# df_mse.append(mse.to_numpy())
		df_times = pd.concat([df_times, times])
	

	print(df_mse)
	# print(df_mse[2].to_numpy()[0])

	# print('-------------')
	# print(df_mse[2].tail(1))
	# # print(df_mse.columns)

	

	path_plot_mse = mypath + '/plots/mse'
	if not os.path.exists(path_plot_mse):
		os.makedirs(path_plot_mse)
	path_plot_mse_png = mypath + '/plots/mse/png'
	if not os.path.exists(path_plot_mse_png):
		os.makedirs(path_plot_mse_png)

	plot_mean(args.exp,
		df_mse[0].to_numpy(),
		df_mse[1].to_numpy(),
		path_plot_mse,
		args.save+'-mse',
		'Erro quadrático médio',
		df_mse[2].to_numpy()[0],
		8
	)



	path_plot_times = mypath + '/plots/times'
	if not os.path.exists(path_plot_times):
		os.makedirs(path_plot_times)
	path_plot_times_png = mypath + '/plots/times/png'
	if not os.path.exists(path_plot_times_png):
		os.makedirs(path_plot_times_png)

	plot_mean(args.exp,
		df_times[0].to_numpy(),
		df_times[1].to_numpy(),
		path_plot_times,
		args.save+'-times',
		'Segundos',
		df_times[2].to_numpy()[0],
		4
	)





if __name__ == '__main__':
	main()