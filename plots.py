import inspect
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'

def plot_mean(labels, mean, std, path_plots, name, ylabel, color, max_y, t, f, zero):

    plt.figure(figsize=(20,10))

    if zero == True:
    	plt.ylim([0, max_y*1.5])
    	yticks = np.arange(0, max_y*1.5, max_y*0.2)
    	print(yticks)
    else:
    	plt.ylim([-(max_y*0.1), max_y*1.5])
    	yticksp = np.arange(0, max_y*1.5, max_y*0.2)
    	print(yticksp)
    	yticksn = np.flip(np.arange(0, -max_y*0.1, -max_y*0.2))
    	print(yticksn)
    	yticks = np.concatenate([yticksn, yticksp])
    	print(yticks)
    	# yticks = np.append(yticks, max_y)       
	    

    m = plt.bar(x=labels, width = 0.55, height=mean, color=color)

    # print(m.errorbar.set_label(alpha=0.3))

    plt.errorbar(labels, mean, yerr=std, ls='', lw=1, capsize=5, color='red', alpha=0.5, capthick=1)

    l = []
    for e in std:
    	l.append('± '+f'{e:.{f}f}')	
    # print(l)

    # plt.bar_label(m, labels=['± '+(':.'+str(t)+'f'.format(e)) for e in std], label_type='edge', fontsize=16, padding=2)
    plt.bar_label(m, labels=l, label_type='edge', fontsize=25, padding=50, color='red', alpha=0.7)
    
    if zero == True:
    	plt.bar_label(m, padding=20, fmt='%.'+str(t)+'f', fontsize=25, label_type='center', color='k')
    else:	
    	plt.bar_label(m, padding=-2, fmt='%.'+str(t)+'f', fontsize=25, label_type='center', color='k')
    	

    	


    
    plt.tick_params(labelsize=30)

    plt.ylabel(ylabel, fontsize=30)
    # plt.xlabel(fontsize=30)
    
    plt.yticks(yticks, fontsize=27)
    plt.savefig(path_plots+'/'+name+'.svg', format='svg')
    plt.savefig(path_plots+'/png/'+name+'.png', format='png')

    # plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')

def main():
	parser = argparse.ArgumentParser(description='plots')
	parser.add_argument("--save", help='name plot', required=True, type=str)
	# parser.add_argument("--exp", help='experiments', required=True, type=str)
	parser.add_argument("--paths", nargs='+', help='list of paths', type=str)
	parser.add_argument("--col", nargs='+', help='list columns of experiments', type=str)
	# parser.add_argument('--zero', help='graph origin', required=True, action='store_true')

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


	# path = mypath+'/mean/'+args.exp+'/'
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
	print(df_times)
	# print(df_mse[2].to_numpy()[0])

	print('-------------')
	# print(df_mse[2].tail(1))
	# # print(df_mse.columns)

	print(df_mse[2])
	print(df_times[2])

	# exit()

	max_y_mse = np.max(df_mse[2].to_numpy())
	max_y_times = np.max(df_times[2].to_numpy())

	print(max_y_mse, max_y_times)


	# path_plot_mse = mypath + '/plots/mse/'+args.exp+'/'
	path_plot_mse = mypath + '/plots/mse/'
	if not os.path.exists(path_plot_mse):
		os.makedirs(path_plot_mse)
	# else:
	# 	shutil.rmtree(mypath + '/plots/mse')
	# 	os.makedirs(path_plot_mse)
	# path_plot_mse_png = mypath + '/plots/mse/'+args.exp+'/png'
	path_plot_mse_png = mypath + '/plots/mse/png'
	if not os.path.exists(path_plot_mse_png):
		os.makedirs(path_plot_mse_png)

	plot_mean(args.col,
		df_mse[0].to_numpy(),
		df_mse[1].to_numpy(),
		path_plot_mse,
		args.save+'-mse',
		'Erro quadrático médio',
		# df_mse[2].to_numpy()[0],
		'cornflowerblue',
		max_y_mse,
		6,
		6,
		zero=False
	)



	# path_plot_times = mypath + '/plots/times/'+args.exp+'/'
	path_plot_times = mypath + '/plots/times/'
	if not os.path.exists(path_plot_times):
		os.makedirs(path_plot_times)
	# else:
	# 	shutil.rmtree(mypath + '/plots/times')
	# 	os.makedirs(path_plot_times)
	# path_plot_times_png = mypath + '/plots/times/'+args.exp+'/png'
	path_plot_times_png = mypath + '/plots/times/png'
	if not os.path.exists(path_plot_times_png):
		os.makedirs(path_plot_times_png)

	plot_mean(args.col,
		df_times[0].to_numpy(),
		df_times[1].to_numpy(),
		path_plot_times,
		args.save+'-times',
		'Segundos',
		# df_times[2].to_numpy()[0],
		'tab:green',
		max_y_times,
		2,
		2,
		zero=True
	)





if __name__ == '__main__':
	main()