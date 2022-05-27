import inspect
import os
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt

def init(method, data_path, args):
	filename = inspect.getframeinfo(inspect.currentframe()).filename
	path = os.path.dirname(os.path.abspath(filename))

	path = path + '/experiments/' + data_path + '/' + method + '/'

	keys = list(args.keys())
	values = list(args.values())
	for i in range(len(keys)-1):
		path = path + keys[i] + str(values[i]) + '-'
	path = path + keys[-1] + str(values[-1])

	path_plots = path + '/plots'
	if not os.path.exists(path_plots):
	    os.makedirs(path_plots)
	    os.makedirs(path_plots+'/png')

	path_outs = path + '/outs'	
	if not os.path.exists(path_outs):
	    os.makedirs(path_outs)
	# else:
	# 	shutil.rmtree(path_outs)
	# 	os.makedirs(path_outs)
	
	return path_outs, path_plots

def mean_squared_error(data, prediction):
    
    return [((prediction[i]-data[i])**2)/len(data) for i in range(len(data))]


def plot_mse(mse, path_plots, name, title, max_y):
    
    plt.figure(figsize=(15,8))

    plt.xlim([-(mse.shape[0]*0.02), mse.shape[0]+(mse.shape[0]*0.02)])

    xticks = np.arange(0, mse.shape[0], mse.shape[0]*0.1)
    xticks = np.append(xticks, mse.shape[0])
    plt.xticks(xticks, fontsize=13)


    plt.ylim([-(max_y*0.02), max_y+max_y*0.02])

    yticks = np.arange(0, max_y, max_y*0.1)
    yticks = np.append(yticks, max_y)       
    plt.yticks(yticks, fontsize=13)


    plt.plot(mse, 'g-', label='mse')
    plt.ylabel('Erro quadrático médio', fontsize=15)
    plt.xlabel("Snapshots", fontsize=15)
    plt.title(title)
    plt.savefig(path_plots+'/'+name+'.svg', format='svg')
    plt.savefig(path_plots+'/png/'+name+'.png', format='png')
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')



def plot_prediction(true, prediction, path_plots, name, title, max_y):

    # max_y = 216.0

    ## PREDICTION TEST ##
    plt.figure(figsize=(15,8))

    plt.xlim([-(true.shape[0]*0.02), true.shape[0]+(true.shape[0]*0.02)])
    
    xticks = np.arange(0, true.shape[0], true.shape[0]*0.1)
    xticks = np.append(xticks, true.shape[0])
    plt.xticks(xticks, fontsize=13)


    plt.ylim([-(max_y*0.02), max_y+max_y*0.02])

    yticks = np.arange(0, max_y, max_y*0.1)
    yticks = np.append(yticks, max_y)       
    plt.yticks(yticks, fontsize=13)


    plt.plot(true, "b-", label="verdadeiro")
    plt.plot(prediction, "r-", label="predição")
    plt.xlabel("Snapshots", fontsize=15)
    plt.ylabel("Quantidade de pares", fontsize=15)
    plt.legend(loc="best", fontsize=15)
    plt.title(title)
    plt.savefig(path_plots+'/'+name+'.svg', format='svg')
    plt.savefig(path_plots+'/png/'+name+'.png', format='png')
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')
