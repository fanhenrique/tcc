# referencias arima
# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# https://github.com/stacktecnologias/stack-repo/blob/164cf328d0e007de260666d75a84bbf76defd2c3/Arima-Tutorial.ipynb

import inspect
import os
from datetime import datetime
import shutil
import time

import argparse
import logging

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima.model import ARIMA

import utils_predictions as utils


DEFAULT_TRAIN_RATE = 0.8
DEFAULT_AR = 1
DEFAULT_MA = 1
DEFAULT_DIFF = 0


DEFALT_CUT_AXIS = 0


DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'


def train_test_split(data, trainrate):
    time_len = data.size
    train_size = int(time_len * trainrate)
    train_data = np.array(data[:train_size])
    test_data = np.array(data[train_size:])
    return train_data, test_data



def plot_autocorrelation(column, data, path_plots):

	## AUTOCORRELATION ##
	plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

	fig, axes = plt.subplots(2, sharex=True)

	axes[0].plot(data)
	axes[0].set_title('tracker')
	plot_acf(data, lags=data.size-1, ax=axes[1], title='Autocorrelation')
	fig.savefig(path_plots+'/autocorrelation_'+str(column)+'.svg', format='svg')
	fig.savefig(path_plots+'/png/autocorrelation_'+str(column)+'.png', format='png')
	# plt.show()
	plt.cla()
	plt.clf()
	plt.close('all')



def main():

	parser = argparse.ArgumentParser(description='Arima')

	# parser.add_argument('--plot', '-p', help='plot mode', action='store_true')
	parser.add_argument('--trainrate', '-tr', help='Taxa de treinamento', default=DEFAULT_TRAIN_RATE, type=float)
	parser.add_argument('--ar', '-p', help='Ordem do termo AR', default=DEFAULT_AR, type=int)
	parser.add_argument('--ma', '-q', help='Ordem do termo MA', default=DEFAULT_MA, type=int)
	parser.add_argument('--diff', '-d', help='Ordem de diferenciação', default=DEFAULT_DIFF, type=int)
	parser.add_argument('--cutaxis', '-c', help='Corta eixo x', default=DEFALT_CUT_AXIS, type=int)
	parser.add_argument('--weigths', '-w', help='Matriz de pessos', required=True, type=str)
	parser.add_argument('--path', help='Date path', required=True, type=str)
	parser.add_argument('--number', '-n', help='number of execution', type=int)

	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)
	
	
	pars = {i: vars(args)[i] for i in ('trainrate', 'ar', 'ma', 'diff')}
	path_outs, path_plots = utils.init('arima-'+str(args.number), args.path, pars)
	# print(path_outs)
	# print(path_plots)

	# if args.plot:

	# 	# print(path_outs, path_plots)

	# 	data = pd.read_csv(path_outs+'/data.csv', header=None)
	# 	data = data[data.shape[1]-1].to_numpy()

	# 	predictions = pd.read_csv(path_outs+'/prediction.csv', header=None).to_numpy()


	# 	train_data, test_data = train_test_split(data, args.trainrate)		
	# 	train_predictions, test_predictions = train_test_split(predictions, args.trainrate)		
	
	
	# 	plot(data, predictions, train_data, test_data, train_predictions, test_predictions, path_plots)
		

	logging.info('Reading file ...')
	df = pd.read_csv(args.weigths, header=None)

	
	#################### df[df.shape[1]] = df.mean(axis=1)


	# result = adfuller(df['mean'].dropna())
	# print('ADF Statistic: %f' % result[0])
	# print('p-value: %f' % result[1])

	# print('adf %d' % ndiffs(df['mean'], test='adf'))
	# print('kpss %d' % ndiffs(df['mean'], test='kpss'))
	# print('pp %d' % ndiffs(df['mean'], test='pp'))


	max_y_data = np.max(df.max())

	
	df_mse = pd.DataFrame()
	df_preds = pd.DataFrame()
	
	logging.info('Start prediction ARIMA\n\
		          Data: %s Train Rate: %s' % (df.shape, args.trainrate))

	times = []

	for column in df:
		
		data = df[column].to_numpy()
		train_data, test_data = train_test_split(data, args.trainrate)
		# logging.info("All data: %s", data.shape)
		# logging.info("Train data: %s", train_data.shape)
		# logging.info("Test data: %s", test_data.shape)

		# prediction = []
		history = [x for x in train_data]

	
		# inicia Walk-Forward
		logging.info('Tracker %s walk-forward ...' % column)
		start_time = time.time()
		for t in range(test_data.shape[0]):
	  
			model = ARIMA(history, order=(args.ar, args.diff, args.ma))
			
			model_fit = model.fit()

			# predict_value = model_fit.forecast()[0]

			# prediction.append(predict_value)

			real_value = test_data[t]

			# history.append(prediction[t])
			history.append(real_value)

			# print('%s %d predito=%.3f, esperado=%3.f' % (column, t, valor_predito, valor_real))

		end_time = time.time()

		times.append(end_time-start_time)

		predictions = model_fit.predict(start=0, end=data.size-1, dynamic=False)
		train_predictions, test_predictions = train_test_split(predictions, args.trainrate)	


		logging.info('Plot prediction tracker %d' % column)
		plot_autocorrelation(column, data, path_plots)
		utils.plot_prediction(test_data[args.cutaxis:], test_predictions[args.cutaxis:], path_plots, 'prediction_test_'+str(column), 'Predição ARIMA tracker '+str(column)+' - Teste', max_y_data)
		np.savetxt(path_outs+'/prediction_test_'+str(column)+'.csv', test_predictions)
		utils.plot_prediction(data, predictions, path_plots, 'prediction_all_'+str(column), 'Predição ARIMA tracker '+str(column)+' - S1', max_y_data)
		np.savetxt(path_outs+'/prediction_all_'+str(column)+'.csv', predictions)	

		# Calculate MSE of the tracker
		mse = np.array(utils.mean_squared_error(test_data, test_predictions))[args.cutaxis:]
		df_mse[column] = mse

		df_preds[column] = predictions


	logging.info('End prediction ARIMA')
		
	a = sum(times)
	# print(type(a))
	np.savetxt(path_outs+'/times.csv', np.array([a]), fmt='%.8f')



	df[df.shape[1]] = df.mean(axis=1)
	df_preds[df_preds.shape[1]] = df_preds.mean(axis=1)

	maen_train_true, mean_test_true = train_test_split(df.iloc[:,-1].to_numpy(), args.trainrate)
	maen_train_pred, mean_test_pred = train_test_split(df_preds.iloc[:,-1].to_numpy(), args.trainrate)
	
	utils.plot_prediction(mean_test_true[args.cutaxis:], mean_test_pred[args.cutaxis:], path_plots, 'prediction_test_'+str(df.shape[1]-1), 'Predição ARIMA tracker '+str(df.shape[1]-1)+' - Teste', max_y_data)
	np.savetxt(path_outs+'/prediction_test_'+str(df.shape[1]-1)+'.csv', mean_test_pred)
	utils.plot_prediction(df.iloc[:,-1], df_preds.iloc[:,-1], path_plots, 'prediction_all_'+str(df.shape[1]-1), 'Predição ARIMA tracker '+str(df.shape[1]-1)+' - S1', max_y_data)
	np.savetxt(path_outs+'/prediction_all_'+str(df.shape[1]-1)+'.csv', df_preds.iloc[:,-1])	

	mse = np.array(utils.mean_squared_error(mean_test_true, mean_test_pred))[args.cutaxis:]
	df_mse[df_mse.shape[1]] = mse




	logging.info('Plots the MSE of all tracker')
	mean_mse = []
	max_y_mse = np.max(df_mse.max())
	for column in df_mse:
		utils.plot_mse(df_mse[column], path_plots, 'mse_'+str(column), 'Erro Quadrático Médio ARIMA tracker '+str(column)+' - Teste', max_y_mse)
		np.savetxt(path_outs+'/mse_'+str(column)+'.csv', df_mse[column], fmt='%.8f')

		mean_mse.append(np.mean(df_mse[column]))
	np.savetxt(path_outs+'/mean.csv', mean_mse, fmt='%.8f')


	logging.info('plots directory: %s' % path_plots)
	logging.info('outputs directory: %s' % path_outs)



if __name__ == '__main__':
	main()


