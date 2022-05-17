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


def init():
	filename = inspect.getframeinfo(inspect.currentframe()).filename
	path = os.path.dirname(os.path.abspath(filename))


	path_date = '%s' % datetime.now().strftime('%m-%d_%H-%M-%S')

	path_plots = path + '/plots'
	path_plots = os.path.join(path_plots, path_date)
	if not os.path.exists(path_plots):
	    os.makedirs(path_plots)
	    os.makedirs(path_plots+'/png')

	path_outs = path + '/outs'
	path_outs = os.path.join(path_outs, path_date)
	if not os.path.exists(path_outs):
	    os.makedirs(path_outs)
	# else:
	# 	shutil.rmtree(path_outs)
	# 	os.makedirs(path_outs)
	
	return path_outs, path_plots

def mean_squared_error(data, prediction):
	
	return [((prediction[i]-data[i])**2)/len(data) for i in range(len(data))]


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


def plot_mse(column, mse, path_plots, title, max_y):
	
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
	plt.savefig(path_plots+'/mse_'+str(column)+'.svg', format='svg')
	plt.savefig(path_plots+'/png/mse_'+str(column)+'.png', format='png')
	# plt.show()
	plt.cla()
	plt.clf()
	plt.close('all')


def plot_prediction(column, true, prediction, path_plots, name, title, max_y):


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
	plt.savefig(path_plots+'/'+name+'_'+str(column)+'.svg', format='svg')
	plt.savefig(path_plots+'/png/'+name+'_'+str(column)+'.png', format='png')
	# plt.show()
	plt.cla()
	plt.clf()
	plt.close('all')


def main():

	parser = argparse.ArgumentParser(description='Arima')

	# parser.add_argument('--plot', '-p', help='plot mode', action='store_true')
	# parser.add_argument('--mean', '-m', help='main mode', action='store_true')
	parser.add_argument('--trainrate', '-tr', help='Taxa de treinamento', default=DEFAULT_TRAIN_RATE, type=float)
	parser.add_argument('--ar', '-p', help='Ordem do termo AR', default=DEFAULT_AR, type=int)
	parser.add_argument('--ma', '-q', help='Ordem do termo MA', default=DEFAULT_MA, type=int)
	parser.add_argument('--diff', '-d', help='Ordem de diferenciação', default=DEFAULT_DIFF, type=int)
	parser.add_argument('--cutaxis', '-c', help='Corta eixo x', default=DEFALT_CUT_AXIS, type=int)
	parser.add_argument('--weigths', '-w', help='Matriz de pessos', required=True, type=str)


	help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	args = parser.parse_args()

	if args.log == logging.DEBUG:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
	else:
		logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)


	path_outs, path_plots = init()
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
		
	# else:

	# if args.mean:


	logging.info('Reading file ...')
	df = pd.read_csv(args.weigths, header=None)

	
	df[df.shape[1]] = df.mean(axis=1)


	# result = adfuller(df['mean'].dropna())
	# print('ADF Statistic: %f' % result[0])
	# print('p-value: %f' % result[1])

	# print('adf %d' % ndiffs(df['mean'], test='adf'))
	# print('kpss %d' % ndiffs(df['mean'], test='kpss'))
	# print('pp %d' % ndiffs(df['mean'], test='pp'))


	max_y_data = np.max(df.max())

	
	df_mse = pd.DataFrame()
	
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

		start_time = time.time()

		# inicia Walk-Forward
		logging.info('Tracker %s walk-forward ...' % column)
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
		plot_prediction(column, test_data[args.cutaxis:], test_predictions[args.cutaxis:], path_plots, 'prediction_test', 'Predição ARIMA tracker '+str(column)+' - Teste', max_y_data)
		np.savetxt(path_outs+'/prediction_'+str(column)+'.csv', test_predictions)
		plot_prediction(column, data, predictions, path_plots, 'prediction_all', 'Predição ARIMA tracker '+str(column)+' - S1', max_y_data)


		# Calculate MSE of the tracker
		mse = np.array(mean_squared_error(test_data, test_predictions))[args.cutaxis:]
		df_mse[column] = mse


	logging.info('End prediction ARIMA')
		
	np.savetxt(path_outs+'/times.csv', times, fmt='%.8f')


	logging.info('Plots the MSE of all tracker')
	mean_mse = []
	max_y_mse = np.max(df_mse.max())
	for column in df_mse:
		plot_mse(column, df_mse[column], path_plots, 'Erro Quadrático Médio ARIMA tracker '+str(column)+' - Teste', max_y_mse)
		np.savetxt(path_outs+'/mse_'+str(column)+'.csv', df_mse[column], fmt='%.8f')

		mean_mse.append(np.mean(df_mse[column]))
	np.savetxt(path_outs+'/mean.csv', mean_mse, fmt='%.8f')


	logging.info('plots directory: %s' % path_plots)
	logging.info('outputs directory: %s' % path_outs)

	# else:

	# 	logging.info('Reading file ...')
	# 	df = pd.read_csv('../out/out-matrices/monitoring-weigths.csv', header=None)

		
	# 	# df[df.shape[1]] = df.mean(axis=1)
	

	# 	# result = adfuller(df['mean'].dropna())
	# 	# print('ADF Statistic: %f' % result[0])
	# 	# print('p-value: %f' % result[1])

	# 	# print('adf %d' % ndiffs(df['mean'], test='adf'))
	# 	# print('kpss %d' % ndiffs(df['mean'], test='kpss'))
	# 	# print('pp %d' % ndiffs(df['mean'], test='pp'))


	# 	max_y = np.max(df.max())


	# 	mean_mse = []
	# 	df_mse = pd.DataFrame()
		
	# 	logging.info('Start prediction ARIMA\n\
	# 		  Data: %s Train Rate: %s' % (df.shape[0], args.trainrate))
	


	# 	df_test_true = pd.DataFrame()
	# 	df_test_pred = pd.DataFrame()

	# 	for column in df:
			
	# 		data = df[column].to_numpy()
	# 		train_data, test_data = train_test_split(data, args.trainrate)
	# 		# print("Train data: ", train_data.shape)
	# 		# print("Test data: ", test_data.shape)
			

	# 		# prediction = []
	# 		history = [x for x in train_data]

	# 		# inicia Walk-Forward
	# 		logging.info('Tracker %s walk-forward ...' % column)
	# 		for t in range(test_data.shape[0]):
		  
	# 			model = ARIMA(history, order=(1,0,1))
				
	# 			model_fit = model.fit()

	# 			# predict_value = model_fit.forecast()[0]

	# 			# prediction.append(predict_value)

	# 			real_value = test_data[t]

	# 			# history.append(prediction[t])
	# 			history.append(real_value)


	# 			# print('%s %d predito=%.3f, esperado=%3.f' % (column, t, valor_predito, valor_real))

	# 		logging.info('End tracker %d' % column)
	# 		predictions = model_fit.predict(start=0, end=data.size-1, dynamic=False)
	# 		train_predictions, test_predictions = train_test_split(predictions, args.trainrate)	


	# 		logging.info('Plot prediction tracker %d' % column)
	# 		# plot_autocorrelation(column, data, path_plots)
	# 		plot_prediction(column, test_data, test_predictions, path_plots, 'prediction_test', max_y)
	# 		np.savetxt(path_outs+'/prediction_'+str(column)+'.csv', test_predictions)
	# 		# plot_prediction(column, data, predictions, path_plots, 'prediction_all', max_y)


	# 		logging.info('Calculate MSE tracker %d' % column)
	# 		mse = np.array(mean_squared_error(test_data, test_predictions))
	# 		df_mse[column] = mse



	# 		df_test_true[column] = test_data	
	# 		df_test_pred[column] = test_predictions
			

	# 	logging.info('End prediction ARIMA')
			
		



		

	# 	# df_test_true = df
	# 	column_mean = df.shape[1]
	# 	df_test_true[column_mean] = df_test_true.mean(axis=1)
	# 	df_test_pred[column_mean] = df_test_pred.mean(axis=1)

	# 	print(df_test_true)
	# 	print(df_test_pred)


	# 	mean_true = df_test_true[column_mean].to_numpy()
	# 	mean_pred = df_test_pred[column_mean].to_numpy()

	# 	plot_prediction(column_mean, mean_true, mean_pred, path_plots, 'prediction_test', max_y)
	# 	np.savetxt(path_outs+'/prediction_'+str(column_mean)+'.csv', mean_pred)


	# 	mse = np.array(mean_squared_error(mean_true, mean_pred))
	# 	df_mse[column_mean] = mse





	# 	logging.info('Plots MSE')
	# 	max_y_mse = np.max(df_mse.max())
	# 	for column in df_mse:
	# 		plot_mse(column, df_mse[column], path_plots, max_y_mse)
	# 		np.savetxt(path_outs+'/mse_'+str(column)+'.csv', df_mse[column])

	# 		mean_mse.append(np.mean(df_mse[column]))
	# 	np.savetxt(path_outs+'/mean.csv', mean_mse)

		


	# 	logging.info('plots directory: %s' % path_plots)
	# 	logging.info('outputs directory: %s' % path_outs)















		# exit()


		# # SPLIT DATA
		# data = df['mean'].to_numpy()
		# train_data, test_data = train_test_split(data, args.trainrate)
		# print("Train data: ", train_data.shape)
		# print("Test data: ", test_data.shape)

		# prediction = []
		# history = [x for x in train_data]

		# # inicia Walk-Forward
		# for t in range(test_data.shape[0]):
		  
		# 	model = ARIMA(history, order=(1,0,1))
			
		# 	model_fit = model.fit()

		# 	valor_predito = model_fit.forecast()[0]

		# 	prediction.append(valor_predito)

		# 	valor_real = test_data[t]

		# 	# history.append(prediction[t])
		# 	history.append(valor_real)


		# 	print('%d predito=%.3f, esperado=%3.f' % (t, valor_predito, valor_real))


		# print(model_fit.summary())

		# model_fit.plot_diagnostics(figsize=(15,8))
		# plt.show()

		# # train_predictions = model_fit.predict(start=train_data.size, end=data.size-1, dynamic=False)
		
		# predictions = model_fit.predict(start=0, end=data.size-1, dynamic=False)

		# train_predictions, test_predictions = train_test_split(predictions, args.trainrate)

		# print('data', data.size)
		# print('train', train_data.size)
		# print('test', test_data.size)
		# print('train predictions', train_predictions.size)
		# print('test predictions', test_predictions.size)

		# plot(data, predictions, train_data, test_data, train_predictions, test_predictions, path_plots)

		# df.to_csv(path_outs+'/data.csv', index=False, header=False)	
		# np.savetxt(path_outs+'/prediction.csv', predictions)


if __name__ == '__main__':
	main()


