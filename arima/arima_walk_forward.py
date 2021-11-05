import inspect
import os
from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima.model import ARIMA

def train_test_split(data, train_portion):
    time_len = data.size
    train_size = int(time_len * train_portion)
    train_data = np.array(data[:train_size])
    test_data = np.array(data[train_size:])
    return train_data, test_data


def init():
	
	filename = inspect.getframeinfo(inspect.currentframe()).filename
	path = os.path.dirname(os.path.abspath(filename))

	out = path+'/out-arima'
	path1 = '%s' % datetime.now().strftime('%m-%d_%H-%M-%S')
	path_out = os.path.join(out,path1)
	if not os.path.exists(path_out):
	    os.makedirs(path_out)

	return path_out

def main():

	path_out = init()

	df = pd.read_csv('../out/out-matrices/monitoring-weigths.csv', header=None)

	print(df)

	df['sum'] = df.sum(axis=1)

	print(df)

	print('sum')
	result = adfuller(df['sum'].dropna())
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])


	print('adf %d' % ndiffs(df['sum'], test='adf'))
	print('kpss %d' % ndiffs(df['sum'], test='kpss'))
	print('pp %d' % ndiffs(df['sum'], test='pp'))


	plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

	fig, axes = plt.subplots(2, sharex=True)

	axes[0].plot(df['sum'])
	axes[0].set_title('sum')
	plot_acf(df['sum'], lags=df['sum'].size-1, ax=axes[1], title='Autocorrelation sum')
	fig.savefig(path_out+'/autocorrlation.svg', format='svg')
	plt.show()


	# SPLIT DATA

	data = df['sum'].to_numpy()


	train_rate = 0.8
	train_data, test_data = train_test_split(data, train_rate)
	print("Train data: ", train_data.shape)
	print("Test data: ", test_data.shape)

	predictions = []
	history = [x for x in train_data]

	# inicia Walk-Forward
	for t in range(test_data.shape[0]):
	  
		
		# cria um modelo ARIMA com os dados de history
		model = ARIMA(history, order=(1,0,1))
		
		# treina o modelo ARIMA
		model_fit = model.fit()

		# a variável valor_predito recebe o valor previsto pelo modelo
		valor_predito = model_fit.forecast()[0]

		
		# adiciona o valor predito na lista de predicões
		predictions.append(valor_predito)

		# a variável valor_real recebe o valor real do teste
		valor_real = test_data[t]

		# adiciona o valor real a variável history
		history.append(valor_real)

		# imprime valor predito e valor real
		print('%d predito=%.3f, esperado=%3.f' % (t, valor_predito, valor_real))


	pred = model_fit.predict(start=train_data.size, end=data.size-1, dynamic=False)
	

	print(data.size)
	print(train_data.size)
	print(test_data.size)
	print(pred.size)	
	

	# plt.plot(data, 'y-', label='data')
	plt.plot(test_data, "b-", label="verdadeiro")
	plt.plot(pred, "r-", label="predição")
	plt.xlabel("Snapshots", fontsize=15)
	plt.ylabel("Quantidade de pares", fontsize=15)
	plt.legend(loc="best", fontsize=15)
	plt.savefig(path_out+'/test_all.svg', format='svg')
	plt.show()



if __name__ == '__main__':
	main()