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



def main():

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
	  
		# difference data
		# meses_no_ano = 12
		# diff = difference(history, meses_no_ano)
		
		# cria um modelo ARIMA com os dados de history
		model = ARIMA(history, order=(0,0,1))
		
		# treina o modelo ARIMA
		model_fit = model.fit()

		# a variável valor_predito recebe o valor previsto pelo modelo
		valor_predito = model_fit.forecast()[0]

		# valor_predito recebe o valor revertido (escala original)
		# valor_predito = inverse_difference(history, valor_predito, meses_no_ano)

		# adiciona o valor predito na lista de predicões
		predictions.append(valor_predito)

		# a variável valor_real recebe o valor real do teste
		valor_real = test_data[t]

		# adiciona o valor real a variável history
		history.append(valor_real)

		# imprime valor predito e valor real
		print('Valor predito=%.3f, Valor esperado=%3.f' % (valor_predito, valor_real))



	pred = model_fit.predict(dynamic=False)

	plt.plot(data, 'y-', label='data')
	plt.plot(train_data, "b-", label="treino")
	plt.plot(pred, "r-", label="predição")
	plt.xlabel("Snapshots", fontsize=15)
	plt.ylabel("Quantidade de pares", fontsize=15)
	plt.legend(loc="best", fontsize=15)
	# plt.savefig(path_out+'/test_all.png', format='png')
	plt.show()



if __name__ == '__main__':
	main()