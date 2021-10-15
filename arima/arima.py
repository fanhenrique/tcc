from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima.model import ARIMA

# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', names=['value'], header=0)

ts = ['t1', 't2', 't3', 't4', 't5', 't6', 't7']

df = pd.read_csv('../out/out-matrices/monitoring-weigths.csv', names=ts, header=None)

print(df)

print(df.t1)

df['sum'] = df.sum(axis=1)

print(df)

# for t in ts:

# 	print(t)
# 	result = adfuller(df[t].dropna())
# 	print('ADF Statistic: %f' % result[0])
# 	print('p-value: %f' % result[1])


# 	print('adf %d' % ndiffs(df[t], test='adf'))
# 	print('kpss %d' % ndiffs(df[t], test='kpss'))
# 	print('pp %d' % ndiffs(df[t], test='pp'))


# 	plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# 	fig, axes = plt.subplots(2, sharex=True)


# 	axes[0].plot(df[t])
# 	axes[0].set_title('%s' % t)
# 	plot_acf(df[t], lags=df[t].size-1, ax=axes[1], title='Autocorrelation %s' % t)

# 	plt.show()

# 	print('---------------------------------------')


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

model = ARIMA(df['sum'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())



pred = model_fit.predict(dynamic=False)

plt.plot(df['sum'], "b-", label="verdadeiro")
plt.plot(pred, "r-", label="predição")
plt.xlabel("Snapshots", fontsize=15)
plt.ylabel("Quantidade de pares", fontsize=15)
plt.legend(loc="best", fontsize=15)
# plt.savefig(path_out+'/test_all.png', format='png')
plt.show()
