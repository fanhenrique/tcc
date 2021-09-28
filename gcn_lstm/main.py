import os
import sys
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


import stellargraph as sg



# los_adj = pd.read_csv(r'../T-GCN/data/los_adj.csv', header=None)
# los_adj = pd.read_csv(r'../T-GCN/data/sz_adj.csv', header=None)

los_adj = pd.read_csv(r'out/out_matrices/monitoring_adj.csv',header=None)
sensor_dist_adj = np.mat(los_adj)


# los_speed = pd.read_csv(r'../T-GCN/data/los_speed.csv')
#los_speed = pd.read_csv(r'../T-GCN/data/sz_speed.csv')

los_speed = pd.read_csv(r'out/out_matrices/monitoring_weigths.csv',header=None)
speed_data = np.mat(los_speed)


sensor_dist_adj = sensor_dist_adj.transpose()
speed_data = speed_data.transpose()


num_nodes, time_len = speed_data.shape
print("No. of sensors:", num_nodes, "\nNo of timesteps:", time_len)


def train_test_split(data, train_portion):
    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = np.array(data[:, :train_size])
    test_data = np.array(data[:, train_size:])
    return train_data, test_data

train_rate = 0.8

train_data, test_data = train_test_split(speed_data, train_rate)
print("Train data: ", train_data.shape)
print("Test data: ", test_data.shape)


def scale_data(train_data, test_data):
    max_speed = train_data.max()
    min_speed = train_data.min()

    # train_scaled = (train_data - min_speed) / (max_speed - min_speed)
    # test_scaled = (test_data - min_speed) / (max_speed - min_speed)
    
    train_scaled = train_data / max_speed    
    test_scaled = test_data / max_speed

    return train_scaled, test_scaled

train_scaled, test_scaled = scale_data(train_data, test_data)


seq_len = 7
pre_len = 2


def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY


trainX, trainY, testX, testY = sequence_data_preparation(
    seq_len, pre_len, train_scaled, test_scaled
)
print('TrainX', trainX.shape)
print('TrainY', trainY.shape)
print('TestX', testX.shape)
print('TestY', testY.shape)


from stellargraph.layer import GCN_LSTM


gcn_lstm = GCN_LSTM(
    seq_len=seq_len,
    adj=sensor_dist_adj,
    gc_layer_sizes=[16],
    gc_activations=["relu"],
    lstm_layer_sizes=[200],
    lstm_activations=["tanh"],
)

x_input, x_output = gcn_lstm.in_out_tensors()

model = Model(inputs=x_input, outputs=x_output)

model.compile(optimizer="adam", loss="mae", metrics=["mse"])

history = model.fit(
    trainX,
    trainY,
    epochs=1000,
    batch_size=32,
    shuffle=True,
    verbose=0,
    validation_data=(testX, testY),
)

model.summary()



print(
    "Train loss: ",
    history.history["loss"][-1],
    "\nTest loss:",
    history.history["val_loss"][-1],
)


model_name = 'gcn_lstm'

out = 'out_%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s' % datetime.now().strftime('%d-%m_%H-%M-%S')
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)



print(history.history.keys())

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.figure(figsize=(15,8))
plt.plot(history.history['loss'], color=colors[0], linestyle='-', label='treino')
plt.plot(history.history['val_loss'], color=colors[1], linestyle='-', label='validação')
plt.xlabel("Épocas", fontsize=12)
plt.ylabel("Erro médio absoluto", fontsize=12)
plt.legend(loc="best", fontsize=15)
plt.savefig(path+'/mae.eps', format='eps')
plt.show()



plt.figure(figsize=(15,8))
plt.plot(history.history['mse'], color=colors[0], linestyle='-', label='treino')
plt.plot(history.history['val_mse'], color=colors[1], linestyle='-', label='validação')
plt.xlabel("Épocas", fontsize=12)
plt.ylabel("Erro quadrático médio", fontsize=12)
plt.legend(loc="best", fontsize=15)
plt.savefig(path+'/mse.eps', format='eps')
plt.show()




# fig = sg.utils.plot_history(history=history, individual_figsize=(7, 7),return_figure=True)
# fig.savefig(path+'/loss_mse.eps', format='eps')



ythat = model.predict(trainX)
yhat = model.predict(testX)


## Rescale values
max_speed = train_data.max()
min_speed = train_data.min()

## actual train and test values
train_rescref = np.array(trainY * max_speed)
test_rescref = np.array(testY * max_speed)

## Rescale model predicted values
train_rescpred = np.array((ythat) * max_speed)
test_rescpred = np.array((yhat) * max_speed)



## Naive prediction benchmark (using previous observed value)

testnpred = np.array(testX)[
    :, :, -1
]  # picking the last speed of the 10 sequence for each segment in each sample
testnpredc = (testnpred) * max_speed


## Performance measures

seg_mael = []
seg_masel = []
seg_nmael = []

for j in range(testX.shape[-1]):

    seg_mael.append(
        np.mean(np.abs(test_rescref.T[j] - test_rescpred.T[j]))
    )  # Mean Absolute Error for NN
    seg_nmael.append(
        np.mean(np.abs(test_rescref.T[j] - testnpredc.T[j]))
    )  # Mean Absolute Error for naive prediction
    if seg_nmael[-1] != 0:
        seg_masel.append(
            seg_mael[-1] / seg_nmael[-1]
        )  # Ratio of the two: Mean Absolute Scaled Error
    else:
        seg_masel.append(np.NaN)

print("Total (ave) MAE for NN: " + str(np.mean(np.array(seg_mael))))
print("Total (ave) MAE for naive prediction: " + str(np.mean(np.array(seg_nmael))))
print(
    "Total (ave) MASE for per-segment NN/naive MAE: "
    + str(np.nanmean(np.array(seg_masel)))
)
print(
    "...note that MASE<1 (for a given segment) means that the NN prediction is better than the naive prediction."
)




# plot violin plot of MAE for naive and NN predictions
fig, ax = plt.subplots()
# xl = minsl

ax.violinplot(
    list(seg_mael), showmeans=True, showmedians=False, showextrema=False, widths=1.0
)

ax.violinplot(
    list(seg_nmael), showmeans=True, showmedians=False, showextrema=False, widths=1.0
)

line1 = mlines.Line2D([], [], label="NN")
line2 = mlines.Line2D([], [], color="C1", label="Instantaneous")

ax.set_xlabel("Scaled distribution amplitude (after Gaussian convolution)")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Distribution over segments: NN pred (blue) and naive pred (orange)")
plt.legend(handles=(line1, line2), title="Prediction Model", loc=2)
plt.savefig(path+'/nn_naivepred.eps', format='eps')
plt.show()




##all test result visualization
# fig1 = plt.figure(figsize=(15, 8))
#    ax1 = fig1.add_subplot(1,1,1)

plt.figure(figsize=(15,8))
a_pred = test_rescpred[:, 0]
a_true = test_rescref[:, 0]

plt.xlim([-(a_pred.shape[0]*0.02), a_pred.shape[0]+(a_pred.shape[0]*0.02)])

xticks = np.arange(0, a_pred.shape[0], 20)
xticks = np.append(xticks, a_pred.shape[0])
plt.xticks(xticks, fontsize=13)

ylim = np.max(a_true)
yticks = np.arange(0, ylim, 10)
yticks = np.append(yticks, ylim)
plt.yticks(yticks, fontsize=13)


plt.plot(a_true, "b-", label="verdadeiro")
plt.plot(a_pred, "r-", label="predição")
plt.xlabel("Snapshots", fontsize=15)
plt.ylabel("Quantidade de pares", fontsize=15)
plt.legend(loc="best", fontsize=15)
plt.savefig(path+'/test_all.eps', format='eps')
plt.show()


with open(path+'/pred.txt', 'w') as file:
    for p in a_pred:
        file.write(str(p)+', ')