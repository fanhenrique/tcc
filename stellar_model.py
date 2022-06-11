import os
import sys
import inspect
import urllib.request
from datetime import datetime
import shutil
import time

import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from stellargraph.layer import GCN_LSTM

import utils_predictions as utils

DEFAULT_TRAIN_RATE = 0.8
DEFAULT_SEQ_LEN = 4
DEFAULT_PRED_LEN = 1
DEFAULT_GCN_SIZE = 16
DEFAULT_LSTM_SIZE = 200
DEFAULT_EPOCHS = 500
DEFAULT_BATCH = 32

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'


def scale_data(train_data, test_data):
    
    max_speed = train_data.max()
    min_speed = train_data.min()

    train_scaled = (train_data - min_speed) / (max_speed - min_speed)
    test_scaled = (test_data - min_speed) / (max_speed - min_speed)
    
    # train_scaled = train_data / max_speed    
    # test_scaled = test_data / max_speed

    return train_scaled, test_scaled

def train_test_split(data, train_rate):
    
    time_len = data.shape[1]
    train_size = int(time_len * train_rate)
    train_data = np.array(data[:, :train_size])
    test_data = np.array(data[:, train_size:])
    return train_data, test_data


def sequence_data_preparation(train_data, test_data, seq_len, pred_len):

    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pred_len - 1)):
        a = train_data[:, i : i + seq_len + pred_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pred_len - 1)):
        b = test_data[:, i : i + seq_len + pred_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY


def plot_validation(train, validation, path_plots, name, title, ylabel):
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(15,8))
    plt.plot(train, color=colors[0], linestyle='-', label='treino')
    plt.plot(validation, color=colors[1], linestyle='-', label='validação')
    plt.xlabel("Épocas", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best", fontsize=15)
    plt.title(title)
    plt.savefig(path_plots+'/'+name+'.svg', format='svg')
    plt.savefig(path_plots+'/png/'+name+'.png', format='png')
    # plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description='gcn-lstm')

    # parser.add_argument('--plot', '-p', help='Plot mode', action='store_true')
    parser.add_argument('--trainrate', '-tr', help='Taxa de treinamento', default=DEFAULT_TRAIN_RATE, type=float)
    parser.add_argument('--seqlen', '-sl', help='Snapshots de entrada', default=DEFAULT_SEQ_LEN, type=int)
    parser.add_argument('--predlen', '-pl', help='Snapshots preditos', default=DEFAULT_PRED_LEN, type=int)    
    parser.add_argument('--gcnsize', '-gs', help='Tamanho da camada GCN', default=DEFAULT_GCN_SIZE, type=int)
    parser.add_argument('--lstmsize', '-ls', help='Tamanho da camada LSTM', default=DEFAULT_LSTM_SIZE, type=int)
    parser.add_argument('--batch', '-b', help='Tamanho do batch', default=DEFAULT_BATCH, type=int)
    parser.add_argument('--epochs', '-e', help='Épocas', default=DEFAULT_EPOCHS, type=int)
    parser.add_argument('--weigths', '-w', help='Matriz de pessos', required=True, type=str)
    parser.add_argument('--adjs', '-a', help='Matriz de adjacencias', required=True, type=str)
    parser.add_argument('--path', help='Date path', required=True, type=str)

    help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

    args = parser.parse_args()

    if args.log == logging.DEBUG:
        logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
    else:
        logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)

    d = dict(vars(args))
    del d['weigths']
    del d['adjs']
    del d['path']
    del d['log']
    path_outs, path_plots = utils.init('gcn-lstm', args.path, d)  

    # print(path_plots)
    # print(path_outs)

    # if args.plot:

        # print(path_outs, path_plots)

        # true = pd.read_csv(path_outs+'/true.csv', header=None).to_numpy()
        # predictions = pd.read_csv(path_outs+'/prediction.csv', header=None).to_numpy()
        # loss = pd.read_csv(path_outs+'/loss.csv', header=None).to_numpy()
        # val_loss = pd.read_csv(path_outs+'/val_loss.csv', header=None).to_numpy()
        # mse = pd.read_csv(path_outs+'/mse.csv', header=None).to_numpy()
        # val_mse = pd.read_csv(path_outs+'/val_mse.csv', header=None).to_numpy()
    
        # plot(true, predictions, loss, val_loss, mse, val_mse, path_plots)
    
    
    logging.info('Reading file ...')

    # los_adj = pd.read_csv(r'../T-GCN/data/los_adj.csv', header=None)
    # los_adj = pd.read_csv(r'../T-GCN/data/sz_adj.csv', header=None)
    los_adj = pd.read_csv(args.adjs, header=None)
    sensor_dist_adj = np.mat(los_adj)


    # los_speed = pd.read_csv(r'../T-GCN/data/los_speed.csv')
    # los_speed = pd.read_csv(r'../T-GCN/data/sz_speed.csv')
    los_speed = pd.read_csv(args.weigths, header=None)
    speed_data = np.mat(los_speed)


    sensor_dist_adj = sensor_dist_adj.transpose()
    speed_data = speed_data.transpose()


    num_nodes, time_len = speed_data.shape
    logging.info('Numero de trackers: %s', num_nodes)
    logging.info('Numero de snapshots: %s', time_len)

    max_y = np.max(speed_data.max())
    # print(max_y)

    train_data, test_data = train_test_split(speed_data, args.trainrate)
    logging.info('All data: %s Train Rate: %f' % (str(speed_data.shape), args.trainrate ))    
    logging.info('Train data: %s' % str(train_data.shape))
    logging.info('Test data: %s' % str(test_data.shape))

    
    logging.info('Normalization ...')
    train_scaled, test_scaled = scale_data(train_data, test_data)


    logging.info('Sequence preparation ...')
    trainX, trainY, testX, testY = sequence_data_preparation(train_scaled, test_scaled, args.seqlen, args.predlen)
    logging.info('TrainX: %s', trainX.shape)
    logging.info('TrainY: %s', trainY.shape)
    logging.info('TestX: %s', testX.shape)
    logging.info('TestY: %s', testY.shape)


    logging.info('Create RNA GCN_LSTM ...')
    gcn_lstm = GCN_LSTM(
        seq_len=args.seqlen,
        adj=sensor_dist_adj,
        gc_layer_sizes=[args.gcnsize],
        gc_activations=["relu"],
        lstm_layer_sizes=[args.lstmsize],
        lstm_activations=["tanh"],
    )

    x_input, x_output = gcn_lstm.in_out_tensors()

    model = Model(inputs=x_input, outputs=x_output)

    model.compile(optimizer="adam", loss="mae", metrics=["mse"])


    logging.info('Fit ...')
    times = []
    start_time = time.time()
    history = model.fit(
        trainX,
        trainY,
        epochs=args.epochs,
        batch_size=args.batch,
        shuffle=True,
        verbose=0,
        validation_data=(testX, testY),
    )

    end_time = time.time()

    times.append(end_time-start_time)

    np.savetxt(path_outs+'/times.csv', times, fmt='%.8f')

    # logging.info(model.summary())


    # print(
    #     "Train loss: ",
    #     history.history["loss"][-1],
    #     "\nTest loss:",
    #     history.history["val_loss"][-1],
    #     "\nTrain mse: ",
    #     history.history["mse"][-1],
    #     "\nTest mse:",
    #     history.history["val_mse"][-1],
    # )

    # print(history.history.keys())

    
    
    # fig = sg.utils.plot_history(history=history, individual_figsize=(7, 7),return_figure=True)
    # fig.savefig(path+'/loss_mse.eps', format='eps')


    logging.info('Denormalization ...')
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



    

    # ## Naive prediction benchmark (using previous observed value)

    # testnpred = np.array(testX)[
    #     :, :, -1
    # ]  # picking the last speed of the 10 sequence for each segment in each sample
    # testnpredc = (testnpred) * max_speed


    # ## Performance measures

    # seg_mael = []
    # seg_masel = []
    # seg_nmael = []

    # for j in range(testX.shape[-1]):

    #     seg_mael.append(
    #         np.mean(np.abs(test_rescref.T[j] - test_rescpred.T[j]))
    #     )  # Mean Absolute Error for NN
    #     seg_nmael.append(
    #         np.mean(np.abs(test_rescref.T[j] - testnpredc.T[j]))
    #     )  # Mean Absolute Error for naive prediction
    #     if seg_nmael[-1] != 0:
    #         seg_masel.append(
    #             seg_mael[-1] / seg_nmael[-1]
    #         )  # Ratio of the two: Mean Absolute Scaled Error
    #     else:
    #         seg_masel.append(np.NaN)

    # print("Total (ave) MAE for NN: " + str(np.mean(np.array(seg_mael))))
    # print("Total (ave) MAE for naive prediction: " + str(np.mean(np.array(seg_nmael))))
    # print(
    #     "Total (ave) MASE for per-segment NN/naive MAE: "
    #     + str(np.nanmean(np.array(seg_masel)))
    # )
    # print(
    #     "...note that MASE<1 (for a given segment) means that the NN prediction is better than the naive prediction."
    # )

    # # plot violin plot of MAE for naive and NN predictions
    # fig, ax = plt.subplots()
    # # xl = minsl

    # ax.violinplot(
    #     list(seg_mael), showmeans=True, showmedians=False, showextrema=False, widths=1.0
    # )

    # ax.violinplot(
    #     list(seg_nmael), showmeans=True, showmedians=False, showextrema=False, widths=1.0
    # )

    # line1 = mlines.Line2D([], [], label="NN")
    # line2 = mlines.Line2D([], [], color="C1", label="Instantaneous")

    # ax.set_xlabel("Scaled distribution amplitude (after Gaussian convolution)")
    # ax.set_ylabel("Mean Absolute Error")
    # ax.set_title("Distribution over segments: NN pred (blue) and naive pred (orange)")
    # plt.legend(handles=(line1, line2), title="Prediction Model", loc=2)
    # plt.savefig(path_plots+'/nn_naivepred.svg', format='svg')
    # plt.show()



    # a_pred = test_rescpred[:, 0]
    # a_true = test_rescref[:, 0]


    loss = history.history['loss']
    val_loss = history.history['val_loss']

    mse = history.history['mse']
    val_mse = history.history['val_mse']

    plot_validation(mse, val_mse, path_plots, 'validation_mse', None, 'Erro quadrático médio')
    plot_validation(loss, val_loss, path_plots, 'validation_loss', None, 'Erro')



    
    ### PREDICTION TEST ###
    for i in range(test_rescref.shape[1]):
        logging.info('Plot test prediction tracker %d' % i)
        utils.plot_prediction(test_rescref[:, i], test_rescpred[:, i], path_plots, 'prediction_test_'+str(i), 'Predição RNA tracker '+str(i)+' - Teste', max_y)
        np.savetxt(path_outs+'/prediction_test_'+str(i)+'.csv', test_rescpred[:, i])

    ### Média de todas as respostas da RNA prediction teste
    df_test_true = pd.DataFrame(test_rescref)
    df_test_pred = pd.DataFrame(test_rescpred)    
    
    df_test_true[speed_data.shape[0]] = test_rescref.mean(axis=1)
    df_test_pred[speed_data.shape[0]] = test_rescpred.mean(axis=1)

    # print(df_test_true)
    # print(df_test_pred)

    mean_test_true = df_test_true[speed_data.shape[0]].to_numpy()
    mean_test_pred = df_test_pred[speed_data.shape[0]].to_numpy()    

    logging.info('Plot prediction test mean')
    utils.plot_prediction(mean_test_true, mean_test_pred, path_plots, 'prediction_test_'+str(speed_data.shape[0]), 'Predição RNA trackers '+str(speed_data.shape[0])+' - Teste', max_y)
    np.savetxt(path_outs+'/prediction_test_'+str(speed_data.shape[0])+'.csv', mean_test_pred)
    


    ### PREDICTION ALL DATASET ###
    for i in range(train_rescref.shape[1]):
        logging.info('Plot all prediction tracker %d' % i)
        ref = np.concatenate((train_rescref[:, i], test_rescref[:, i]))
        pred = np.concatenate((train_rescpred[:, i], test_rescpred[:, i]))
        utils.plot_prediction(ref, pred, path_plots, 'prediction_all_'+str(i), 'Predição RNA tracker '+str(i)+' - S1', max_y)
        np.savetxt(path_outs+'/prediction_all_'+str(i)+'.csv', pred)

    ## Média de todas as respostas da RNA prediction all dataset
    df_train_true = pd.DataFrame(train_rescref)
    df_train_pred = pd.DataFrame(train_rescpred)    
    
    df_train_true[speed_data.shape[0]] = train_rescref.mean(axis=1)
    df_train_pred[speed_data.shape[0]] = train_rescpred.mean(axis=1)

    # print(df_train_true)
    # print(df_train_pred)

    mean_train_true = df_train_true[speed_data.shape[0]].to_numpy()
    mean_train_pred = df_train_pred[speed_data.shape[0]].to_numpy()    

    ref = np.concatenate((mean_train_true, mean_test_true))
    pred = np.concatenate((mean_train_pred, mean_test_pred)) 

    logging.info('Plot prediction all mean')
    utils.plot_prediction(ref, pred, path_plots, 'prediction_all_'+str(speed_data.shape[0]), 'Predição RNA trackers '+str(speed_data.shape[0])+' - S1', max_y)
    np.savetxt(path_outs+'/prediction_all_'+str(speed_data.shape[0])+'.csv', pred)




    df_mse = pd.DataFrame()
    for i in range(test_rescref.shape[1]):
        logging.info('Calculate MSE tracker %d' % i)
        mse = np.array(utils.mean_squared_error(test_rescref[:, i], test_rescpred[:, i]))
        df_mse[i] = mse

    df_mse[df_mse.shape[1]] = np.array(utils.mean_squared_error(mean_test_true, mean_test_pred))



    logging.info('Plots MSEs')

    max_y = np.max(df_mse.max())
    mean_mse = []
    for column in df_mse:
        utils.plot_mse(df_mse[column], path_plots, 'mse_'+str(column), 'Erro Quadrático Médio RNA tracker '+str(column)+' - Teste', max_y)
        np.savetxt(path_outs+'/mse_'+str(column)+'.csv', df_mse[column], fmt='%.8f')
        mean_mse.append(np.mean(df_mse[column]))                

    np.savetxt(path_outs+'/mean.csv', mean_mse, fmt='%.8f')

        
    logging.info('plots directory: %s' % path_plots)
    logging.info('outputs directory: %s' % path_outs)


    # np.savetxt(path_outs+'/loss.csv', loss)
    # np.savetxt(path_outs+'/val_loss.csv', val_loss)

    # np.savetxt(path_outs+'/mse.csv', mse)
    # np.savetxt(path_outs+'/val_mse.csv', val_mse)

    # np.savetxt(path_outs+'/true.csv', a_true)
    # np.savetxt(path_outs+'/prediction.csv', a_pred)


if __name__ == '__main__':
    main()