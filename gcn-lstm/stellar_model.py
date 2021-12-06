import os
import sys
import inspect
import urllib.request
from datetime import datetime
import shutil

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


TRAIN_RATE = 0.8
SEQ_LEN = 7
PRE_LEN = 2

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

def train_test_split(data):
    
    time_len = data.shape[1]
    train_size = int(time_len * TRAIN_RATE)
    train_data = np.array(data[:, :train_size])
    test_data = np.array(data[:, train_size:])
    return train_data, test_data


def sequence_data_preparation(train_data, test_data):

    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(SEQ_LEN + PRE_LEN - 1)):
        a = train_data[:, i : i + SEQ_LEN + PRE_LEN]
        trainX.append(a[:, :SEQ_LEN])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(SEQ_LEN + PRE_LEN - 1)):
        b = test_data[:, i : i + SEQ_LEN + PRE_LEN]
        testX.append(b[:, :SEQ_LEN])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY


def init():
    
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))

    path_plots = path + '/plots'
    path_data = '%s' % datetime.now().strftime('%m-%d_%H-%M-%S')
    path_plots = os.path.join(path_plots, path_data)
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)

    path_outs = path + '/outs'
    if not os.path.exists(path_outs):
        os.makedirs(path_outs)
    else:
        shutil.rmtree(path_outs)
        os.makedirs(path_outs)
    
    return path_outs, path_plots

def mean_squared_error(data, prediction):
    
    return [((prediction[i]-data[i])**2)/len(data) for i in range(len(data))]


def plot_mse(column, mse, path_plots, max_y):
    
    ## MSE TEST ##
    
    plt.figure(figsize=(15,8))

    plt.xlim([-(mse.shape[0]*0.02), mse.shape[0]+(mse.shape[0]*0.02)])

    xticks = np.arange(0, mse.shape[0], mse.shape[0]*0.1)
    xticks = np.append(xticks, mse.shape[0])
    plt.xticks(xticks, fontsize=13)


    plt.ylim(0, max_y+max_y*0.02)

    yticks = np.arange(0, max_y, max_y*0.1)
    yticks = np.append(yticks, max_y)       
    plt.yticks(yticks, fontsize=13)


    plt.plot(mse, 'y-', label='mse')
    plt.ylabel('mean squared error', fontsize=12)
    plt.title('RNA')
    plt.savefig(path_plots+'/mse_'+str(column)+'.svg', format='svg')
    # plt.show()


def plot_prediction(column, true, prediction, path_plots, name, max_y):

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
    plt.ylabel("Média da quantidade de pares", fontsize=15)
    plt.legend(loc="best", fontsize=15)
    plt.title('Predição RNA - Teste')
    plt.savefig(path_plots+'/'+name+'_'+str(column)+'.svg', format='svg')
    # plt.show()

def plot_mse_validation(mse, val_mse, path_plots):
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(15,8))
    plt.plot(mse, color=colors[0], linestyle='-', label='treino')
    plt.plot(val_mse, color=colors[1], linestyle='-', label='validação')
    plt.xlabel("Épocas", fontsize=12)
    plt.ylabel("Erro quadrático médio", fontsize=12)
    plt.legend(loc="best", fontsize=15)
    plt.savefig(path_plots+'/mse_validation.svg', format='svg')
    # plt.show()



# def plot(test_true, test_pred, loss, val_loss, mse, val_mse, path_plots):

#     colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

#     plt.figure(figsize=(15,8))
#     plt.plot(loss, color=colors[0], linestyle='-', label='treino')
#     plt.plot(val_loss, color=colors[1], linestyle='-', label='validação')
#     plt.xlabel("Épocas", fontsize=12)
#     plt.ylabel("Erro médio absoluto", fontsize=12)
#     plt.legend(loc="best", fontsize=15)
#     plt.savefig(path_plots+'/mae.svg', format='svg')
#     plt.show()

#     plt.figure(figsize=(15,8))
#     plt.plot(mse, color=colors[0], linestyle='-', label='treino')
#     plt.plot(val_mse, color=colors[1], linestyle='-', label='validação')
#     plt.xlabel("Épocas", fontsize=12)
#     plt.ylabel("Erro quadrático médio", fontsize=12)
#     plt.legend(loc="best", fontsize=15)
#     plt.savefig(path_plots+'/mse.svg', format='svg')
#     plt.show()


#     # plt.figure(figsize=(15,8))
#     # plt.xlim([-(a_pred.shape[0]*0.02), a_pred.shape[0]+(a_pred.shape[0]*0.02)])

#     # xticks = np.arange(0, a_pred.shape[0], 20)
#     # xticks = np.append(xticks, a_pred.shape[0])
#     # plt.xticks(xticks, fontsize=13)

#     # # ylim = np.max(a_true)
#     # # yticks = np.arange(0, ylim, 10)
#     # # yticks = np.append(yticks, ylim)
#     # # plt.yticks(yticks, fontsize=13)

#     # plt.plot(a_true, "b-", label="verdadeiro")
#     # plt.plot(a_pred, "r-", label="predição")
#     # plt.xlabel("Snapshots", fontsize=15)
#     # plt.ylabel("Quantidade de pares", fontsize=15)
#     # plt.legend(loc="best", fontsize=15)
#     # plt.ylim(0, 60)
#     # plt.savefig(path_plots+'/test_all.svg', format='svg')
#     # plt.show()



#     df_test_true = pd.DataFrame(test_true)
#     df_test_pred = pd.DataFrame(test_pred)    
        
#     df_test_true['mean'] = test_true.mean(axis=1)
#     df_test_pred['mean'] = test_pred.mean(axis=1)

#     mean_true = df_test_true['mean'].to_numpy()
#     mean_pred = df_test_pred['mean'].to_numpy()



#     ## PREDICTION TEST ##
#     plt.figure(figsize=(15,8))

#     plt.xlim([-(mean_true.shape[0]*0.02), mean_true.shape[0]+(mean_true.shape[0]*0.02)])

#     xticks = np.arange(0, mean_true.shape[0], 20)
#     xticks = np.append(xticks, mean_true.shape[0])
#     plt.xticks(xticks, fontsize=13)

#     plt.plot(mean_true, 'b-', label='verdadeiro')
#     plt.plot(mean_pred, 'r-', label='predição')
#     plt.xlabel("Snapshots", fontsize=15)
#     plt.ylabel("Média dos resultados das predições", fontsize=15)
#     plt.ylim(0, 60)    
#     plt.legend(loc="best", fontsize=15)
#     plt.title('Predição RNA - Teste')
#     plt.savefig(path_plots+'/prediction_test.svg', format='svg')
#     plt.show()





#     for i in range(test_true.shape[1]):
#         plt.figure(figsize=(15,8))
#         plt.xlim([-(test_pred[:, i].shape[0]*0.02), test_pred[:, i].shape[0]+(test_pred[:, i].shape[0]*0.02)])

#         xticks = np.arange(0, test_pred[:, i].shape[0], 20)
#         xticks = np.append(xticks, test_pred[:, i].shape[0])
#         plt.xticks(xticks, fontsize=13)

#         # ylim = np.max(a_true)
#         # yticks = np.arange(0, ylim, 10)
#         # yticks = np.append(yticks, ylim)
#         # plt.yticks(yticks, fontsize=13)

#         plt.plot(test_true[:, i], "b-", label="verdadeiro")
#         plt.plot(test_pred[:, i], "r-", label="predição")
#         plt.xlabel("Snapshots", fontsize=15)
#         plt.ylabel("Quantidade de pares", fontsize=15)
#         plt.legend(loc="best", fontsize=15)
#         plt.title('Predição RNA - Teste Tracker '+str(i))
#         plt.savefig(path_plots+'/prediction_test'+str(i)+'.svg', format='svg')
#         plt.show()




def main():


    parser = argparse.ArgumentParser(description='gcn_lstm')

    parser.add_argument('--plot', '-p', help='plot mode', action='store_true')

    help_msg = "Logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
    parser.add_argument("--log", "-l", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

    args = parser.parse_args()

    if args.log == logging.DEBUG:
        logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt=TIME_FORMAT, level=args.log)
    else:
        logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt=TIME_FORMAT, level=args.log)


    path_outs, path_plots = init()    

    print(path_plots)

    if args.plot:

        # print(path_outs, path_plots)

        true = pd.read_csv(path_outs+'/true.csv', header=None).to_numpy()
        predictions = pd.read_csv(path_outs+'/prediction.csv', header=None).to_numpy()
        loss = pd.read_csv(path_outs+'/loss.csv', header=None).to_numpy()
        val_loss = pd.read_csv(path_outs+'/val_loss.csv', header=None).to_numpy()
        mse = pd.read_csv(path_outs+'/mse.csv', header=None).to_numpy()
        val_mse = pd.read_csv(path_outs+'/val_mse.csv', header=None).to_numpy()
        

        plot(true, predictions, loss, val_loss, mse, val_mse, path_plots)
    
    else:


        # los_adj = pd.read_csv(r'../T-GCN/data/los_adj.csv', header=None)
        # los_adj = pd.read_csv(r'../T-GCN/data/sz_adj.csv', header=None)

        los_adj = pd.read_csv('../out/out-matrices/monitoring-adj.csv', header=None)
        sensor_dist_adj = np.mat(los_adj)


        # los_speed = pd.read_csv(r'../T-GCN/data/los_speed.csv')
        #los_speed = pd.read_csv(r'../T-GCN/data/sz_speed.csv')

        los_speed = pd.read_csv('../out/out-matrices/monitoring-weigths.csv', header=None)
        speed_data = np.mat(los_speed)


        sensor_dist_adj = sensor_dist_adj.transpose()
        speed_data = speed_data.transpose()


        num_nodes, time_len = speed_data.shape
        print("No. of sensors:", num_nodes, "\nNo of timesteps:", time_len)

    
        max_y = np.max(speed_data.max())

        # print(max_y)

        train_data, test_data = train_test_split(speed_data)
        print("Train data: ", train_data.shape)
        print("Test data: ", test_data.shape)

        

        train_scaled, test_scaled = scale_data(train_data, test_data)


        trainX, trainY, testX, testY = sequence_data_preparation(train_scaled, test_scaled)
        print('TrainX', trainX.shape)
        print('TrainY', trainY.shape)
        print('TestX', testX.shape)
        print('TestY', testY.shape)



        gcn_lstm = GCN_LSTM(
            seq_len=SEQ_LEN,
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
            epochs=500,
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
            "\nTrain mse: ",
            history.history["mse"][-1],
            "\nTest mse:",
            history.history["val_mse"][-1],
        )

        # print(history.history.keys())

        
        
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


        # ## Rescale model predicted values
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
        plot_mse_validation(mse, val_mse, path_plots)




        ### Respostas da RNA ###
        for i in range(test_rescref.shape[1]):
            plot_prediction(i, test_rescref[:, i], test_rescpred[:, i], path_plots, 'prediction_test', max_y)
            np.savetxt(path_outs+'/prediction_'+str(i)+'.csv', test_rescpred[:, i])




        ### Média de todas as respostas da RNA ###   
        df_test_true = pd.DataFrame(test_rescref)
        df_test_pred = pd.DataFrame(test_rescpred)    
            
        df_test_true[speed_data.shape[0]] = test_rescref.mean(axis=1)
        df_test_pred[speed_data.shape[0]] = test_rescpred.mean(axis=1)

        mean_true = df_test_true[speed_data.shape[0]].to_numpy()
        mean_pred = df_test_pred[speed_data.shape[0]].to_numpy()    


        plot_prediction(speed_data.shape[0], mean_true, mean_pred, path_plots, 'prediction_test', max_y)
        np.savetxt(path_outs+'/prediction_'+str(speed_data.shape[0])+'.csv', mean_pred)




        df_mse = pd.DataFrame()
        for i in range(test_rescref.shape[1]):
            mse = np.array(mean_squared_error(test_rescref[:, i], test_rescpred[:, i]))
            
            df_mse[i] = mse
        df_mse[df_mse.shape[0]] = np.array(mean_squared_error(mean_true, mean_pred))





        max_y = np.max(df_mse.max())

        print(max_y)



        mean_mse = []
        for column in df_mse:

            plot_mse(column, df_mse[column], path_plots, max_y)
            np.savetxt(path_outs+'/mse_'+str(i)+'.csv', mse)

            # mean_mse.append(np.mean(df_mse[column]))                
        
            # mean_mse.append(np.mean(mse))

        

        # mse = 
 
        # np.savetxt(path_outs+'/mse_mean.csv', mse)

        # plot_mse('mean', mse, path_plots)

        mean_mse.append(np.mean(mse))


        np.savetxt(path_outs+'/mean_mse.csv', mean_mse)

            



        # np.savetxt(path_outs+'/loss.csv', loss)
        # np.savetxt(path_outs+'/val_loss.csv', val_loss)

        # np.savetxt(path_outs+'/mse.csv', mse)
        # np.savetxt(path_outs+'/val_mse.csv', val_mse)


        # np.savetxt(path_outs+'/true.csv', a_true)
        # np.savetxt(path_outs+'/prediction.csv', a_pred)

    


if __name__ == '__main__':
    main()