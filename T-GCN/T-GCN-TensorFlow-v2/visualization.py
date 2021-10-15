#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_result(test_result,test_label1,path):
    ##all test result visualization
    fig1 = plt.figure(figsize=(15,9))
    # ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[:,0]
    a_true = test_label1[:,0]
    print(a_pred)
    print(len(a_pred))
    print(a_pred.shape)
    print(a_true)
    print(len(a_true))
    print(a_true.shape)
    plt.plot(a_true,'b-',label='true')
    plt.plot(a_pred,'r-',label='prediction')
    plt.legend(loc='best',fontsize=10)
    plt.xlabel('Quantidade de janelas')
    plt.ylabel('Quantidade de pares')    
    plt.savefig(path+'/test_all.eps', format='eps')
    plt.show()
    ## oneday test result visualization
    fig1 = plt.figure(figsize=(15,9))
    # ax1 = fig1.add_subplot(1,1,1)
    a_pred = test_result[0:96,0]
    a_true = test_label1[0:96,0]
    plt.plot(a_true,'b-',label="true")
    plt.plot(a_pred,'r-',label="prediction")
    plt.legend(loc='best',fontsize=10)
    plt.xlabel('Quantidade de janelas')
    plt.ylabel('Quantidade de pares')
    plt.savefig(path+'/test_oneday.eps', format='eps')
    plt.show()
    
def plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path):
    ###train_rmse & test_rmse 
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(train_rmse, 'r-', label="train_rmse")
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/rmse.eps', format='eps')
    plt.show()
    #### train_loss & train_rmse
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(train_loss,'b-', label='train_loss')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_loss.eps', format='eps')
    plt.show()

    fig1 = plt.figure(figsize=(10,6))
    plt.plot(train_rmse,'b-', label='train_rmse')
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/train_rmse.eps', format='eps')
    plt.show()

    ### accuracy
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(test_acc, 'b-', label="test_acc")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_acc.eps', format='eps')
    plt.show()
    ### rmse
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(test_rmse, 'b-', label="test_rmse")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_rmse.eps', format='eps')
    plt.show()
    ### mae
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(test_mae, 'b-', label="test_mae")
    plt.legend(loc='best',fontsize=10)
    plt.savefig(path+'/test_mae.eps', format='eps')
    plt.show()


