import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


TRAIN_RATE = 0.8

def train_test_split(data):
    time_len = data.shape[1]
    train_size = int(time_len * TRAIN_RATE)
    train_data = np.array(data[:, :train_size])
    test_data = np.array(data[:, train_size:])
    return train_data, test_data

def train_test_split_vector(data):
    time_len = data.size
    train_size = int(time_len * TRAIN_RATE)
    train_data = np.array(data[:train_size])
    test_data = np.array(data[train_size:])
    return train_data, test_data


def main():

    df = pd.read_csv('out/out-matrices/monitoring-weigths.csv', header=None)

    print(df)
    print(df.shape)

    data = np.mat(df)

    print(data)
    print(data.shape)
    
    dataTranspose = data.transpose()

    print(dataTranspose)
    print(dataTranspose.shape)

    train_data, test_data = train_test_split(dataTranspose)
    print("Train data: ", train_data.shape)
    print("Test data: ", test_data.shape)

    # exit()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    plt.figure(figsize=(15,8))
    for i in range(df.shape[1]):
        plt.plot(df[i], color=colors[i], label='t'+str(i))
    plt.xlabel("Snapshots", fontsize=12)
    plt.ylabel("Quantidade de peers por tracker", fontsize=12)
    plt.legend(loc="best", fontsize=15)
    plt.ylim(0, 100)
    plt.show()


    df['mean'] = df.mean(axis=1)
    mean = df['mean'].to_numpy()

    plt.figure(figsize=(15,8))
    plt.plot(mean, 'r-')
    plt.xlabel("Snapshots", fontsize=12)
    plt.ylabel("Média de peers por tracker", fontsize=12)
    plt.ylim(0, 100)
    plt.show() 


    plt.figure(figsize=(15,8))
    for i in range(test_data.shape[0]):
        plt.plot(test_data[i], color=colors[i], label='t'+str(i))
    plt.xlabel("Snapshots", fontsize=12)
    plt.ylabel("Quantidade de peers por tracker", fontsize=12)
    plt.legend(loc="best", fontsize=15)
    plt.title('TESTE')
    plt.ylim(0, 100)
    plt.show()



    mean_train, mean_test = train_test_split_vector(df['mean'].to_numpy())

    plt.figure(figsize=(15,8))
    plt.plot(mean_test, 'r-')
    plt.xlabel("Snapshots", fontsize=12)
    plt.ylabel("Média de peers por tracker", fontsize=12)
    plt.title('TESTE')
    plt.ylim(0, 60)
    plt.show() 



if __name__ == '__main__':
    main()
       
