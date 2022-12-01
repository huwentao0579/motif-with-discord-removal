import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matrixprofile as mp
from matplotlib import pyplot as plt

def plot_dataset(name):
    taxi = mp.datasets.load(name)
    x = range(len(taxi['data']))
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 7))
    ax.plot(x, taxi['data'], c='#99CCFF', label='ECG')
    plt.legend(fontsize=14)
    f = plt.gcf()
    f.savefig('./{}.png'.format(name))
    f.clear()

def haptics(name):
    with open('/Users/huwentao/Desktop/pythonProject/Datasets/Haptics/Haptics_TRAIN', 'r') as f:
        df = f.readlines()
    for i in range(len(df)):
        df[i] =np.array(df[i].split(','), float)
    tp = []
    for i in range(40):
        tp = np.concatenate([tp, df[i]])
    x = range(len(tp))
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20, 7))
    ax.plot(x, tp, c='#99CCFF', label='Haptics')
    plt.legend(fontsize=14)
    f = plt.gcf()
    f.savefig('./{}.png'.format(name))
    plt.show()

if __name__ == '__main__':
    data_name = 'nyc-taxi-anomalies'
    windows = 48
    taxi = mp.datasets.load(data_name)
    ts = taxi['data']
    profile = mp.compute(ts, windows=windows, n_jobs=-1)
    tp = mp.discover.motifs(profile, k=1)
    tp1 = mp.discover.discords(profile, k=1)
    # figures = mp.visualize(tp)
    # for i in range(4):
    #     figures[i].show()

    # Create a plot with three subplots
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(10, 6))
    ax.plot(np.array(range(10320)), tp1['data']['ts'], c='#99CCFF')

    discord = 10098
    x = np.arange(discord, discord + tp1['w'])
    y = tp1['data']['ts'][discord:discord + tp1['w']]
    x1 = np.arange(1932, 1932 + tp1['w'])
    x2 = np.arange(2604, 2604 + tp1['w'])
    z1 = tp1['data']['ts'][1932:1932 + tp1['w']]
    z2 = tp1['data']['ts'][2604:2604 + tp1['w']]
    ax.plot(x, y, c='#FF3300', label='Discord')
    ax.plot(x1, z1, c='#00CC00', label='Motif')
    ax.plot(x2, z2, c='#00CC00', label='Neighbor')
    plt.legend(fontsize=14)
    plt.show()
    # for discord in profile['discords']:
    #     x = np.arange(discord, discord + tp1['w'])
    #     y = tp1['data']['ts'][discord:discord + tp1['w']]
    #     x1 = np.arange(1230, 1230 + tp1['w'])
    #     x2 = np.arange(894, 894 + tp1['w'])
    #     z1 = tp1['data']['ts'][1230:1230 + tp1['w']]
    #     z2 = tp1['data']['ts'][894:894 + tp1['w']]
    #     ax.plot(x, y, c='#FF3300', label='Discord')
    #     ax.plot(x1, z1, c='#00CC00', label='Motif')
    #     ax.plot(x2, z2, c='#00CC00', label='Neighbor')
    #     plt.legend(fontsize=14)
    # f = plt.gcf()
    # f.savefig('./test.png')
    # plt.show()
    # f.clear()