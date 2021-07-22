import json
import numpy as np
import scipy.stats as st
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.use('TkAgg')

print(matplotlib.get_configdir())

if __name__ == '__main__':
    nodes = 3
    number_test = 0

    # fr = open('./emulator/MIMII/saxsNew.pkl', 'rb')
    # saxs = pickle.load(fr)
    # ss, aa, xx = saxs
    # s = ss[number_test]
    # x = xx[number_test]

    path_csv = 'measurement/pICA_'+str(nodes)+'details.csv'
    node_details = np.loadtxt(path_csv, delimiter=',', usecols=np.arange(
        1, nodes+2, 1))
    len_subset = node_details[0,:]
    process_latency = node_details[1,:]
    separation_accuracy = node_details[2, :]

    with plt.style.context(['science', 'ieee']):
        fig_width = 6.5
        barwidth = 0.15
        # bardistance = barwidth * 1.2
        colordict = {
            'compute_forward': '#0077BB',
            'store_forward': '#DDAA33',
            'darkblue': '#024B7A',
            'lightblue': '#3F9ABF',
            'midblue': '#7ACFE5'
        }
        markerdict = {
            'compute_forward': 'o',
            'store_forward': 'v',
            'store_forward_ia': 's'
        }

        plt.rcParams.update({'font.size': 11})

        fig = plt.figure(figsize=(fig_width, fig_width / 1.618))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(True, linestyle='--', which='major',
                      color='lightgrey', alpha=0.5, linewidth=0.2)
        x_index = np.arange(nodes + 1)
        bar1 = ax.bar(x_index-barwidth, len_subset/np.max(len_subset)*100, barwidth, fill=True,
                      color=colordict['darkblue'], edgecolor='#FFFFFF', ecolor='#555555', hatch='\\')
        bar2 = ax.bar(x_index, process_latency/sum(process_latency)*100, barwidth, fill=True,
                      color=colordict['lightblue'], edgecolor='#FFFFFF', ecolor='#555555', hatch='/')
        bar3 = ax.bar(x_index+barwidth, separation_accuracy/separation_accuracy[-1]*100, barwidth, fill=True,
                      color=colordict['midblue'], edgecolor='#FFFFFF', ecolor='#555555', hatch='//')
        
        # ax.set_xlabel(r'Number of nodes $k$')
        ax.set_ylabel(r'Percent in total cost ($\%$)')
        ax.set_yticks(np.arange(0, 101, 20))
        # ax.set_xlim([2, 60000])
        # ax.set_ylim([0, 201])
        ax.legend([bar1, bar2, bar3], [
            'Subset data size', 'Processing time', 'Processing precision'], loc='upper left', ncol=1)
        plt.xticks(x_index, ['Node 1', 'Node 2', 'Node 3', 'Remote \n Agent'], rotation=30)
        plt.savefig('./measurement/nodes_performance_simu.pdf',
                    dpi=600, bbox_inches='tight')
