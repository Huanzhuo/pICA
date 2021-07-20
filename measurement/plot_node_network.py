import json
import numpy as np
import scipy.stats as st

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

matplotlib.use('TkAgg')

print(matplotlib.get_configdir())


def get_conf_interval(index, data, conf_rate):
    data_stat = []
    # index = data[:, 0].astype(int)
    for i in range(len(index)):
        conf_interval_low, conf_interval_high = st.t.interval(conf_rate, len(
            data[i, :])-1, loc=np.mean(data[i, :]), scale=st.sem(data[i, :]))
        conf_mean = np.mean(data[i, :])
        data_stat.append([index[i], conf_interval_low,
                         conf_mean, conf_interval_high])
    return np.array(data_stat)


if __name__ == '__main__':
    number_node = [0, 1, 2, 3, 4, 5, 6, 7]
    conf_rate = 0.95
    number_test = 50

    process_latency_cf = np.zeros(number_test)
    process_latency_sf = np.zeros(number_test)
    separation_accuracy_cf = np.zeros(number_test)
    separation_accuracy_sf = np.zeros(number_test)
    for node in number_node:
        path_time_compute_client = 'measurement/pICA_'+str(node)+'.csv'
        path_time_store_client = 'measurement/FastICA_'+str(node)+'.csv'
        compute_forward = np.loadtxt(
            path_time_compute_client, delimiter=',', usecols=[3, 5])
        store_forward = np.loadtxt(
            path_time_store_client, delimiter=',', usecols=[3, 5])

        process_latency_cf = np.row_stack(
            (process_latency_cf, compute_forward[:, 1]))
        process_latency_sf = np.row_stack(
            (process_latency_sf, store_forward[:, 1]))
        separation_accuracy_cf = np.row_stack(
            (separation_accuracy_cf, compute_forward[:, 0]))
        separation_accuracy_sf = np.row_stack(
            (separation_accuracy_sf, store_forward[:, 0]))

    process_latency_cf = process_latency_cf[1:, :]/1000
    process_latency_sf = process_latency_sf[1:, :]/1000
    separation_accuracy_cf = separation_accuracy_cf[1:, :]
    separation_accuracy_sf = separation_accuracy_sf[1:, :]

    tp_cf_conf = get_conf_interval(number_node, process_latency_cf, conf_rate)
    tp_sf_conf = get_conf_interval(number_node, process_latency_sf, conf_rate)
    db_cf_conf = get_conf_interval(
        number_node, separation_accuracy_cf, conf_rate)
    db_sf_conf = get_conf_interval(
        number_node, separation_accuracy_sf, conf_rate)

    # codes for plot figures
    with plt.style.context(['science', 'ieee']):
        fig_width = 6.5
        # barwidth = 0.15
        # bardistance = barwidth * 1.2
        colordict = {
            'compute_forward': '#0077BB',
            'store_forward': '#DDAA33',
            'store_forward_ia': '#009988',
            'orange': '#EE7733',
            'red': '#993C00',
            'blue': '#3340AD'
        }
        markerdict = {
            'compute_forward': 'o',
            'store_forward': 'v',
            'store_forward_ia': 's'
        }
        barwidth = 0.3

        plt.rcParams.update({'font.size': 11})

        fig = plt.figure(figsize=(fig_width, fig_width / 1.618))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(True, linestyle='--', which='major',
                      color='lightgrey', alpha=0.5, linewidth=0.2)
        x_index = np.arange(len(number_node))
        line1 = ax.errorbar(
            x_index, tp_cf_conf[:, 2], color=colordict['compute_forward'], lw=1, ls='-', marker=markerdict['compute_forward'], ms=5)
        line1_fill = ax.fill_between(x_index, tp_cf_conf[:, 1],
                                     tp_cf_conf[:, 3], color=colordict['compute_forward'], alpha=0.2)
        line2 = ax.errorbar(
            x_index, tp_sf_conf[:, 2], color=colordict['store_forward'], lw=1, ls='-', marker=markerdict['store_forward'], ms=5)
        line2_fill = ax.fill_between(x_index, tp_sf_conf[:, 1],
                                     tp_sf_conf[:, 3], color=colordict['store_forward'], alpha=0.2)
        ax.set_xlabel(r'Number of nodes $k$')
        ax.set_ylabel(r'Process latency $t_p$ ($s$)')
        ax.set_yticks(np.arange(0, 0.251, 0.05))
        # ax.set_xlim([-0.2, 4.2])
        # ax.set_yticks(np.arange(0, 151, 30))
        ax.legend([line1, line2], ['pICA',
                                   'FastICA'], loc='upper right')
        plt.xticks(range(len(number_node)), number_node)
        plt.savefig('measurement/process_latency_simu.pdf',
                    dpi=600, bbox_inches='tight')

        fig = plt.figure(figsize=(fig_width, fig_width / 1.618))
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.grid(True, linestyle='--', which='major',
                      color='lightgrey', alpha=0.5, linewidth=0.2)
        x_index = np.arange(len(number_node))
        bar1 = ax.bar(x_index-barwidth/2, db_cf_conf[:, 2], barwidth, yerr=db_cf_conf[:, 3] - db_cf_conf[:, 2],
                      error_kw=dict(lw=1, capsize=2, capthick=1), fill=True, color=colordict['compute_forward'], edgecolor='#FFFFFF', ecolor='#555555', hatch='\\')
        bar2 = ax.bar(x_index+barwidth/2, db_sf_conf[:, 2], barwidth, yerr=db_sf_conf[:, 3] - db_sf_conf[:, 2],
                      error_kw=dict(lw=1, capsize=2, capthick=1), fill=True, color=colordict['store_forward'], edgecolor='#FFFFFF', ecolor='#555555', hatch='/')

        ax.set_xlabel(r'Number of nodes $k$')
        ax.set_ylabel(r'Processing precision SDR ($dB$)')
        ax.set_yticks(np.arange(0, 41, 5))
        # ax.set_xlim([2, 60000])
        # ax.set_ylim([0, 201])
        ax.legend([bar1, bar2], [
            'pICA', 'FastICA'], loc='upper right', ncol=1)
        plt.xticks(range(len(number_node)), number_node)
        plt.savefig('measurement/process_accuracy_simu.pdf',
                    dpi=600, bbox_inches='tight')
