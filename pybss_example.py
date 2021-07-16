from pybss_core import *
from pybss_testbed import *
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from measurement.measure import measure_write
import pickle

if __name__ == '__main__':

    # Load input data: S, A, and W_0
    dataset_id = 2 # i-th input data
    fr = open('dataset/saxs10.pkl','rb')
    saxs = pickle.load(fr)
    ss,aa,xx = saxs
    fr.close()
    S,A,X = ss[dataset_id].copy(),aa[dataset_id].copy(),xx[dataset_id].copy()
    W = np.load("dataset/W.npy")
    # W = np.random.random((4,4))

    # Evaluation setup
    test_num = 40 #repeat number
    eval_type = 'sdr'
    node_num = 6

    Eval_dB = 0
    for i in range(test_num):
        print('*** N_test: ', i+1, ' with ', node_num, ' network nodes.')
        pybss_tb.timer_start()
        hat_S = picalite.pica(X, init_ext_interval=4000, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200, w_init=W, node_num=node_num)
        # hat_S = picalite.pica(X, init_ext_interval=4000, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200)
        pybss_tb.timer_suspend()
        Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
        pybss_tb.timer_resume()
        time = pybss_tb.timer_value()
        measure_write('pICA_'+str(node_num), ['vnf', node_num, 'separation_accuracy', Eval_dB, 'separation_time', time])
        print('pICA    *** separation accuracy (dB): '+ str(Eval_dB) + ', separation time (ms): ' + str(time))

        pybss_tb.timer_start()
        hat_S = picalite.fastica(X, tol=0.0001, fun='logcosh', max_iter=200, w_init=W)
        pybss_tb.timer_suspend()
        Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
        pybss_tb.timer_resume()
        time = pybss_tb.timer_value()
        measure_write('FastICA_'+str(node_num), ['vnf', node_num, 'separation_accuracy', Eval_dB, 'separation_time', time])
        print('FastICA *** separation accuracy (dB): '+ str(Eval_dB) + ', separation time (ms): ' + str(time))