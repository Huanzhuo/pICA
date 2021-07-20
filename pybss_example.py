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
    dataset_id = 3  # i-th input data
    node_num = None

    fr = open('dataset/saxsNew.pkl', 'rb')
    saxs = pickle.load(fr)
    ss, aa, xx = saxs
    fr.close()
    # W = np.load("dataset/W.npy")
    W = np.ones((4, 4))*0.25

    # Evaluation setup
    test_num = 30  # repeat number
    eval_type = 'psnr'
    node_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    extraction_num = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    for node in np.arange(0, len(node_num), 1):
        Eval_dB = 0
        node_id = node_num[node]
        for i in range(test_num):
            print('*** N_test:', i+1, ' with ', node_id, ' nodes.')
            pybss_tb.timer_start()
            i = dataset_id
            S, A, X = ss[dataset_id].copy(), aa[dataset_id].copy(
            ), xx[dataset_id].copy().astype(np.float32)
            hat_S = picalite.pica(X, init_ext_interval=125, dynamic_adj_coef=2, tol=0.0001,
                                  grad_var_tol=0.90, fun='logcosh', max_iter=200, w_init=W, node_num=extraction_num[node])
            pybss_tb.timer_suspend()
            Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
            pybss_tb.timer_resume()
            time = pybss_tb.timer_value()
            measure_write('pICA_'+str(node_num[node]), [
                          'vnf', node_num[node], 'separation_accuracy', Eval_dB, 'separation_time', time])
            print('pICA    *** separation accuracy (dB): ' +
                  str(Eval_dB) + ', separation time (ms): ' + str(time))

            pybss_tb.timer_start()
            hat_S = picalite.fastica(
                X, tol=0.0001, fun='logcosh', max_iter=200, w_init=W)
            pybss_tb.timer_suspend()
            Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
            pybss_tb.timer_resume()
            time = pybss_tb.timer_value()
            measure_write('FastICA_'+str(node_num[node]), [
                          'vnf', node_num[node], 'separation_accuracy', Eval_dB, 'separation_time', time])
            print('FastICA *** separation accuracy (dB): ' +
                  str(Eval_dB) + ', separation time (ms): ' + str(time))
