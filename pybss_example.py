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
    dataset_id = 3 # i-th input data
    fr = open('dataset/saxsNew.pkl','rb')
    saxs = pickle.load(fr)
    ss,aa,xx = saxs
    fr.close()
    W = np.load("dataset/W.npy")
    W = np.ones((4,4)) * 0.25
    # W = np.random.random((4,4))

    # Evaluation setup
    test_num = 1 #repeat number
    eval_type = 'psnr'

    # res = {}
    # res['picalite_db'] = 0
    # res['picalite_time'] = 0
    # res['fastica_db'] = 0
    # res['fastica_time'] = 0

    Eval_dB = 0
    for i in range(test_num):
        print('*** N_test:', i+1)
        pybss_tb.timer_start()
        i = dataset_id
        S,A,X = ss[dataset_id].copy(),aa[dataset_id].copy(),xx[dataset_id].copy().astype(np.float32)
        hat_S = picalite.pica(X, init_ext_interval=125, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200, w_init=W)
        # hat_S = picalite.pica(X, init_ext_interval=4000, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200)
        pybss_tb.timer_suspend()
        Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
        pybss_tb.timer_resume()
        time = pybss_tb.timer_value()
        measure_write('pICA_'+str(2), ['separation_accuracy', Eval_dB, 'separation_time', time])
        print('pICA *** separation accuracy (dB): '+ str(Eval_dB) + ', separation time (ms): ' + str(time))


    for i in range(test_num):
        print('*** N_test:', i+1)
        pybss_tb.timer_start()
        i = dataset_id
        S,A,X = ss[dataset_id].copy(),aa[dataset_id].copy(),xx[dataset_id].copy()
        hat_S = picalite.fastica(X, tol=0.0001, fun='logcosh', max_iter=200, w_init=W)
        pybss_tb.timer_suspend()
        Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
        pybss_tb.timer_resume()
        time = pybss_tb.timer_value()
        measure_write('Picalite_FastICA_'+str(2), ['separation_accuracy', Eval_dB, 'separation_time', time])
        print('Picalite FastICA *** separation accuracy (dB): '+ str(Eval_dB) + ', separation time (ms): ' + str(time))


    # transformer = FastICA(w_init=W)
    # # transformer = FastICA()
    # for i in range(test_num):
    #     print('*** N_test:', i+1)
    #     i = dataset_id
    #     S,A,X = ss[dataset_id].copy(),aa[dataset_id].copy(),xx[dataset_id].copy()
    #     X = X.T
    #     pybss_tb.timer_start()
    #     hat_S = transformer.fit_transform(X)
    #     pybss_tb.timer_suspend()
    #     hat_S = hat_S.T
    #     Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
    #     pybss_tb.timer_resume()
    #     time = pybss_tb.timer_value()
    #     measure_write('Sklearn_FastICA_'+str(2), ['separation_accuracy', Eval_dB, 'separation_time', time])
    #     print('Sklearn FastICA *** separation accuracy (dB): '+ str(Eval_dB) + ', separation time (ms): ' + str(time))
