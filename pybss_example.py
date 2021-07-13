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
    W = np.load("dataset/W.npy")
    S,A,X = ss[dataset_id],aa[dataset_id],xx[dataset_id]
    
    # Evaluation setup
    test_num = 5 #repeat number
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
        # S,A,X = ss[i],aa[i],xx[i]
        # hat_S = picalite.pica(X, proc_mode='precise', init_ext_interval=4000, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200, w_init=W)
        hat_S = picalite.pica(X, proc_mode='precise', init_ext_interval=4000, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200)
        pybss_tb.timer_suspend()
        Eval_dB += pybss_tb.bss_evaluation(S, hat_S, eval_type)
        pybss_tb.timer_resume()
        time = pybss_tb.timer_value()
        measure_write('pICA_'+str(2), ['separation_accuracy', Eval_dB, 'separation_time', time])
        print('pICA *** separation accuracy (dB): '+ str(Eval_dB) + ', separation time (ms): ' + str(time))

        # res['picalite_db'] += Eval_dB
        # res['picalite_time'] += time

    # Eval_dB = 0
    # pybss_tb.timer_start()
    # # transformer = FastICA(w_init=W)
    # transformer = FastICA()
    # X = X.T
    # for i in range(test_num):
    #     # S,A,X = ss[i],aa[i],xx[i]
    #     hat_S = transformer.fit_transform(X)
    #     pybss_tb.timer_suspend()
    #     hat_S = hat_S.T
    #     Eval_dB += pybss_tb.bss_evaluation(S, hat_S, eval_type)
    #     pybss_tb.timer_resume()
    # time = pybss_tb.timer_value()
    # res['fastica_db'] += Eval_dB
    # res['fastica_time'] += time


    
    # print('type        eval_dB            time(ms) for       ' +
    #         str(4)+' sources, ' + str(i) + '-th test.')
    # print('---------------------------------------------------------------------------')
    
    # print('picalite: ', res['picalite_db'], '; ', res['picalite_time'])
    # print('fastica: ', res['fastica_db'], '; ', res['fastica_time'])
