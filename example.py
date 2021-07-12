from picalite  import *
from pybss_testbed import *
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
import pickle

if __name__ == '__main__': #'/Users/shenyunbin/Downloads/clean/audio' #

    fr = open('saxs10.pkl','rb')
    saxs = pickle.load(fr)
    ss,aa,xx = saxs
    fr.close()
    

    res = {}
    res['picalite_db'] = 0
    res['picalite_time'] = 0
    res['fastica_db'] = 0
    res['fastica_time'] = 0

    repeat_num = 10 #repeat number
    i = 1 #pick the i-th input data
    S,A,X = ss[i],aa[i],xx[i]
    eval_type = 'psnr'

    
    Eval_dB = 0
    pybss_tb.timer_start()
    for i in range(repeat_num):
        # S,A,X = ss[i],aa[i],xx[i]
        hat_S = picalite.pica(X, proc_mode='precise', init_ext_interval=4000, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200)
        pybss_tb.timer_suspend()
        Eval_dB += pybss_tb.bss_evaluation(S, hat_S, eval_type)
        pybss_tb.timer_resume()
    time = pybss_tb.timer_value()
    res['picalite_db'] += Eval_dB
    res['picalite_time'] += time

    Eval_dB = 0
    pybss_tb.timer_start()
    transformer = FastICA()
    X = X.T
    for i in range(repeat_num):
        # S,A,X = ss[i],aa[i],xx[i]
        hat_S = transformer.fit_transform(X)
        pybss_tb.timer_suspend()
        hat_S = hat_S.T
        Eval_dB += pybss_tb.bss_evaluation(S, hat_S, eval_type)
        pybss_tb.timer_resume()
    time = pybss_tb.timer_value()
    res['fastica_db'] += Eval_dB
    res['fastica_time'] += time


    
    print('type        eval_dB            time(ms) for       ' +
            str(4)+' sources, ' + str(0) + '-th test.')
    print('---------------------------------------------------------------------------')
    
    print('picalite: ', res['picalite_db'], '; ', res['picalite_time'])
    print('fastica: ', res['fastica_db'], '; ', res['fastica_time'])
