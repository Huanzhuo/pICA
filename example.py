from picalite  import *
from pybss_testbed import *
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA

if __name__ == '__main__': #'/Users/shenyunbin/Downloads/clean/audio' #
    addrs = [0]*100

    addrs[0] = '/Volumes/BINWORK/VoiceDataset/MIMII/pump/id_00/normal'
    addrs[1] = '/Volumes/BINWORK/VoiceDataset/MIMII/fan/id_00/normal'
    addrs[2] = '/Volumes/BINWORK/VoiceDataset/MIMII/slider/id_00/normal'
    addrs[3] = '/Volumes/BINWORK/VoiceDataset/MIMII/valve/id_00/normal'

    addrs[4] = '/Volumes/BINWORK/VoiceDataset/MIMII/pump/id_02/normal'
    addrs[5] = '/Volumes/BINWORK/VoiceDataset/MIMII/fan/id_02/normal'
    addrs[6] = '/Volumes/BINWORK/VoiceDataset/MIMII/slider/id_02/normal'
    addrs[7] = '/Volumes/BINWORK/VoiceDataset/MIMII/valve/id_02/normal'

    addrs[8] = '/Volumes/BINWORK/VoiceDataset/MIMII/pump/id_04/normal'
    addrs[9] = '/Volumes/BINWORK/VoiceDataset/MIMII/fan/id_04/normal'
    addrs[10] = '/Volumes/BINWORK/VoiceDataset/MIMII/slider/id_04/normal'
    addrs[11] = '/Volumes/BINWORK/VoiceDataset/MIMII/valve/id_04/normal'

    addrs[12] = '/Volumes/BINWORK/VoiceDataset/MIMII/pump/id_06/normal'
    addrs[13] = '/Volumes/BINWORK/VoiceDataset/MIMII/fan/id_06/normal'
    addrs[14] = '/Volumes/BINWORK/VoiceDataset/MIMII/slider/id_06/normal'
    addrs[15] = '/Volumes/BINWORK/VoiceDataset/MIMII/valve/id_06/normal'
    duration = 10
    source_number = 20

    for test_i in [0]:#0,1,2,3,4,5,6,7

        folder_address = addrs[test_i]

        res = {}
        res['picalite_db'] = 0
        res['picalite_time'] = 0
        res['picaold_db'] = 0
        res['picaold_time'] = 0


        S, A, X = pybss_tb.generate_matrix_S_A_X(
            folder_address, duration, source_number, mixing_type="normal", max_min=(1, 0.01), mu_sigma=(0, 1))

        eval_type = 'psnr'
        
        Eval_dB = 0
        pybss_tb.timer_start()
        for i in range(1):
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
        for i in range(1):
            hat_S = transformer.fit_transform(X)
            pybss_tb.timer_suspend()
            hat_S = hat_S.T
            Eval_dB += pybss_tb.bss_evaluation(S, hat_S, eval_type)
            pybss_tb.timer_resume()
        time = pybss_tb.timer_value()
        res['picaold_db'] += Eval_dB
        res['picaold_time'] += time


        
        print('type        eval_dB            time(ms) for       ' +
                str(source_number)+' sources, ' + str(test_i) + '-th test.')
        print('---------------------------------------------------------------------------')
        
        print('picalite: ', res['picalite_db'], '; ', res['picalite_time'])
        print('picaold: ', res['picaold_db'], '; ', res['picaold_time'])
