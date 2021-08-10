import numpy as np
import warnings


class ProgressiveICALite():

    def __init__(self) -> None:
        pass

    def count_init(self):
        self.grad_break_count = 0
        self.tol_break_count = 0

    def _whiten_with_inv_v(self, X):
        X -= X.mean(axis=-1)[:, np.newaxis]
        A = np.dot(X, X.T)
        np.clip(A, 1e-15, None, out=A)
        D, P = np.linalg.eig(A)
        D = np.diag(abs(D))
        D_half = np.sqrt(np.linalg.inv(D))
        V = np.dot(D_half, P.T)
        V_inv = np.dot(P, np.sqrt(D))
        X1 = np.sqrt(X.shape[1]) * np.dot(V, X)
        return X1, V, V_inv

    def _logcosh(self, x, alpha=1.0):
        '''
        # Usage:
        g_1(u)

        # Parameters:


        # Output:

            gx: separation matrix W.
        '''
        x *= alpha
        gx = np.tanh(x, x)
        g_x = np.empty(x.shape[0])
        for i, gx_i in enumerate(gx):
            g_x[i] = (alpha * (1 - gx_i ** 2)).sum()
        return gx, g_x

    def _exp(self, x):
        '''
        # Usage:
        g_2(u)

        # Parameters:


        # Output:

            gx: separation matrix W.
        '''
        exp = np.exp(-(x ** 2) / 2)
        gx = x * exp
        g_x = (1 - x ** 2) * exp
        return gx, g_x.sum(axis=-1)

    def _cube(self, x):
        '''
        # Usage:
        g_3(u)

        # Parameters:


        # Output:

            gx: separation matrix W.
        '''
        return x ** 3, (3 * x ** 2).sum(axis=-1)

    def _sym_decorrelation(self, W):
        '''
        # Usage:
        Decorrelation of W

        # Parameters:

            W: separation matrix

        # Output:

            Decorrelated separation matrix W.
        '''
        S, U = np.linalg.eigh(np.dot(W, W.T))
        np.clip(S, 1e-15, None, out=S)
        return np.linalg.multi_dot([U * (1. / np.sqrt(S)), U.T, W])

    def _ica_par(self, X, W, grad_var_tol, tol, g, max_iter):
        '''
        # Usage:
        pICA process logic

        # Parameters:

            X: Whitened subset of mixed signals X_k.
            W: w_{k-1}
            grad_var_tol: maximum grandient decreasing rate, h
            tol: tolerance
            g: g(u)
            max_iter: Maximum number of iteration.

        # Output:

            Separation matrix w_k.
        '''
        lim_sum = 0
        lim_max = 0
        for i in range(max_iter):
            gbx, g_bx = g(np.dot(W, X))
            W1 = self._sym_decorrelation(np.dot(gbx, X.T) - g_bx[:, None] * W)
            lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
            W = W1
            if lim < tol:
                self.tol_break_count += 1
                break
            if lim > lim_max:
                lim_max = lim
            lim_sum += lim
            if lim_sum < grad_var_tol*0.5*(lim_max+lim)*(i+1):
                self.grad_break_count += 1
                break
        else:
            warnings.warn(
                'pICA/FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.')
        return W, lim

    def _pica(self, X, init_ext_interval, dynamic_adj_coef, tol, grad_var_tol, g, max_iter, w_init, node_num):
        '''
        # Usage:
        Convergent pICA.

        # Parameters:

            X: Whitened mixed signals.
            init_ext_interval: initial extration interval of pICA, minimum m//n
            dynamic_adj_coef: alpha_1, increasing rate of subset
            tol: tolerance of convergence
            grad_var_tol: maximum grandient decreasing rate, h
            g: g(u)
            max_iter: Maximum number of iteration.
            w_init: w_0
            node_num: The number of the network nodes

        # Output:

            Separation matrix: W.
        '''
        n, m = X.shape
        W = w_init
        ext_interval = init_ext_interval
        ext_interval_divisor = dynamic_adj_coef
        while(True):
            if node_num is not None:
                node_num -=1
                if node_num < 0:
                    ext_interval = 1
            if ext_interval <= 1:
                ext_interval = 1
                grad_var_tol = 0
            _X = X[:, :int(m / ext_interval)].copy().astype(np.float16).astype(np.float32)
            # _X = X[:, ::int(ext_interval)].copy()
            _X, V, V_inv = self._whiten_with_inv_v(_X)
            W = self._sym_decorrelation(np.dot(W, V_inv))
            W, lim = self._ica_par(_X, W, grad_var_tol, tol, g, max_iter)
            W = np.dot(W, V)
            if grad_var_tol == 0:
                break
            if lim < tol:
                ext_interval_divisor *= dynamic_adj_coef
            else:
                ext_interval_divisor = max(
                    dynamic_adj_coef, ext_interval_divisor / dynamic_adj_coef)
            ext_interval /= ext_interval_divisor
        return W

    def pica(self, X, init_ext_interval=None, dynamic_adj_coef=2, tol=0.001, grad_var_tol=0.9, fun='logcosh', max_iter=200, w_init=None, node_num=None):
        '''
        # Usage:

        # Parameters:

            X: Whitened mixed signals.
            init_ext_interval: initial extration interval of pICA, maximum m//n
            dynamic_adj_coef: alpha_1, increasing rate of subset
            tol: tolerance of convergence
            grad_var_tol: maximum grandient decreasing rate, h
            fun: g(u)
            max_iter: Maximum number of iteration.
            w_init: w_0

        # Output:

            Separated source S_hat.
        '''
        self.count_init()
        n, m = np.shape(X)
        if fun == 'logcosh':
            g = self._logcosh
        elif fun == 'exp':
            g = self._exp
        elif fun == 'cube':
            g = self._cube
        else:
            raise ValueError(
                "Unknown function, the value of 'fun' should be one of 'logcosh', 'exp', 'cube'.")
        if init_ext_interval is None:
            init_ext_interval = m // n
        if w_init is None:
            w_init = np.random.random_sample((n, n))
        W = self._pica(X, init_ext_interval, dynamic_adj_coef,
                       tol, grad_var_tol, g, max_iter, w_init, node_num)
        hat_S = np.dot(W, X)
        return hat_S


    def fastica(self, X, tol=0.001, fun='logcosh', max_iter=200, w_init=None):
        '''
        # Usage:

        # Parameters:

            X: Whitened mixed signals.
            tol: tolerance of convergence
            fun: g(u)
            max_iter: Maximum number of iteration.
            w_init: w_0

        # Output:

            Separated source S_hat.
        '''
        n, m = np.shape(X)
        if fun == 'logcosh':
            g = self._logcosh
        elif fun == 'exp':
            g = self._exp
        elif fun == 'cube':
            g = self._cube
        else:
            raise ValueError(
                "Unknown function, the value of 'fun' should be one of 'logcosh', 'exp', 'cube'.")
        if w_init is None:
            w_init = np.random.random_sample((n, n))
        _X = X.copy()
        _X, V, V_inv = self._whiten_with_inv_v(_X)
        W = self._sym_decorrelation(np.dot(w_init, V_inv))
        W, lim = self._ica_par(_X, W, grad_var_tol=0, tol=tol, g=g, max_iter=max_iter)
        W = np.dot(W, V)
        hat_S = np.dot(W, X)
        return hat_S



    def adaptive_extraction_iteration(self, X, W, g, max_iter, tol, ext_adapt_ica):
        '''
        # adaptive_extraction_iteration(self, X, B, max_iter, tol, _ext_adapt_ica):

        # Usage:

            Adaptive extraction newton iteration.
            It is a combination of several fastica algorithm with different partial
            signals, which is extracted by different intervals. the extraction 
            interval can be detemined by the convergence of the iteration.

        # Parameters:

            X: Mixed signals, which is obtained from the observers.
            max_iter: Maximum number of iteration.
            tol: Tolerance of the convergence of the matrix B 
                calculated from the last iteration and the 
                matrix B calculated from current newton iteration.
            _ext_adapt_ica: The intial and the maximum extraction interval of the 
                input signals.

        # Output:

            Estimated source separation matrix B.
        '''
        _prop_series = np.arange(1, ext_adapt_ica)
        grads_num = _prop_series.shape[0]
        _tols = tol*(_prop_series**0.5)
        _tol = 1
        for i in range(grads_num-1, 0, -1):
            if _tol > _tols[i]:
                _X = X[:, ::int(_prop_series[i])]
                _X, V, V_inv = self._whiten_with_inv_v(_X)
                W = self._sym_decorrelation(np.dot(W, V_inv))
                W, _tol = self._ica_par(_X, W, grad_var_tol=0, tol=_tols[i], g=g, max_iter=max_iter)
                W = np.dot(W, V)
        _X = X
        _X, V, V_inv = self._whiten_with_inv_v(_X)
        W = self._sym_decorrelation(np.dot(W, V_inv))
        W, _tol = self._ica_par(_X, W, grad_var_tol=0, tol=tol, g=g, max_iter=max_iter)
        W = np.dot(W, V)
        return W

    def aeica(self, X, ext_adapt_ica=50, tol=1e-04, fun='logcosh', max_iter=100, w_init=None, node_num=5):

        '''
        # aeica(self, X, max_iter=100, tol=1e-04, ext_adapt_ica=30):

        # Usage:

            Adaptive extraction ICA.
            It is a combination of several fastica algorithm with different partial
            signals, which is extracted by different intervals. the extraction 
            interval can be detemined by the convergence of the iteration.
            A original fastica is added at the end, in order to get the best result.

        # Parameters:

            X: Mixed signals, which is obtained from the observers.
            max_iter: Maximum number of iteration.
            tol: Tolerance of the convergence of the matrix B 
                calculated from the last iteration and the 
                matrix B calculated from current newton iteration.
            _ext_adapt_ica: The intial and the maximum extraction interval of the 
                input signals.

        # Output:

            Estimated source signals matrix S.
        '''
        n, m = np.shape(X)
        if fun == 'logcosh':
            g = self._logcosh
        elif fun == 'exp':
            g = self._exp
        elif fun == 'cube':
            g = self._cube
        else:
            raise ValueError(
                "Unknown function, the value of 'fun' should be one of 'logcosh', 'exp', 'cube'.")
        if w_init is None:
            w_init = np.random.random_sample((n, n))
        self.init_node_num = node_num
        self.node_num = node_num
        W = self.adaptive_extraction_iteration(X, w_init, g, max_iter, tol, ext_adapt_ica)
        hat_S = np.dot(W, X)
        return hat_S

picalite = ProgressiveICALite()


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
    test_num = 50  # repeat number
    eval_type = 'psnr'
    # node_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    node_num = [1, 2, 3, 4, 5, 6, 7]
    extraction_num = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    for node in np.arange(0, len(node_num), 1):
        Eval_dB = 0
        node_id = node_num[node]
        for i in range(test_num):
            dataset_id = i
            print('*** N_test:', i+1, ' with ', node_id, ' nodes.')
            S, A, X = ss[dataset_id].copy(), aa[dataset_id].copy(
            ), xx[dataset_id].copy().astype(np.float32)

            pybss_tb.timer_start()
            hat_S = picalite.pica(X, init_ext_interval=125, dynamic_adj_coef=2, tol=0.0001, grad_var_tol=0.90, fun='logcosh', max_iter=200, w_init=W, node_num=node_num[node])
 
            pybss_tb.timer_suspend()
            Eval_dB = pybss_tb.bss_evaluation(S, hat_S, eval_type)
            pybss_tb.timer_resume()
            time = pybss_tb.timer_value()
            measure_write('pICA_'+str(node_num[node])+"_break_count", [
                          'vnf', node_num[node], 'grad_break_count', picalite.grad_break_count, 'tol_break_count', picalite.tol_break_count])
            print('pICA    *** separation accuracy (dB): ' +
                  str(Eval_dB) + ', separation time (ms): ' + str(time))
