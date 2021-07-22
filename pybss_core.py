import numpy as np
import warnings


class ProgressiveICALite():

    def __init__(self) -> None:
        pass

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
                break
            if lim > lim_max:
                lim_max = lim
            lim_sum += lim
            if lim_sum < grad_var_tol*0.5*(lim_max+lim)*(i+1):
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
            _X = X[:, :int(m / ext_interval)].copy().astype(np.float32)
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

picalite = ProgressiveICALite()
