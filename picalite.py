import numpy as np
import warnings
# 作者: 沈云彬
# 日期: 2021年5月3日


class ProgressiveICALite():

    # Pica 改进后的矩阵白化, 多了一个V的逆矩阵, 加快后面计算
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

    # 来自于Fastica的tanh函数
    def _logcosh(self, x, alpha=1.0):
        x *= alpha
        gx = np.tanh(x, x)
        g_x = np.empty(x.shape[0])
        for i, gx_i in enumerate(gx):
            g_x[i] = (alpha * (1 - gx_i ** 2)).sum()
        return gx, g_x

    def _exp(self, x):
        exp = np.exp(-(x ** 2) / 2)
        gx = x * exp
        g_x = (1 - x ** 2) * exp
        return gx, g_x.sum(axis=-1)

    def _cube(self, x):
        return x ** 3, (3 * x ** 2).sum(axis=-1)

    # 来自于FastICA的去相关化
    def _sym_decorrelation(self, W):
        S, U = np.linalg.eigh(np.dot(W, W.T))
        np.clip(S, 1e-15, None, out=S)
        return np.linalg.multi_dot([U * (1. / np.sqrt(S)), U.T, W])

    # Pica 带有梯度下降速率判断的牛顿迭代

    def _ica_par(self, X, W, grad_var_tol, tol, g, max_iter):
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
                'PICA did not converge. Consider increasing tolerance or the maximum number of iterations.')
        return W, lim

    # Pica 主要部分
    def _pica(self, X, proc_mode, init_ext_interval, dynamic_adj_coef, tol, grad_var_tol, g, max_iter):
        n, m = X.shape
        W = np.random.random_sample((n, n))
        ext_interval = int(init_ext_interval)
        ext_interval_divisor = dynamic_adj_coef
        while(True):
            # Pica 的收尾模式选择, 有快速和普通模式两种选择, 快速模式根据计算中间结果判断是否计算部分数据就提前跳出, 普通模式会将全部数据迭代一遍
            if ext_interval < ext_interval_divisor:
                if proc_mode == 'fast':
                    ext_interval = ext_interval_divisor // dynamic_adj_coef
                elif proc_mode == 'normal':
                    ext_interval = 1
                elif proc_mode == 'precise':
                    ext_interval = 1
                    grad_var_tol = 0
                else:
                    raise NameError(
                        'The value of proc_mode is invalid, it must be "fast" , "normal" or "precise". ')
            # Pica 按等间隔提取数据
            _X = X[:, :int(m//ext_interval)]
            # _X = X[:, ::int(ext_interval)]
            # Pica 重新白化并计算白化矩阵V的逆矩阵
            _X, V, V_inv = self._whiten_with_inv_v(_X)
            # 计算适重新应白化后的X的W矩阵
            W = self._sym_decorrelation(np.dot(W, V_inv))
            # Pica 根据梯度下降和 tol 判断是否该跳出当前 Extraction level 的牛顿迭代
            W, lim = self._ica_par(_X, W, grad_var_tol, tol, g, max_iter)
            # 计算适用于未白化前的X的W矩阵
            W = np.dot(W, V)
            # 判断是否计算完成并跳出
            if ext_interval < ext_interval_divisor:
                break
            # 根据梯度下降速率跳出法跳出模式最终计算出的lim 值和 tol比较, 如果 lim 小于 tol 则说明当前的 W 对于当前 Extraction 的数据已经过于收敛, 换句话说再计算下去, W 也不会改变太多, 反而会浪费大量时间, 这时候需要动态增加下次计算的数据量来增加 W 的收敛速度
            if lim < tol:
                ext_interval_divisor *= dynamic_adj_coef
            else:
                ext_interval_divisor = max(
                    dynamic_adj_coef, ext_interval_divisor // dynamic_adj_coef)
            ext_interval //= ext_interval_divisor
        return W

    def pica(self, X, proc_mode='precise', init_ext_interval=None, dynamic_adj_coef=2, tol=0.001, grad_var_tol=0.9, fun='logcosh', max_iter=200):
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
            n, m = np.shape(X)
            init_ext_interval = m // n
        W = self._pica(X, proc_mode,  init_ext_interval, dynamic_adj_coef,
                       tol, grad_var_tol, g, max_iter)
        S2 = np.dot(W, X)
        return S2


picalite = ProgressiveICALite()
