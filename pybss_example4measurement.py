from typing import no_type_check
from numpy.core.fromnumeric import repeat
from pybss_testbed import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pickle
import json
import warnings

colordict = {
    'meica': '#7ACFE5',
    'pica': '#7ACFE5',
    'pica_seq': '#3F9ABF',
    'aeica': '#EDED00',
    'aeica_seq': '#C6C600',
    'fastica': '#009988',
    'darkblue': '#024B7A',
    'lightblue': '#3F9ABF',
    'midblue': '#7ACFE5'
}
markerdict = {
    'meica': 'd',
    'pica': 'd',
    'pica_seq': 'D',
    'aeica': '^',
    'aeica_seq': 'v',
    'fastica': 'o'
}
barwidth = 0.18
bardistance = 0.2
figwidth = 6.5
figheight = 6.5 / 1.618


class REC():
    def init(self,node_num,S,X) -> None:
        self.X_len = X.shape[1]
        self.tb = PyFastbssTestbed()
        self.node_num =node_num
        self.ws = [None]*(node_num+2)
        self.snrs = [0]*(node_num+2)
        self.sdrs = [0]*(node_num+2)
        self.u0s = [0]*(node_num+2)
        self.xlens = [0]*(node_num+2)
        self.ts_all = [0]*(node_num+2)
        self.ts = [0]*(node_num+2)
        self.init_node_num = 0
        self.S = S
        self.X = X

    def __rec_node_info__(self,node_i,W,u0):
        t = self.tb.timer_value()
        self.tb.timer_suspend()
        self.ts_all[node_i] = t
        if node_i > 0 and node_i < (self.init_node_num+2):
            self.ts[node_i] = self.ts_all[node_i]-self.ts_all[node_i-1]
        self.u0s[node_i] = u0
        self.xlens[node_i] = self.X_len // u0
        hat_S = np.dot(W, self.X)
        self.snrs[node_i] = self.tb.bss_evaluation(self.S,hat_S,type='psnr')
        self.sdrs[node_i] = self.tb.bss_evaluation(self.S,hat_S,type='sdr')
        self.ws[node_i] = [list(arr) for arr in W] #W #list(W.flatten())
        self.tb.timer_resume()

    def __rec_finish___(self,node_i):
        self.xlens[0] = 0
        for i in range(node_i+1,self.init_node_num+2):
            self.ts_all[i] = self.ts_all[node_i]
            self.snrs[i] = self.snrs[node_i]
            self.sdrs[i] = self.sdrs[node_i]
            self.u0s[i] = self.u0s[node_i]


class ProgressiveICALite(REC):

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
                if node_num <= 0:
                    ext_interval = 1
            #++
            self.node_num = node_num
            #--
            if ext_interval <= 1:
                ext_interval = 1
                grad_var_tol = 0
            _X = X[:, :int(m / ext_interval)].copy()
            # _X = X[:, ::int(ext_interval)].copy()
            _X, V, V_inv = self._whiten_with_inv_v(_X)
            W = self._sym_decorrelation(np.dot(W, V_inv))
            W, lim = self._ica_par(_X, W, grad_var_tol, tol, g, max_iter)
            W = np.dot(W, V)
            if ext_interval < ext_interval_divisor:
                break
            #++
            self.__rec_node_info__(self.init_node_num-self.node_num,W,ext_interval)
            #--
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
        #++
        self.init_node_num = node_num
        self.node_num = node_num
        self.tb.timer_start()
        self.__rec_node_info__(0,w_init,init_ext_interval)
        #--
        W = self._pica(X, init_ext_interval, dynamic_adj_coef,
                       tol, grad_var_tol, g, max_iter, w_init, node_num)
        hat_S = np.dot(W, X)
        #++
        self.__rec_node_info__(self.init_node_num-self.node_num,W,1)
        self.__rec_finish___(self.init_node_num-self.node_num)
        #--
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


############################################################################################################
############################################################################################################
############################################################################################################





if __name__ == '__main__': 
    ext = ''
    repeat_num = 2
    machine_num = 4
    if False:
        res = {}
        fr = open('dataset/saxs10.pkl','rb')
        saxs = pickle.load(fr)
        ss,aa,xx = saxs
        fr.close()

        w_init = np.load("dataset/W.npy")
        
        base = 2
        u_0 = 2000
        tol = 1e-4
        dataset_id = 2
        
        for node_num in [10]:
            _res = {}
            _res['pica'] = {'ws':[],'snrs':[],'sdrs':[],'u0s':[],'times':[],'times_all':[],'xlens':[]}

            for i in tqdm(range(repeat_num)):
                i = dataset_id
                S,A,X = ss[dataset_id].copy(),aa[dataset_id].copy(),xx[dataset_id].copy()
                picalite.init(node_num,S,X)
                hat_S = picalite.pica(X, init_ext_interval=u_0, dynamic_adj_coef=2, tol=0.001, grad_var_tol=0.9, fun='logcosh', max_iter=200, w_init=w_init, node_num=node_num)
                _res['pica']['times'].append(picalite.ts)
                _res['pica']['times_all'].append(picalite.ts_all)
                _res['pica']['ws'].append(picalite.ws)
                _res['pica']['snrs'].append(picalite.snrs)
                _res['pica']['sdrs'].append(picalite.sdrs)
                _res['pica']['u0s'].append(picalite.u0s)
                _res['pica']['xlens'].append(picalite.xlens)

            res[node_num]=_res
        json.dump(res,open('measurement/measure_node_execution_50'+ext+'.json', 'w'))
    
    
    
    
    res = json.load(open('measurement/measure_node_execution_50'+ext+'.json', 'r'))

    _res = res['10']
    
    def draw_line(typ,eval_type):
        x0 = np.arange(0, 9, 1)
        vals = _res[typ][eval_type]
        y = np.mean(vals,axis=0)
        std = np.std(vals,axis=0)
        y1 = y - 1.96*std
        y2 = y + 1.96*std
        line = ax.errorbar(
            x0, y, color=colordict[typ], lw=1, ls='-', marker=markerdict[typ], ms=3)
        line_fill = ax.fill_between(x0, y1,y2, color=colordict[typ], alpha=0.2)
        return line
    
    def draw_box(typ,eval_type,off_set):
        vals = _res[typ][eval_type]
        x0 = np.arange(1,len(vals)+1, 1)
        box = ax.boxplot(vals,positions=x0+off_set,widths=barwidth,patch_artist=True, boxprops = {'color':'black','facecolor':colordict[typ]})
        return box
    
    def draw_scatter(typ,eval_type,off_set):
        vals = _res[typ][eval_type]
        y = np.mean(vals,axis=0)
        x0 = list(np.arange(0,len(vals[0]), 1))
        ys = _res[typ][eval_type]
        xx,yy = [],[]
        for i in range(len(ys)):
            yy += ys[i]
            xx += x0
        line = ax.scatter(xx, yy, color=colordict[typ], marker=markerdict[typ],s=20)
        x,y =xx,yy
        res ={}
        for _x,_y in zip(x,y):
            if _x in res:
                if _x>0:
                    res[_x] = res[_x] + [_y]
            else:
                if _x>0:
                    res[_x] = [_y]
        xx,yy = [],[]
        for key in sorted(res.keys()):
            xx.append(key)
            yy.append(np.mean(res[key]))
        line = ax.errorbar(
            xx, yy, color=colordict[typ], lw=1, ls='-')         
        return line
    
    def draw_scatter_payload(typ,eval_type,off_set):
        ys = _res[typ][eval_type]
        xs = _res[typ]['xlens'] #np.arange(0,len(vals[0]), 1)
        x,y=[],[]
        for i in range(len(ys)):
            x += xs[i]
            y += ys[i]
        x = np.array(x)
        y = np.array(y)
        line = ax.scatter(x, y, color=colordict[typ], marker=markerdict[typ],s=20)
        res ={}
        for _x,_y in zip(x,y):
            if _x in res:
                if _x>0:
                    res[_x] = res[_x] + [_y]
            else:
                if _x>0:
                    res[_x] = [_y]
        xx,yy = [],[]
        for key in sorted(res.keys()):
            xx.append(key)
            yy.append(np.mean(res[key]))
        line = ax.errorbar(
            xx, yy, color=colordict[typ], lw=1, ls='-')        
        return line
    
    # Node Execution ####################################################
    for eval_type in ['times','sdrs','xlens']:
        with plt.style.context(['science', 'ieee']):
            fig = plt.figure(figsize=(figwidth, figheight))
            ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
            x_step = np.arange(0, 12, 1)
            line2 = draw_scatter('pica',eval_type,-barwidth)
            if eval_type == 'times':
                ax.set_ylabel(r'Processing time ($ms$)')
                ax.set_yticks(np.arange(0, 55, 10))
            if eval_type == 'sdrs':
                ax.set_ylabel(r'SDR ($dB$)')
                ax.set_yticks(np.arange(0, 40, 5))
            if eval_type == 'xlens':
                ax.set_ylabel(r'Data usage $l_k$')
                ax.set_yscale('log')
                ax.set_ylim([40,1600000])
            plt.xticks(x_step, ['Client', 'Node 1', 'Node 2', 'Node 3',
                                'Node 4', 'Node 5', 'Node 6', 'Node 7', 'Server'], rotation=45)
            ax.set_xlim([-0.5, 11.5])
            ax.legend([line2,], [
                'pICA', ], loc='upper left')
            plt.savefig('measurement/'+eval_type+'_intermeidate_res_50.pdf', dpi=1200, bbox_inches='tight')

    # Payload size #########################
    for eval_type in ['times','sdrs',]:
        with plt.style.context(['science', 'ieee']):
            fig = plt.figure(figsize=(figwidth, figheight))
            ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.grid(True, linestyle='--', which='major',
                        color='lightgrey', alpha=0.5, linewidth=0.2)
            x_step = np.arange(0, 9, 1)
            line2 = draw_scatter_payload('pica',eval_type,-barwidth)
            if eval_type == 'times':
                ax.set_ylabel(r'Processing time ($ms$)')
                ax.set_yticks(np.arange(0, 125, 10))
                ax.set_ylim([0,125])
            if eval_type == 'sdrs':
                ax.set_ylabel(r'SDR ($dB$)')
                ax.set_yticks(np.arange(0, 40, 5))
            ax.set_xlabel(r'Data usage (Payload size) $l_k$')
            ax.set_xscale('log')
            ax.set_xlim([40,300000])
            ax.legend([line2,], [
                'pICA', ], loc='upper left')
            plt.savefig('measurement/'+eval_type+'_mid_payload_size_res_50.pdf', dpi=1200, bbox_inches='tight')



    # Time #########################################################
    def draw_line(typ, eval_type):
        x0 = _res[typ]['times_all'][0]
        y = _res[typ][eval_type][0]
        line = ax.errorbar(
            x0, y, color=colordict[typ], lw=1, ls='-', marker=markerdict[typ], ms=3)
        return line
    for eval_type in ['sdrs',]:
        with plt.style.context(['science']):
            fig = plt.figure(figsize=(figwidth,  figheight))
            ax = fig.add_subplot(1, 1, 1)
            ax.yaxis.grid(True, linestyle='--', which='major',
                            color='lightgrey', alpha=0.5, linewidth=0.2)
            line1 = draw_line('pica', eval_type)
            ax.set_xlabel(r'Processing time $t_p$ $(ms)$')    
            if eval_type == 'sdrs':
                ax.set_ylabel(r'SDR ($dB$)')
                ax.set_yticks(np.arange(0, 40, 5))
            # ax.set_xlim([-0.5, 9.5])
            # plt.xticks(x0,160000//np.array(list(map(int, np.array(x_u0['pica'])+0.5))))
            ax.legend([line1], [
                'pICA',], loc='upper left')
            plt.savefig('measurement/'+eval_type+'_time_intermeidate_res_50.pdf',dpi=1200, bbox_inches='tight')