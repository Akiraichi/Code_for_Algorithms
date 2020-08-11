import numpy as np
from matplotlib import pyplot as plt


class Markov_dice(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho

    def generate_spots(self, n):
        s, x = [], []
        s += self.__spots(self.rho.reshape(1, -1), 0)
        for n in range(n):
            x += self.__spots(self.B, s[-1])
            s += self.__spots(self.A, s[-1])
        return np.array(s[:-1]), np.array(x)

    @staticmethod
    def __spots(theta, s_t):
        p = np.random.rand()
        for i in range(theta.shape[0]):
            if p < theta[s_t, :i + 1].sum():
                break
        return [i]


class Markov(object):
    def __init__(self):
        self.A = None
        self.B = None
        self.n = None
        self.m = None

    def estimate(self, x, s):
        self.__initialize(x, s)
        self.n, self.m = self.__count(x, s)
        self.A = self.m / self.n.sum(axis=1).reshape(-1, 1)
        self.B = self.n / self.n.sum(axis=1).reshape(-1, 1)

    def __initialize(self, x, s):
        self.A = np.zeros((max(s) + 1, max(s) + 1))
        self.B = np.zeros((max(s) + 1, max(x) + 1))

    @staticmethod
    def __count(x, s):
        n = np.zeros((max(s) + 1, max(x) + 1))
        for x_, s_ in zip(x, s):
            n[s_][x_] += 1

        m = np.zeros((max(s) + 1, max(s) + 1))
        for i in range(len(s) - 1):
            m[s[i]][s[i + 1]] += 1

        return n, m


class Forward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]

    def evaluate(self, x):
        self.__initialize(x)
        self.__forward_seq(x, self.__forward)
        # Step 3
        return self.alpha[-1, :].sum()

    def scaled_evaluate(self, x):
        self.__scaled_initialize(x)
        self.__forward_seq(x, self.__scaled_forward)
        return self.C.prod()

    def __initialize(self, x):
        self.alpha = np.zeros((x.size, self.c))
        # Step1 初期化
        self.alpha[0, :] = self.rho * self.B[:, x[0]]

    def __scaled_initialize(self, x):
        self.__initialize(x)
        self.C = np.zeros(x.size)
        self.C[0] = self.alpha[0, :].sum()
        self.alpha[0, :] /= self.C[0]

    def __forward_seq(self, x, forward_func):
        # Step 2
        for (i, x_t) in enumerate(x[1:]):
            self.alpha[i + 1, :] = forward_func(x_t, i)

    def __forward(self, x_t, i):
        return self.alpha[i, :].dot(self.A) * self.B[:, x_t]

    def __scaled_forward(self, x_t, i):
        _alpha = self.__forward(x_t, i)
        self.C[i + 1] = _alpha.sum()
        return _alpha / self.C[i + 1]


class Backward(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c = A.shape[0]

    def evaluate(self, x):
        self.__initialize(x)
        self.__backward_seq(x, self.__backward)
        return (self.rho * self.B[:, x[0]] * self.beta[0, :]).sum()

    def scaled_evaluate(self, x, C):
        self.__scaled_initialize(x, C)
        self.__backward_seq(x, self.__scaled_backward)
        return self.C.prod()

    def __initialize(self, x):
        self.beta = np.zeros((x.size, self.c))
        # step1
        self.beta[-1, :] = np.ones(self.c)

    def __scaled_initialize(self, x, C):
        self.__initialize(x)
        self.C = C
        self.beta[-1, :] /= self.C[-1]

    def __backward_seq(self, x, backward_func):
        for (i, x_t) in list(enumerate(x[1:]))[::-1]:
            self.beta[i, :] = backward_func(x_t, i + 1)

    def __backward(self, x_t, i):
        return (self.A * self.B[:, x_t] * self.beta[i, :]).sum(axis=1)

    def __scaled_backward(self, x_t, i):
        _beta = self.__backward(x_t, i)
        return _beta / self.C[i]


class BaumWelch(object):

    def __init__(self, A, B, rho):
        self.A = A
        self.B = B
        self.rho = rho
        self.c, self.m = B.shape

    def estimate(self, x, T):
        self.__init_storage(T)
        for t in range(T):
            fw, bw = Forward(self.A, self.B, self.rho), Backward(self.A, self.B, self.rho)
            P, _ = fw.evaluate(x), bw.evaluate(x)
            self.__update(fw.alpha, bw.beta, x)
            self.__store(np.log(P), t)

    def scaled_estimate(self, x, T):
        self.__init_storage(T)
        for t in range(T):
            fw, bw = Forward(self.A, self.B, self.rho), Backward(self.A, self.B, self.rho)
            P, _ = fw.scaled_evaluate(x), bw.scaled_evaluate(x, fw.C)
            self.__scaled_update(fw.alpha, bw.beta, fw.C, x)
            self.__store(np.log(fw.C).sum(), t)

    def graph_A(self, A):
        self.__graph(self.__A, A, 'A')

    def graph_B(self, B):
        self.__graph(self.__B, B, 'B')

    def graph_P(self):
        seq = np.arange(self.__logP.size)
        plt.plot(seq, self.__logP)
        plt.savefig('graph/logP.png')
        plt.clf()

    def __update(self, alpha, beta, x):
        denom = alpha[-1, :].sum().reshape(-1, 1)
        numer = self.__numer(alpha, beta, x)
        gamma = alpha * beta / denom
        xi = numer / denom
        self.__update_param(gamma, xi, x)

    def __scaled_update(self, alpha, beta, C, x):
        denom = np.tile(C[1:], (self.c, 1)).T
        numer = self.__numer(alpha, beta, x)
        gamma = alpha * beta
        xi = numer / denom
        self.__update_param(gamma, xi, x)

    def __numer(self, alpha, beta, x):
        """"""
        # alpha[:-1, :]ではalphaの最後の行は含まれていないことに注意。t+1の分までalphaは必要ないから。
        _a = (np.tile(alpha[:-1, :], (self.c, 1, 1)).swapaxes(0, 1) * self.A.T).transpose(2, 0, 1)
        _b = self.B[:, x[1:]].T * beta[1:, :]
        return _a * _b

    def __update_param(self, gamma, xi, x):
        """Step3 パラメータの更新"""
        self.A = xi.sum(axis=1) / gamma[:-1, :].sum(axis=0).reshape(-1, 1)
        # gammaとx==kであるself.mを積する。gammaの0番目の添字と self.mの1番目の添え字をダミー変数とする
        self.B = np.tensordot(gamma, [(x == k) for k in range(self.m)], axes=(0, 1)) / gamma.sum(axis=0).reshape(-1, 1)
        self.rho = gamma[0, :]

    def __init_storage(self, T):
        self.__logP = np.zeros(T)
        self.__A = np.zeros((T + 1, self.c, self.c))
        self.__B = np.zeros((T + 1, self.c, self.m))
        self.__A[0, :, :] = self.A
        self.__B[0, :, :] = self.B

    def __store(self, logP, t):
        self.__logP[t] = logP
        self.__A[t + 1, :, :] = self.A
        self.__B[t + 1, :, :] = self.B

    def __graph(self, param_e, param_t, param_str):
        seq = np.tile(np.arange(param_e.shape[0]), (self.c, 1)).T
        for (i, pe) in enumerate(param_e.transpose(2, 0, 1)):
            plt.ylim(0.0, 1.0)
            plt.plot(seq, np.tile(param_t[:, i].reshape(-1, 1), seq.shape[0]).T, 'k-', linewidth=0.5)
            plt.plot(seq, pe)
            plt.savefig('graph/{}{}.png'.format(param_str, i + 1))
            plt.clf()
