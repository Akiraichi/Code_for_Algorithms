import numpy as np
import collections
import sys


class dice(object):

    def __init__(self, theta, pi):
        self.theta = theta
        self.pi = pi

    def generate_data(self, num):
        x, s = [], []
        for _ in range(num):
            s.append(self.__spots(self.pi))
            x.append(self.__spots(self.theta[s[-1]]))

        return x, s

    @staticmethod
    def __spots(pi):
        p = np.random.rand()
        for i in range(len(pi)):
            if p < pi[:i + 1].sum():
                return i


class Supervised_learning(object):
    def __init__(self):
        self.pi = None
        self.theta = None
        self.n = None

    def estimate(self, x, s):
        self.n = self.__count(x, s)
        self.pi = self.n.sum(axis=1) / self.n.sum()
        self.theta = self.n / np.tile(self.n.sum(axis=1), (self.n.shape[1], 1)).T

    @staticmethod
    def __count(x, s):
        n = np.zeros((max(s) + 1, max(x) + 1))
        for x_, s_ in zip(x, s):
            n[s_][x_] += 1
        return n


class Unsupervised_learning(object):
    def __init__(self, theta=None, pi=None, c=3, m=2):
        self.c = c
        self.m = m

        # Step1 If no initial value is specified, it will be determined at random
        if pi is None:
            pi = np.random.rand(self.c)
        if theta is None:
            rand = np.random.rand(self.c, self.m)
            theta = rand / rand.sum(axis=1).reshape(-1, 1)  # normalisation
        self.pi = pi
        self.theta = theta
        self.r = None
        self.p = None

    def estimate(self, x):
        self.r = np.array([i for i in collections.Counter(x).values()])

        # Step2
        self.__update_p()

        # termination condition
        epsilon = 1e-6
        log_p = -sys.float_info.max
        delta = None

        while delta is None or delta >= epsilon:
            # Step3
            self.__update_param()

            # Step4
            log_p_new = self.__log_likelihood()
            delta = log_p_new - log_p
            log_p = log_p_new

    def __update_p(self):
        numer = self.pi.reshape(-1, 1) * self.theta
        self.p = numer / numer.sum(axis=0)

    def __update_param(self):
        self.pi = (self.r * self.p).sum(axis=1) / self.r.sum()
        self.__update_p()
        numer = self.r * self.p
        # self.theta = (numer.T / numer.sum(axis=1)).T  # if theta is known, comment out on this line.

    def __log_likelihood(self):
        log_p = np.log((self.pi.reshape(-1, 1) * self.theta).sum(axis=0))
        return (self.r * log_p).sum(axis=0)
