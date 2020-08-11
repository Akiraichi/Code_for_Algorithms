import numpy as np
from matplotlib import pyplot as plt
import sys
import scipy.stats as scs


class mixed_normal_dice(object):
    """mixed normal distribution model"""

    def __init__(self, K=3, mu=None, sigma=None):
        self.K = K
        # If no initial value is specified, it will be determined at random
        if mu is None:
            mu = []
            for i in range(self.K):
                mu += [(np.random.rand() - 0.5) * 10.0]  # range: -5.0 - 5.0
        if sigma is None:
            sigma = []
            for i in range(self.K):
                sigma += [np.random.rand() * 3.0]  # range: 0.0 - 3.0:

        self.mu = mu
        self.scale = sigma

    def generate_data(self, N):
        X = []
        mu_star = []
        sigma_star = []
        for i in range(self.K):
            if self.mu.ndim >= 2:
                # In the case of multidimensional normal distribution
                X = np.append(X, np.random.multivariate_normal(self.mu[i], self.scale[i], N // self.K))
                result = X.reshape(-1, self.mu.ndim), mu_star, sigma_star
            else:
                # In the case of one-dimensional normal distribution
                X = np.append(X, np.random.normal(self.mu[i], self.scale[i], N // self.K))
                result = X, mu_star, sigma_star
            mu_star = np.append(mu_star, self.mu[i])
            sigma_star = np.append(sigma_star, self.scale[i])

        return result


class mixed_normal_distribution(object):
    """Parameter Estimation for Mixed Normal Distribution"""

    def __init__(self, K, pi=None, mu=None, sigma=None):
        self.K = K
        # If no initial value is specified, it will be determined at random
        if pi is None:
            pi = np.random.rand(self.K)
        if mu is None:
            mu = np.random.randn(K)
        if sigma is None:
            sigma = np.abs(np.random.randn(K))
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def estimate(self, X):
        # termination condition
        epsilon = 0.000001

        # Step1 initialize gmm parameter
        Q = -sys.float_info.max
        delta = None

        # EM algorithm

        while delta is None or delta >= epsilon:
            gf = self.__gaussian(self.mu, self.sigma)
            # E step: example posterior probability of hidden variable gamma
            gamma = self.__estimate_posterior_likelihood(X, self.pi, gf)

            # M step: miximize Q function by estimating mu, sigma and pi
            self.__estimate_gmm_parameter(X, gamma)

            # calculate Q function
            Q_new = self.__calc_Q(X, gamma)
            delta = Q_new - Q
            Q = Q_new

    @staticmethod
    def __estimate_posterior_likelihood(X, pi, gf):
        l = np.zeros((X.size, pi.size))
        for (i, x) in enumerate(X):
            l[i, :] = gf(x)

        return pi * l * np.vectorize(lambda y: 1 / y)(l.sum(axis=1).reshape(-1, 1))

    @staticmethod
    def __gaussian(mu, sigma):
        def f(x):
            return np.exp(-0.5 * (x - mu) ** 2 / sigma) / np.sqrt(2 * np.pi * sigma)

        return f

    def __estimate_gmm_parameter(self, X, gamma):
        N = gamma.sum(axis=0)
        self.mu = (gamma * X.reshape((-1, 1))).sum(axis=0) / N
        self.sigma = (gamma * (X.reshape(-1, 1) - self.mu) ** 2).sum(axis=0) / N
        self.pi = N / X.size

    def __calc_Q(self, X, gamma):
        Q = (gamma * (np.log(self.pi * (2 * np.pi * self.sigma) ** (-0.5)))).sum()
        for (i, x) in enumerate(X):
            Q += (gamma[i, :] * (-0.5 * (x - self.mu) ** 2 / self.sigma)).sum()
        return Q

    def result(self, mu_star, sigma_star):
        print(u'mu*: %s, sigma*: %s' % (str(np.sort(np.around(mu_star, 3))), str(np.sort(np.around(sigma_star, 3)))))
        print(u'mu : %s, sigma : %s' % (str(np.sort(np.around(self.mu, 3))), str(np.sort(np.around(self.sigma, 3)))))

    def graph(self, X):
        n, bins, _ = plt.hist(X, 50, density=True, alpha=0.3)
        seq = np.arange(-15, 15, 0.02)
        for i in range(self.K):
            plt.plot(seq, self.__gaussian(self.mu[i], self.sigma[i])(seq), linewidth=2.0)
        plt.savefig('gmm_graph.png')
        plt.show()


class convex_clustering(object):
    def __init__(self, sigma):
        # Step1 Set the variance to an appropriate value.
        self.sigma = sigma
        self.pi = None
        self.n = None
        self.X = None

    def clustering(self, x):
        self.X = x
        # Step2 Calculating a prior f
        self.n = len(x)
        f = self.__calc_f(self.X, self.sigma, self.__gaussian)

        # Step3 Set the initial value of prior probability pi
        self.pi = np.repeat(1 / self.n, self.n)

        # termination condition
        epsilon = 1e-6
        log_p = -sys.float_info.max
        delta = None

        # Step4 Update parameters
        while delta is None or delta > epsilon:
            self.__update_pi(f)
            log_p_new = self.__log_likelihood(self.pi, f)
            delta = log_p_new - log_p
            log_p = log_p_new

    def __calc_f(self, x, sigma, gauss_func):
        """Calculating f

        Args:
            x: pattern
            sigma: variance
            gauss_func: normal distribution function

        Returns: f

        """
        f = [gauss_func(x, mu, sigma) for mu in x]
        return np.reshape(f, (self.n, self.n))

    def __update_pi(self, f):
        numer = self.pi.reshape(-1, 1) * f
        denom = numer.sum(axis=0)
        self.pi = np.sum(numer / denom, axis=1) / self.n
        # To improve the convergence efficiency, set the pi near zero to zero
        for i, pi_i in enumerate(self.pi):
            if pi_i < 1e-7:
                self.pi[i] = 0

    @staticmethod
    def __log_likelihood(pi, f):
        return np.log((pi * f.T).sum(axis=1)).sum(axis=0)

    def __gaussian(self, x, mu, sigma):
        if self.sigma.ndim >= 2:
            # In the case of multidimensional normal distribution
            result = scs.multivariate_normal.pdf(x, mu, sigma[0])
        else:
            # In the case of 1D normal distribution
            result = scs.norm.pdf(x, np.tile(mu, len(x)), np.tile(sigma, len(x) // len(sigma)))

        return result

    def graph(self):
        # Half an initial pi is used as threshold.
        threshold = 0.01
        mu_valid, pi_valid = self.__select_valid_centroid(self.X, self.pi, threshold)
        K = len(mu_valid)
        print(f"The number of used centroids is ({K}/{self.n}) using sigma:({np.max(self.sigma)}).")
        const_sigma = np.array([self.sigma, ] * K)

        self.__plot_sample(self.X)
        self.__plot_gauss(mu_valid, const_sigma, self.X, pi_valid)
        plt.show()

    def __plot_sample(self, X):
        if self.sigma.ndim >= 2:
            x, y = X.T
            plt.plot(x, y, "bo")
        else:
            bins = int(np.ceil(max(X) - min(X))) * 4
            plt.hist(X, bins, fc="b")

    def __plot_gauss(self, mu, sigma, X, pi):
        if self.sigma.ndim >= 2:
            # In the case of multidimensional normal distribution
            delta = 0.025  # Sampling rate of contour.
            x_min, y_min = np.floor(X.min(axis=0))
            x_max, y_max = np.ceil(X.max(axis=0))
            X, Y = np.meshgrid(np.arange(x_min, x_max, delta),
                               np.arange(y_min, y_max, delta))
            grid = np.array([X.T, Y.T]).T

            circle_rate = 3  # Adjust size of a drawn circle.
            Z = 0
            for i in np.arange(0, len(mu)):
                Z += self.__gaussian(grid, mu[i], sigma[i] * circle_rate) * pi[i]
            plt.contour(X, Y, Z, 2, linewidths=2, colors="g")
        else:
            # In the case of 1D normal distribution
            grid = np.linspace(np.floor(X.min()), np.ceil(X.max()), self.n)
            weight = self.n / np.sqrt(2 * np.pi)
            dist = np.zeros(self.n)
            for i in range(len(mu)):
                dist += self.__gaussian(grid, mu[i], sigma[i]) * weight * pi[i]
            plt.plot(grid, dist, "g", linewidth=2.0)

    @staticmethod
    def __select_valid_centroid(x, pi, threshold):
        """
        1 Get the index of the most probable pi
        2 Add the highest prior probability x value (centroid) to mu
        3 Add the highest prior probability pi value to the ratio
        4 If it is below the threshold, break

        Args:
            x: Data (centroid)
            pi: prior probability
            threshold: The threshold for judging prior probability pi as valid

        Returns: Prior probability pi determined to be valid and its centroid

        """
        mu_valid = np.array([])
        pi_valid = np.array([])
        indexs = pi.argsort()[::-1]
        for row, index in enumerate(indexs):
            if pi[index] < threshold:
                if 0 < row:
                    mu_valid = np.reshape(mu_valid, (row, mu_valid.size // row))
                break
            else:
                mu_valid = np.append(mu_valid, x[index])
                pi_valid = np.append(pi_valid, pi[index])
        return mu_valid, pi_valid
