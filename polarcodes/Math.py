#!/usr/bin/env python

"""
Math provides miscellaneous math operations to the other classes, which are important for polar codes
algorithm implementations.
"""

import numpy as np

class Math:
    def bit_reversed(self, x, n):
        """
        Bit-reversal operation.

        :param x: a vector of indices
        :param n: number of bits per index in ``x``
        :type x: ndarray<int> or int
        :type n: int
        :return: bit-reversed version of x
        :rtype: ndarray<int> or int
        """

        result = 0
        for i in range(n):  # for each bit number
            if (x & (1 << i)):  # if it matches that bit
                result |= (1 << (n - 1 - i))  # set the "opposite" bit in result
        return result

    def logdomain_diff(self, x, y):
        """
        Find the difference between x and y in log-domain. It uses log1p to improve numerical stability.

        :param x: any number in the log-domain
        :param y: any number in the log-domain
        :type x: float
        :type y: float
        :return: the result of x - y
        :rtype: float
        """

        if x > y:
            z = x + np.log1p(-np.exp(y - x))
        else:
            z = y + np.log1p(-np.exp(x - y))
        return z

    def logdomain_sum(self, x, y):
        """
        Find the addition of x and y in log-domain. It uses log1p to improve numerical stability.

        :param x: any number in the log-domain
        :param y: any number in the log-domain
        :type x: float
        :type y: float
        :return: the result of x + y
        :rtype: float
        """

        if x > y:
            z = x + np.log1p(np.exp(y - x))
        else:
            z = y + np.log1p(np.exp(x - y))
        return z

    def bit_perm(self, x, p, n):
        """
        Find the permutation of an index.

        :param x: a vector of indices
        :param p: permutation vector, ex: bit-reversal is (0,1,...,n-1)
        :param n: number of bits per index in ``x``
        :type x: ndarray<int> or int
        :type p: ndarray<int>
        :type n: int
        :return: permuted indices
        :rtype: ndarray<int> or int
        """

        result = 0
        for i in range(n):
            b = (x >> p[i]) & 1
            result ^= (-b ^ result) & (1 << (n - i - 1))
        return result

    # find hamming weight of an index x
    def hamming_wt(self, x, n):
        """
        Find the bit-wise hamming weight of an index.

        :param x: an index
        :param n: number of bits in ``x``
        :type x: int
        :type n: int
        :return: bit-wise hamming weight of ``x``
        :rtype: int
        """

        m = 1
        wt = 0
        for i in range(n):
            b = (x >> i) & m
            if b:
                wt = wt + 1
        return wt

    # sort by hamming_wt()
    def sort_by_wt(self, x, n):
        """
        Sort a vector by index hamming weights using hamming_wt().

        :param x: a vector of indices
        :param n: number of bits per index in ``x``
        :type x: ndarray<int>
        :type n: int
        :return: sorted vector
        :rtype: ndarray<int>
        """

        wts = np.zeros(len(x), dtype=int)
        for i in range(len(x)):
            wts[i] = self.hamming_wt(x[i], n)
        mask = np.argsort(wts)
        return x[mask]

    def inverse_set(self, F, N):
        """
        Find {0,1,...,N-1}\F. This is useful for finding the information set given a frozen set as ``F``.

        :param F: a vector of indices
        :param N: block length
        :type F: ndarray<int>
        :type N: int
        :return: inverted set as a vector
        :rtype: ndarray<int>
        """

        n = int(np.log2(N))
        not_F = []
        for i in range(N):
            if i not in F:
                not_F.append(i)
        return np.array(not_F)

    def subtract_set(self, X, Y):
        """
        Subtraction of two sets.

        :param X: a vector of indices
        :param Y: a vector of indices
        :type X: ndarray<int>
        :type Y: ndarray<int>
        :return: result of subtracted set as a vector
        :rtype: ndarray<int>
        """

        X_new = []
        for x in X:
            if x not in Y:
                X_new.append(x)
        return np.array(X_new)

    def arikan_gen(self, n):
        """
        The n-th kronecker product of [[1, 1], [0, 1]], commonly referred to as Arikan's kernel.

        :param n: log2(N), where N is the block length
        :type n: int
        :return: polar code generator matrix
        :rtype: ndarray<int>
        """

        F = np.array([[1, 1], [0, 1]])
        F_n = F
        for i in range(n - 1):
            F_n = np.kron(F, F_n)
        return F_n

    # Gaussian Approximation helper functions:

    def phi_residual(self, x, val):
        return self.phi(x) - val

    def phi(self, x):
        if x < 10:
            y = -0.4527 * (x ** 0.86) + 0.0218
            y = np.exp(y)
        else:
            y = np.sqrt(3.14159 / x) * (1 - 10 / (7 * x)) * np.exp(-x / 4)
        return y

    def phi_inv(self, y):
        return self.bisection(y, 0, 10000)

    def bisection(self, val, a, b):
        c = a
        while (b - a) >= 0.01:
            # check if middle point is root
            c = (a + b) / 2
            if (self.phi_residual(c, val) == 0.0):
                break

            # choose which side to repeat the steps
            if (self.phi_residual(c, val) * self.phi_residual(a, val) < 0):
                b = c
            else:
                a = c
        return c

    def logQ_Borjesson(self, x):
        a = 0.339
        b = 5.510
        half_log2pi = 0.5 * np.log(2 * np.pi)
        if x < 0:
            x = -x
            y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
            y = np.log(1 - np.exp(y))
        else:
            y = -np.log((1 - a) * x + a * np.sqrt(b + x * x)) - (x * x / 2) - half_log2pi
        return y
