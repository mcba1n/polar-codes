#!/usr/bin/env python

"""
Math provides miscellaneous math operations to the other classes, which are important for polar codes
algorithm implementations.
"""

import numpy as np

class Math:
    def bit_reversed(self, x, n):
        """
        Description:
            Bit-reversal operation.
        Arguments:
            x -- an index.
            n -- number of bits in x.
        Returns:
            Bit-reversed version of x.
        """

        result = 0
        for i in range(n):  # for each bit number
            if (x & (1 << i)):  # if it matches that bit
                result |= (1 << (n - 1 - i))  # set the "opposite" bit in result
        return result

    def logdomain_diff(self, x, y):
        """
        Description:
            Find the difference between x and y in log-domain.
            It uses log1p to improve numerical stability.
        Arguments:
            x, y -- any number in the log-domain.
        Returns:
            The result of x - y.
        """

        if x > y:
            z = x + np.log1p(-np.exp(y - x))
        else:
            z = y + np.log1p(-np.exp(x - y))
        return z

    def logdomain_sum(self, x, y):
        """
        Description:
            Find the addition of x and y in log-domain.
            It uses log1p to improve numerical stability.
        Arguments:
            x, y -- any number in the log-domain.
        Returns:
            The result of x + y.
        """

        if x > y:
            z = x + np.log1p(np.exp(y - x))
        else:
            z = y + np.log1p(np.exp(x - y))
        return z

    def bit_perm(self, x, p, n):
        """
        Description:
            Find the permutation of an index.
        Arguments:
            x -- an index.
            p -- permutation vector, ex: bit-reversal is (0,1,...,n-1)
            n -- number of bits in x.
        Returns:
            The permuted index.
        """

        result = 0
        for i in range(n):
            b = (x >> p[i]) & 1
            result ^= (-b ^ result) & (1 << (n - i - 1))
        return result

    # find hamming weight of an index x
    def hamming_wt(self, x, n):
        """
        Description:
            Find the bit-wise hamming weight of an index.
        Arguments:
            x -- an index.
            n -- number of bits in x.
        Returns:
            The hamming weight.
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
        Description:
            Sort a vector by index hamming weights using hamming_wt().
        Arguments:
            x -- an index.
            n -- number of bits in x.
        Returns:
            The sorted vector.
        """

        wts = np.zeros(len(x), dtype=int)
        for i in range(len(x)):
            wts[i] = self.hamming_wt(x[i], n)
        mask = np.argsort(wts)
        return x[mask]

    def inverse_set(self, F, N):
        """
        Description:
            Find {0,1,...,N-1}\F.
        Arguments:
            x -- an index.
            n -- number of bits in x.
        Returns:
            The inverted set as a vector.
        """

        n = int(np.log2(N))
        not_F = []
        for i in range(N):
            if i not in F:
                not_F.append(i)
        return np.array(not_F)

    def arikan_gen(self, n):
        """
        Description:
            The n-th kronecker product of [[1, 1], [0, 1]],
            commonly referred to as Arikan's kernel.
        Arguments:
            n -- log2(N), where N is the block length.
        Returns:
            Polar code generator matrix.
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
        """
        Description:

            Helper function for the Gaussian Approximation.
        Arguments:
            x --
        Returns:
            The result of phi(x).
        """

        if x < 10:
            y = -0.4527 * (x ** 0.86) + 0.0218
            y = np.exp(y)
        else:
            y = np.sqrt(3.14159 / x) * (1 - 10 / (7 * x)) * np.exp(-x / 4)
        return y

    def phi_inv(self, y):
        """
        Description:
            The inverse of y=phi(x) using the bisection method.
            It returns the inverse by numerically finding the roots of phi(x)-y.
            Depends on bisection() and phi_residual().
            Helper function for the Gaussian Approximation.
        Arguments:
            y -- any number.
        Returns:
            The result of phi_inverse(y).
        """

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