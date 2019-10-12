#!/usr/bin/env python

"""
A class dedicated to shortening.
This means that the likelihoods for each coded shortened bit are set to infinity at the channel output given by class AWGN.
Shortening techniques supported: Wang-Liu Shortening (WLS), Bit-Reversal Shortening (BRS), and permutations of WLS.
"""

import numpy as np
from classes.Math import Math

class Shorten(Math):
    def __init__(self, myPC):
        """
        Constructor arguments:
            myPC -- a polar code created using the PolarCode class.
        """

        self.myPC = myPC

        # select shortening construction method
        if self.myPC.punct_algorithm == 'brs':  # BRS shortening
            self.myPC.punct_set = self.brs_pattern()
            self.myPC.recip_flag = True
        elif self.myPC.punct_algorithm == 'wls':  # WLS shortening
            self.myPC.punct_set = self.last_pattern()
            self.myPC.recip_flag = True
        elif self.myPC.punct_algorithm == 'perm':  # Perm shortening
            self.myPC.punct_set = self.perm()
            self.myPC.recip_flag = True

        # reciprocal patterns force frozen set to be subset of S
        if self.myPC.recip_flag == True:
            self.myPC.frozen = self.frozen_from_pattern(self.myPC.punct_set)

        # update the pattern lookup
        self.myPC.punct_set_lookup = np.ones(self.myPC.N, dtype=int)
        self.myPC.punct_set_lookup[self.myPC.punct_set] = 0

    def last_pattern(self):
        """
        Description:
            Wang-Liu Shortening (WLS).
            The common pattern from the Wang-Liu algorithm.
        Returns:
            The WLS shortening pattern for myPC.
        """

        punct_set = np.array(range(self.myPC.N - self.myPC.s, self.myPC.N))
        return punct_set

    # A known high-performing shortening pattern
    def brs_pattern(self):
        """
        Description:
            Bit-Reversal Shortening (BRS).
        Returns:
            The BRS shortening pattern for myPC.
        """
        punct_set_last = self.last_pattern()
        punct_set = [self.bit_reversed(i, self.myPC.n) for i in punct_set_last]
        return punct_set

    def frozen_from_pattern(self, punct_set):
        """
        Description:
            Assumes a reciprocal shortening pattern given and forces the frozen bits to include the shortening set.
        Arguments:
            punt_set -- PolarCode.punt_set.
        Returns:
            The new frozen set, that is typically stored in PolarCode.frozen.
        """

        R_m = []
        for i in range(self.myPC.N):    # add elements from R not in S to Rm
            if self.myPC.reliabilities[i] not in punct_set:
                R_m.append(self.myPC.reliabilities[i])

        t = self.myPC.M - self.myPC.K   # number of frozen bits left to select
        frozen = np.append(np.array(R_m[:t]), punct_set)   # first t bits of R_m, then append S
        return frozen

    # Generate random permutation to find a random reciprocal pattern
    def perm(self):
        """
        Description:
            Bit-wise permutation of the indices of the WLS pattern.
            The permutation is specified by myPC.perm.
        Returns:
            The permuted shortening pattern for myPC.
        """
        punct_set_last = self.last_pattern()
        punct_set = self.bit_perm(punct_set_last, self.myPC.perm, self.myPC.n)  # specify perm before construction
        return punct_set

    def wang_liu(self):
        """
        Description:
            The Wang-Liu Algorithm.
            This function is for educational purposes, since last_pattern() returns the same result.
            However, new patterns can arise if you change the order in which the row of weight one is chosen
            -- BRS is one such example.
        Returns:
            The Wang-Liu shortening pattern.
        """

        N = 2**self.myPC.n
        G = self.arikan_gen(2)
        s = []

        for r in range(self.myPC.punct_set):
            for i in range(N):
                row = G[i, :]
                row_wt = np.sum(row)
                if row_wt == 1:
                    j = (np.where(row==1)[0]).item()
                    G[i, :] = np.zeros(N)
                    G[:, j] = np.zeros(N)
                    s.append(i)
                    break
        return s