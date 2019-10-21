#!/usr/bin/env python

"""
A polar decoder class. Currently only Successive Cancellation Decoder (SCD) is supported.
"""

import numpy as np
from classes.Math import Math
from timeit import default_timer as timer

class Decode(Math):
    def __init__(self, myPC, decoder_name = 'scd'):
        """
        Constructor arguments:
            myPC -- a polar code created using the PolarCode class.
            decoder_name -- name of decoder to use (default is 'scd').
        """

        self.myPC = myPC

        # select decoding algorithm
        if decoder_name == 'scd':
            self.polar_decode(self.myPC.likelihoods)

    def upper_llr(self, l1, l2):
        """
        Description:
            Update top branch LLR in the log-domain.
            This function supports shortening by checking for infinite LLR cases.
        Arguments:
            l1 -- input LLR corresponding to the top branch.
            l2 -- input LLR corresponding to the bottom branch.
        Returns:
            The top branch LLR at the next stage of the decoding tree.
        """

        # check for infinite LLR, used in shortening
        if l1 == np.inf and l2 != np.inf:
                return l2
        elif l1 != np.inf and l2 == np.inf:
            return l1
        elif l1 == np.inf and l2 == np.inf:
            return np.inf
        else:
            # principal decoding equation
            return self.logdomain_sum(l1 + l2, 0) - self.logdomain_sum(l1, l2)

    def lower_llr(self, l1, l2, b):
        """
        Description:
            Update bottom branch LLR in the log-domain.
            This function supports shortening by checking for infinite LLR cases.
        Arguments:
            l1 -- input LLR corresponding to the top branch.
            l2 -- input LLR corresponding to the bottom branch.
            b -- the decoded bit of the top branch.
        Returns:
            The bottom branch LLR at the next stage of the decoding tree.
        """

        if b == 0:
            # check for infinite LLRs, used in shortening
            if l1 == np.inf or l2 == np.inf:
                return np.inf
            else:
                # principal decoding equation
                return l1 + l2
        elif b == 1:
            # Principal decoding equation
            return l1 - l2
        return np.nan

    def update_LLR(self, x_ind):
        """
        Description:
            Update all possible likelihoods at stage (n-j).
            This is a non-recursive implementation of update_LLR_recursive().
        Arguments:
            x_ind -- the root index.
        """

        for j in range(self.myPC.n - 1, -1, -1):
            s = 2 ** (self.myPC.n - j)  # partition length
            half_s = int(s / 2)  # half partition length

            for i in range(x_ind, self.myPC.N, s):  # do not need to update indices less than x_ind
                l = i % s
                if l < half_s:  # upper branch
                    l1 = self.L[i, j + 1]
                    l2 = self.L[i + half_s, j + 1]
                    self.L[i, j] = self.upper_llr(self.L[i, j + 1], l2)
                else:  # lower branch
                    l1 = self.L[i, j + 1]
                    l2 = self.L[i - half_s, j + 1]
                    b = self.B[i - half_s, j]
                    self.L[i, j] = self.lower_llr(l1, l2, b)

    def update_LLR_recursive(self, i, j):
        """
        Description:
            Update all possible likelihoods at stage (n-j).
            This is a recursive implementation.
        Arguments:
            x_ind -- the root index.
        """

        s = 2 ** (self.myPC.n - j)  # partition size
        half_s = int(s / 2)
        l = i % s

        if l < half_s:  # upper branch
            if np.isnan(self.L[i, j + 1]):
                self.update_LLR_recursive(i, j + 1)
            if np.isnan(self.L[i + half_s, j + 1]):
                self.update_LLR_recursive(i + half_s, j + 1)

            # evaluate the f function
            l1 = self.L[i, j + 1]
            l2 = self.L[i + half_s, j + 1]
            self.L[i, j] = self.upper_llr(self.L[i, j + 1], l2)
        else:  # lower branch
            # evaluate the g function
            l1 = self.L[i, j + 1]
            l2 = self.L[i - half_s, j + 1]
            b = self.B[i - half_s, j]
            self.L[i, j] = self.lower_llr(l1, l2, b)

    def update_bits(self, x_ind):
        """
        Description:
            Update all possible bits at stage (n-j).
            This is a non-recursive implementation.
        Arguments:
            x_ind -- the root index.
        """

        next_i = [x_ind]  # store branch indices to update in next stage
        for j in range(self.myPC.n):
            s = 2 ** (self.myPC.n - j)  # partition length
            half_s = int(s / 2)  # half partition length

            next_i_temp = []
            for i in next_i:
                l = i % s
                if l >= half_s:  # lower branch
                    # propagate upper branch bit
                    self.B[i - half_s, j + 1] = int(self.B[i, j]) ^ int(self.B[i - half_s, j])
                    self.B[i, j + 1] = self.B[i, j]

                    # append branches to search in next stage
                    next_i_temp.append(i)
                    next_i_temp.append(i - half_s)
                next_i = next_i_temp  # update search array

    def polar_decode(self, y):
        """
        Description:
            Successive Cancellation Decoder.
            The decoded message is set to self.myPC.message_received.
            The decoder will use the frozen set as defined by self.myPC.frozen.
            Depends on update_LLR() and update_bits().
            Reference:
        Arguments:
            y -- a vector of likelihoods at the channel output.
        """

        # decoding initial params
        self.L = np.full((self.myPC.N, self.myPC.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.myPC.N, self.myPC.n + 1), np.nan)
        self.L[:, self.myPC.n] = y

        # decode rung by rung
        for i in range(0, self.myPC.N):
            # evaluate tree of LLRs for bit i
            l = self.bit_reversed(i, self.myPC.n)  # use bit reversal of i for the "natural order"
            self.update_LLR_recursive(l, 0)

            # make hard decision at output (first column of B)
            if l in self.myPC.frozen:
                self.B[l, 0] = 0
            else:
                if self.L[l, 0] >= 0:
                    self.B[l, 0] = 0
                else:
                    self.B[l, 0] = 1

            # propagate the hard decision just made
            self.update_bits(l)

        x_noisy = self.B[:, 0].astype(int)
        self.myPC.message_received = x_noisy[self.myPC.frozen_lookup == 1]
