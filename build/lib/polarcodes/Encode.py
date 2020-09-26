#!/usr/bin/env python

"""
A polar encoder class with systematic and non-systematic methods.
"""

import numpy as np
from polarcodes.utils import *

class Encode:
    def __init__(self, myPC, encoder_name = 'polar_encode'):
        """
        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the :class:`PolarCode` class
        encoder_name: string
            the name of the polar encoder implementation.
                            'polar_encode' => a non_recursive implementation (default).
                            'polar_encode_recursive' => a recursive implementation.
        """

        self.myPC = myPC
        if encoder_name == 'polar_encode':
            self.polar_encode()
        elif encoder_name == 'polar_encode_recursive':
            self.polar_encode2(0, myPC.N-1)
        elif encoder_name == 'systematic_encode':
            if myPC.T == None:
                self.systematic_init()
            self.systematic_encode()

    def polar_encode2(self, i1, i2):
        """
        Encodes a message using polar coding with a recursive implementation.
        The message ``x`` is encoded using in-place operations of output ``u`` in ``myPC``.
        The initial call of `polar_encode2` should set (i1, i2) = (0, N-1) for a block length N.
        For example, the second partition indices will be (0, N/2-1) and (N/2, N-1).

        Parameters
        ----------
        i1: int
            start index of the partition
        i2: int
            end index of the partition

        """
        h_shift = int((i2 - i1 + 1) / 2)  # length of each new partition
        mid = i1 + h_shift  # right-aligned mid-point

        for k in range(i1, mid):
            self.myPC.u[k] = self.myPC.u[k] ^ self.myPC.u[k + h_shift]  # add right partition to the left partition (mod 2)

        if h_shift >= 2:
            self.polar_encode2(i1, mid - 1)     # recursion on left partition
            self.polar_encode2(mid, i2)         # recursion on right partition

    # Polar encode x without using recursion
    # x = message bit field
    def polar_encode(self):
        """
        Encodes a message using polar coding with a non-recursive implementation.
        The message ``x`` is encoded using in-place operations of output ``u`` in ``myPC``.
        """

        # loop over the M stages
        n = self.myPC.N
        for i in range(self.myPC.N):
            if n == 1:  # base case: when partition length is 1
                break
            n_split = int(n / 2)

            # select the first index for each split partition, we always have n=2*n_split
            for p in range(0, self.myPC.N, n):
                # loop through left partitions and add the right partitions for a previous partition
                for k in range(n_split):
                    l = p + k
                    self.myPC.u[l] = self.myPC.u[l] ^ self.myPC.u[l + n_split]
            n = n_split

    def systematic_encode(self):
        """
        Encodes a message using systematic polar encode by matrix multiplications.
        The input message is transformed using matrix T such that the subsequent polar transformation
        will result in the message bits being in the codeword, i.e. a systematic polar code.
        """

        # systematic polar encoding operations
        x = np.array([self.myPC.x])
        v = np.mod(np.dot(self.myPC.T, x.T), 2)
        self.myPC.u = np.transpose(np.mod(np.dot(self.myPC.F, v), 2))[0]

    def systematic_init(self):
        """
        Calculate the systematic T transformation required in ``Encode.systematic_encode``,
        then store it in ``self.myPC`` for quick access by the systematic encoding function.
        """

        # T = [I_(N-K,N)|F_(K,N)]
        # multiply with the inverse of sub-matrix F_A for indices in A,
        # leaving the other bits of u unchanged (rows in A.T set to identity)
        T = np.eye(self.myPC.N, dtype=int)
        A = inverse_set(self.myPC.frozen, self.myPC.N)
        T[A, :] = self.myPC.F[A, :]
        self.myPC.T = T
