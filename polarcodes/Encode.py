#!/usr/bin/env python

"""
A polar encoder class. Currently only non-systematic encoding is supported.
"""

import numpy as np
from polarcodes.Math import Math

class Encode(Math):
    def __init__(self, myPC, encoder_name = 'polar_encode'):
        """
        :param myPC: a polar code object created using the :class:`PolarCode` class
        :param encoder_name: the name of the polar encoder implementation.
                            'polar_encode' => a non_recursive implementation (default).
                            'polar_encode_recursive' => a recursive implementation.
        :type myPC: :class:`PolarCode`
        :type encoder_name: string
        """

        self.myPC = myPC
        if encoder_name == 'polar_encode':
            self.polar_encode()
        elif encoder_name == 'polar_encode_recursive':
            self.polar_encode2(0, myPC.N-1)

    def polar_encode2(self, i1, i2):
        """
        Encodes a message using polar coding with a recursive implementation.
        The message ``x`` is encoded using in-place operations of output ``u`` in ``myPC``.
        The initial call of :func:`polar_encode2` should set (i1, i2) = (0, N-1) for a block length N.
        For example, the second partition indices will be (0, N/2-1) and (N/2, N-1).

        :param i1: start index of the partition
        :param i2: end index of the partition
        :type i1: int
        :type i2: int
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