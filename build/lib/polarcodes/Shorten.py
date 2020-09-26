#!/usr/bin/env python

"""
A class dedicated to shortening. This means that the likelihoods for each coded shortened bit are set to infinity at the channel output given by class AWGN.
Shortening techniques supported: Wang-Liu Shortening (WLS), Bit-Reversal Shortening (BRS), Bioglio-Gabry-Land Shortening (BGL), and Permuted WLS (PWLS).
"""

import numpy as np
from polarcodes.utils import *
from polarcodes.Construct import Construct

class Shorten(Construct):
    def __init__(self, myPC, design_SNR, manual=False):
        """

        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the :class:`PolarCode` class
        design_SNR: float
            the design SNR in decibels
        manual: bool
            suppress the constructor init

        """

        super().__init__(myPC, design_SNR, True)
        if manual:
            return
        else:
            self.update_spcc(myPC, design_SNR)

    def update_spcc(self, myPC, design_SNR):
        # select shortening construction method
        if myPC.punct_algorithm == 'brs':  # BRS shortening
            myPC.punct_set = self.brs_pattern(myPC)
            myPC.source_set = myPC.punct_set
        elif myPC.punct_algorithm == 'wls':  # WLS shortening
            myPC.punct_set = self.wls_pattern(myPC)
            myPC.source_set = myPC.punct_set
        elif myPC.punct_algorithm == 'bgl':  # BGL shortening
            myPC.punct_set = self.bgl_pattern(myPC)
            myPC.source_set = myPC.punct_set
        elif myPC.punct_algorithm == 'perm':  # Perm shortening
            myPC.punct_set = self.perm(myPC)
            myPC.source_set = myPC.punct_set

        myPC.punct_set_lookup = myPC.get_lut(myPC.punct_set)
        myPC.source_set_lookup = myPC.get_lut(myPC.source_set)

        # decide if we want a puncturing-dependent frozen set
        if not myPC.update_frozen_flag:
            self.update_mpcc(myPC, design_SNR)
        else:
            self.shortened_pcc(myPC, design_SNR)
        myPC.frozen = self.frozen_from_pattern(myPC)
        myPC.frozen_lookup = myPC.get_lut(myPC.frozen)
        myPC.FERestimate = self.FER_estimate(myPC.frozen, myPC.z)

    def shortened_pcc(self, myPC, design_SNR):
        """
        Find the shortened polar code construction and update ``frozen`` in ``myPC``.
        This is not strictly necessary, since many shortening patterns work just fine with the mothercode reliabilities.

        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the :class:`PolarCode` class
        design_SNR: float
            the design SNR in decibels

        """

        # select the construction method
        design_SNR_normalised = myPC.get_normalised_SNR(design_SNR)
        if myPC.construction_type == 'bb':
            z0 = np.array([-design_SNR_normalised] * myPC.N)
            z0[myPC.punct_set] = -np.inf
            myPC.reliabilities, myPC.frozen, myPC.FERestimate = self.general_pcc(myPC, z0)
        elif myPC.construction_type == 'ga':
            z0 = np.array([4 * design_SNR_normalised] * myPC.N)
            z0[myPC.punct_set] = np.inf
            myPC.reliabilities, myPC.frozen, myPC.FERestimate = self.general_ga(myPC, z0)

    def wls_pattern(self, myPC):
        """
        Wang-Liu Shortening (WLS). The common pattern from the Wang-Liu algorithm.

        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the :class:`PolarCode` class

        Returns
        ----------
        ndarray<int>
            WLS shortening set

        -------------
        **References:**

        * Runxin Wang, & Rongke Liu. (2014). A Novel Puncturing Scheme for Polar Codes. IEEE Communications Letters, 18(12), 2081–2084. https://doi.org/10.1109/LCOMM.2014.2364845

        """

        punct_set = np.array(range(myPC.N - myPC.s, myPC.N))
        return punct_set

    def brs_pattern(self, myPC):
        """
        Bit-Reversal Shortening (BRS). A known high-performing shortening set, often called RQUP.

        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the `PolarCode` class

        Returns
        ----------
        ndarray<int>
            BRS shortening set

        -------------
        **References:**

        * Niu, Dai, Chen, Lin, Zhang, & Vasilakos. (2017). Rate-Compatible Punctured Polar Codes: Optimal Construction Based on Polar Spectra. arXiv.org. Retrieved from http://search.proquest.com/docview/2076458581/

        """
        punct_set_last = self.wls_pattern(myPC)
        punct_set = np.array([bit_reversed(i, myPC.n) for i in punct_set_last])
        return punct_set

    def frozen_from_pattern(self, myPC):
        """
        Forces the frozen bits to include the corresponding puncturing source bits in ``source_set`` in ``myPC``.

        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the :class:`PolarCode` class

        Returns
        ----------
        ndarray<int>
            the new frozen set, that is typically assigned to ``frozen`` in ``myPC``.

        """

        R_m = []
        for i in range(myPC.N):    # add elements from R not in punct_set to R_m
            if myPC.reliabilities[i] not in myPC.source_set:
                R_m.append(myPC.reliabilities[i])
        t = myPC.M - myPC.K   # number of frozen bits left to select
        frozen = np.array(np.append(np.array(R_m[:t]), myPC.source_set))   # first t bits of R_m, then append S
        return frozen

    def perm(self, myPC):
        """
        Bit-wise permutation of the indices of the WLS pattern. This has been shown to produce other reciprocal
        shortening patterns, and so it is useful in enumerating them all for analysis.

        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the :class:`PolarCode` class

        Returns
        ----------
        ndarray<int>
            the permuted shortening pattern for ``myPC``

        """

        punct_set_last = self.wls_pattern(myPC)
        punct_set = np.array(bit_perm(punct_set_last, myPC.perm, myPC.n))  # specify perm before construction
        return punct_set

    def wang_liu(self, myPC):
        N = 2**myPC.n
        G = arikan_gen(myPC.n)
        s = []

        for r in range(myPC.punct_set):
            for i in range(N):
                row = G[i, :]
                row_wt = np.sum(row)
                if row_wt == 1:
                    j = (np.where(row==1)[0]).item()
                    G[i, :] = np.zeros(N)
                    G[:, j] = np.zeros(N)
                    s.append(i)
                    break
        return np.array(s)

    def bgl_pattern(self, myPC):
        """
        Bioglio-Gabry-Land (BGL) Shortening

        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the :class:`PolarCode` class
        ndarray<int>
            the BGL shortening set

        -------------
        **References:**

        * Bioglio, V., Gabry, F., & Land, I. (2017). Low-Complexity Puncturing and Shortening of Polar Codes. arXiv.org. Retrieved from http://search.proquest.com/docview/2075581442/

        """

        num_bits = myPC.n
        s = myPC.N - myPC.M  # number bits to shorten
        reversed_indices = np.array([bit_reversed(i, num_bits) for i in myPC.reliabilities])
        punct_set = np.array(reversed_indices[-s:])  # last s bits of reversed_indices
        return punct_set