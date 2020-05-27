#!/usr/bin/env python

"""
Construct performs the mothercode construction.
It uses the algorithm specified by ``construction_type`` in ``myPC``.
Mothercode constructions supported: Bhattacharyya Bounds, Gaussian Approximation.
"""

import numpy as np
from polarcodes.utils import *

class Construct:
    def __init__(self, myPC, design_SNR, manual=False):
        """
        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the `PolarCode` class
        design_SNR: float
            the design SNR in decibels
        manual: bool
            suppress the constructor init
        """

        if manual:
            return
        else:
            self.update_mpcc(myPC, design_SNR)

    def update_mpcc(self, myPC, design_SNR):
        # select the mothercode construction method
        design_SNR_normalised = myPC.get_normalised_SNR(design_SNR)
        if myPC.construction_type == 'bb':
            z0 = -design_SNR_normalised
            myPC.reliabilities, myPC.frozen, myPC.FERestimate = self.general_pcc(myPC, z0)
        elif myPC.construction_type == 'ga':
            z0 = np.array([4 * design_SNR_normalised] * myPC.N)
            myPC.reliabilities, myPC.frozen, myPC.FERestimate = self.general_ga(myPC, z0)
        myPC.frozen_lookup = myPC.get_lut(myPC.frozen)

    def general_pcc(self, myPC, z0):
        """
        Polar code construction using Bhattacharyya Bounds. Each bit-channel can have different parameters.
        Supports shortening by adding extra cases for infinite likelihoods.

        Parameters
        ----------
        z0: ndarray<float>, float
            a vector of the initial Bhattacharyya parameters in the log-domain, -E_b/N_o.
            > Note that this SNR should be normalised using `get_normalised_SNR` in `PolarCode`

        Returns
        ----------
        ndarray<int>, ndarray<int>
            channel reliabilities in log-domain (least reliable first), and the frozen indices

        -------------
        **References:**

        * Vangala, H., Viterbo, E., & Hong, Y. (2015). A Comparative Study of Polar Code Constructions for the AWGN Channel. arXiv.org. Retrieved from http://search.proquest.com/docview/2081709282/

        """

        n = myPC.n
        z = np.zeros((myPC.N, n + 1))
        z[:, 0] = z0  # initial channel states

        for j in range(1, n + 1):
            u = 2 ** j  # number of branches at depth j
            for t in range(0, myPC.N, u):  # loop over top branches at this stage
                for s in range(int(u / 2)):
                    k = t + s
                    z_top = z[k, j - 1]
                    z_bottom = z[k + int(u / 2), j - 1]

                    # shortening infinity cases
                    if z_top == -np.inf and z_bottom != -np.inf:
                        z[k, j] = z_bottom
                        z[k + int(u / 2), j] = -np.inf
                    elif z_top != -np.inf and z_bottom == -np.inf:
                        z[k, j] = z_top
                        z[k + int(u / 2), j] = -np.inf
                    elif z_top == -np.inf and z_bottom == -np.inf:
                        z[k, j] = -np.inf
                        z[k + int(u / 2), j] = -np.inf
                    # principal equations
                    else:
                        z[k, j] = logdomain_diff(logdomain_sum(z_top, z_bottom), z_top + z_bottom)
                        z[k + int(u / 2), j] = z_top + z_bottom

        reliabilities = np.argsort(-z[:, n], kind='mergesort')   # ordered by least reliable to most reliable
        frozen = np.argsort(z[:, n], kind='mergesort')[myPC.K:]     # select N-K least reliable channels
        FERest = self.FER_estimate(frozen, z[:, n])
        myPC.z = z[:, n]
        return reliabilities, frozen, FERest

    def perfect_pcc(self, myPC, p):
        """
        Boolean expression approach to puncturing pattern construction.

        Parameters
        ----------
        p: ndarray<int>
            lookup table for coded puncturing bits. "0" => punctured, "1" => information.
            For shortening, take the complement of p.

        Returns
        ----------
        ndarray<int>
            a lookup table for which uncoded bits should be punctured.
            For shortening, take the complement of this output.

        -------------
        **References:**

        * Song-Nam, H., & Hui, D. (2018). On the Analysis of Puncturing for Finite-Length Polar Codes: Boolean Function Approach. arXiv.org. Retrieved from http://search.proquest.com/docview/2071252269/

        """

        n = int(np.log2(myPC.N))
        z = np.zeros((myPC.N, n + 1), dtype=int)
        z[:, 0] = p

        for j in range(1, n + 1):
            u = 2 ** j  # number of branches at depth j
            # loop over top branches at this stage
            for t in range(0, myPC.N, u):
                for s in range(int(u / 2)):
                    k = t + s
                    z_top = z[k, j - 1]
                    z_bottom = z[k + int(u / 2), j - 1]
                    z[k, j] = z_top & z_bottom
                    z[k + int(u / 2), j] = z_top | z_bottom
        return z[:, n]

    def general_ga(self, myPC, z0):
        """
        Polar code construction using density evolution with the Gaussian Approximation. Each channel can have different parameters.

        Parameters
        ----------
        z0: ndarray<float>, float
            a vector of the initial mean likelihood densities, 4 * E_b/N_o.
            > Note that this SNR should be normalised using `get_normalised_SNR` in `PolarCode`

        Returns
        ----------
        ndarray<int>, ndarray<int>
            channel reliabilities in log-domain (least reliable first), and the frozen indices

        -------------
        **References:**

        * Trifonov, P. (2012). Efficient Design and Decoding of Polar Codes. IEEE Transactions on Communications, 60(11), 3221–3227. https://doi.org/10.1109/TCOMM.2012.081512.110872

        * Vangala, H., Viterbo, E., & Hong, Y. (2015). A Comparative Study of Polar Code Constructions for the AWGN Channel. arXiv.org. Retrieved from http://search.proquest.com/docview/2081709282/

        """

        z = np.zeros((myPC.N, myPC.n + 1))
        z[:, 0] = z0  # initial channel states

        for j in range(1, myPC.n + 1):
            u = 2 ** j  # number of branches at depth j
            # loop over top branches at this stage
            for t in range(0, myPC.N, u):
                for s in range(int(u / 2)):
                    k = t + s
                    z_top = z[k, j - 1]
                    z_bottom = z[k + int(u / 2), j - 1]

                    z[k, j] = phi_inv(1 - (1 - phi(z_top)) * (1 - phi(z_bottom)))
                    z[k + int(u / 2), j] = z_top + z_bottom

        m = np.array([logQ_Borjesson(0.707*np.sqrt(z[i, myPC.n])) for i in range(myPC.N)])
        reliabilities = np.argsort(-m, kind='mergesort')    # ordered by least reliable to most reliable
        frozen = np.argsort(m, kind='mergesort')[myPC.K:]     # select N-K least reliable channels
        FERest = self.FER_estimate(frozen, m)
        myPC.z = m
        return reliabilities, frozen, FERest

    def FER_estimate(self, frozen, z):
        FERest = 0
        for i in range(len(z)):
            if i not in frozen:
                FERest = FERest + np.exp(z[i]) - np.exp(z[i]) * FERest
        return FERest