#!/usr/bin/env python

"""
Construct performs the mothercode construction.
It uses the algorithm specified by PolarCode.construction_type. If the puncturing flag PolarCode.punct_flag is True
the input PolarCode instance will go to class Shorten.
Mothercode constructions supported: Bhattacharyya Bounds, Gaussian Approximation.
"""

import numpy as np
from classes.Math import Math
from classes.Shorten import Shorten

class Construct(Math):
    def __init__(self, myPC, design_SNR, manual = False):
        """
        Constructor arguments:
            myPC -- a polar code created using the PolarCode class.
            design_SNR -- E_b/N_o (in decibels) to be used for construction.
            manual -- set to True if you want this class init to do nothing.
        """

        if manual:
            return
        self.myPC = myPC
        design_SNR_normalised = self.myPC.get_normalised_SNR(design_SNR)

        # select the mothercode construction method
        if self.myPC.construction_type == 'bb':
            self.myPC.z0 = -design_SNR_normalised
            self.myPC.reliabilities, self.myPC.frozen = self.general_pcc(self.myPC.z0)
        elif self.myPC.construction_type == 'ga':
            self.myPC.z0 = np.array([4 * design_SNR_normalised] * self.myPC.N)
            self.myPC.reliabilities, self.myPC.frozen = self.general_ga(self.myPC.z0)

        if self.myPC.punct_flag == True:
            Shorten(self.myPC)

        # update the frozen lookup
        self.myPC.frozen_lookup = np.ones(self.myPC.N, dtype=int)
        self.myPC.frozen_lookup[self.myPC.frozen] = 0

    def pcc(self, z0):
        """
        Description:
            Polar code construction using Bhattacharyya Bounds.
            All N channels have the same parameters.
            Reference:
        Arguments:
            z0 -- Initial Bhattacharyya parameter in log-domain, -E_b/N_o
        Returns:
            Channel reliabilities in log-domain (least reliable first), and the frozen indices.
        """

        n = self.myPC.n
        z = np.zeros((self.myPC.N, n + 1))
        z[0, 0] = z0  # initial channel state

        # evaluate tree up to depth (n+1) and store in z
        for j in range(1, n + 1):
            u = 2 ** j  # number of branches at depth j
            for t in range(int(u / 2)):  # index for each top tree branch
                z_prev = z[t, j - 1]
                z[t, j] = self.logdomain_diff(np.log(2) + z_prev, 2 * z_prev)
                z[int(u / 2) + t, j] = 2 * z_prev

        reliabilities = np.argsort(-z[:, n], kind='mergesort')   # ordered by least reliable to most reliable
        frozen = np.argsort(z[:, n], kind='mergesort')[self.myPC.K:]  # argmax{z_n} for (n-k) bit channels
        return reliabilities, frozen

    def general_pcc(self, z0):
        """
        Description:
            Polar code construction using Bhattacharyya Bounds.
            Each channel can have different parameters.
            Supports shortening by adding extra cases for infinite likelihoods.
            Reference:
        Arguments:
            z0 -- A vector of the initial Bhattacharyya parameters in log-domain, -E_b/N_o.
        Returns:
            Channel reliabilities in log-domain (least reliable first), and the frozen indices.
        """

        n = self.myPC.n
        z = np.zeros((self.myPC.N, n + 1))
        z[:, 0] = z0  # initial channel states

        for j in range(1, n + 1):
            u = 2 ** j  # number of branches at depth j
            # loop over top branches at this stage
            for t in range(0, self.myPC.N, u):
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
                    else:
                        z[k, j] = self.logdomain_diff(self.logdomain_sum(z_top, z_bottom), z_top + z_bottom)
                        z[k + int(u / 2), j] = z_top + z_bottom

        self.myPC.b_params = z[:,n]
        reliabilities = np.argsort(-z[:, n], kind='mergesort')   # ordered by least reliable to most reliable
        frozen = np.argsort(z[:, n], kind='mergesort')[self.myPC.K:]  # arg max of z for (n-k) bit channels
        return reliabilities, frozen

    # useful for verifying which bits to shorten before encoding
    def perfect_pcc(self, p):
        """
        Description:
            Boolean expression approach to puncturing pattern construction.
            Reference:
        Arguments:
            p -- Lookup table for coded puncturing bits. "0" => punctured, "1" => information.
            For shortening, take the complement of p.
        Returns:
            A lookup table for which uncoded bits should be punctured.
            For shortening, take the complement of this output.
        """

        n = int(np.log2(self.myPC.N))
        z = np.zeros((self.myPC.N, n + 1), dtype=int)
        z[:, 0] = p

        for j in range(1, n + 1):
            u = 2 ** j  # number of branches at depth j
            # loop over top branches at this stage
            for t in range(0, self.myPC.N, u):
                for s in range(int(u / 2)):
                    k = t + s
                    z_top = z[k, j - 1]
                    z_bottom = z[k + int(u / 2), j - 1]
                    z[k, j] = z_top & z_bottom
                    z[k + int(u / 2), j] = z_top | z_bottom
        return z[:, n]

    # Generalised Gaussian Approximation
    def general_ga(self, z0):
        """
        Description:
            Polar code construction using density evolution with the Gaussian Approximation.
            Each channel can have different parameters.
            Reference:
        Arguments:
            z0 -- a vector of the initial mean likelihood densities, 4 * E_b/N_o.
        Returns:
            Channel reliabilities in log-domain (least reliable first), and the frozen indices.
        """

        z = np.zeros((self.myPC.N, self.myPC.n + 1))
        z[:, 0] = z0  # initial channel states

        for j in range(1, self.myPC.n + 1):
            u = 2 ** j  # number of branches at depth j
            # loop over top branches at this stage
            for t in range(0, self.myPC.N, u):
                for s in range(int(u / 2)):
                    k = t + s
                    z_top = z[k, j - 1]
                    z_bottom = z[k + int(u / 2), j - 1]

                    z[k, j] = self.phi_inv(1 - (1 - self.phi(z_top)) * (1 - self.phi(z_bottom)))
                    z[k + int(u / 2), j] = z_top + z_bottom

        m = np.array([self.logQ_Borjesson(0.707*np.sqrt(z[i, self.myPC.n])) for i in range(self.myPC.N)])
        reliabilities = np.argsort(-m, kind='mergesort')   # ordered by least reliable to most reliable
        frozen = np.argsort(m, kind='mergesort')[self.myPC.K:]  # arg max of z for (n-k) bit channels
        self.myPC.log_ber = m
        return reliabilities, frozen
