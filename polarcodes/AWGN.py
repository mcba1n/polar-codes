#!/usr/bin/env python

"""
This class simulates an AWGN channel by adding gaussian noise with double-sided noise power.
It updates ``likelihoods`` in `PolarCode` with randomly generated log-likelihood ratios
for ``u`` in `PolarCode`. For puncturing, the likelihoods for the punctured bits given by
``source_set_lookup`` in `PolarCode` will be set to zero. For shortening,
these likelihoods will be set to infinity. Currently only BPSK modulation is supported.
"""

import matplotlib.pyplot as plt
import numpy as np

class AWGN:
    def __init__(self, myPC, Eb_No, plot_noise = False):
        """
        Parameters
        ----------
        myPC: `PolarCode`
            a polar code object created using the `PolarCode` class
        Eb_No: float
            the design SNR in decibels
        plot_noise: bool
            a flag to view the modeled noise

        """

        self.myPC = myPC
        self.Es = myPC.get_normalised_SNR(Eb_No)
        self.No = 1
        self.plot_noise = plot_noise

        tx = self.modulation(self.myPC.u)
        rx = tx + self.noise(self.myPC.N)
        self.myPC.likelihoods = np.array(self.get_likelihoods(rx), dtype=np.float64)

        # change shortened/punctured bit LLRs
        if self.myPC.punct_flag:
            if self.myPC.punct_type == 'shorten':
                self.myPC.likelihoods[self.myPC.source_set_lookup == 0] = np.inf
            elif self.myPC.punct_type == 'punct':
                self.myPC.likelihoods[self.myPC.source_set_lookup == 0] = 0

    def LLR(self, y):
        """
        > Finds the log-likelihood ratio of a received signal.
        LLR = Pr(y=0)/Pr(y=1).

        Parameters
        ----------
        y: float
            a received signal from a gaussian-distributed channel

        Returns
        ----------
        float
            log-likelihood ratio for the input signal ``y``

        """

        return -2 * y * np.sqrt(self.Es) / self.No

    def get_likelihoods(self, y):
        """
        Finds the log-likelihood ratio of an ensemble of received signals using :func:`LLR`.

        Parameters
        ----------
        y: ndarray<float>
            an ensemble of received signals

        Returns
        ----------
        ndarray<float>
            log-likelihood ratios for the input signals ``y``

        """
        return [self.LLR(y[i]) for i in range(len(y))]

    def modulation(self, x):
        """
        BPSK modulation for a bit field.
        "1" maps to +sqrt(E_s) and "0" maps to -sqrt(E_s).

        Parameters
        ----------
        x: ndarray<int>
            an ensemble of information to send

        Returns
        ----------
        ndarray<float>
            modulated signal with the information from ``x``

        """

        return 2 * (x - 0.5) * np.sqrt(self.Es)

    def noise(self, N):
        """
        Generate gaussian noise with a specified noise power.
        For a noise power N_o, the double-side noise power is N_o/2.

        Parameters
        ----------
        N: float
            the noise power

        Returns
        ----------
        ndarray<float>
            white gaussian noise vector

        """

        # gaussian RNG vector
        s = np.random.normal(0, np.sqrt(self.No / 2), size=N)

        # display RNG values with ideal gaussian pdf
        if self.plot_noise:
            num_bins = 1000
            count, bins, ignored = plt.hist(s, num_bins, density=True)
            plt.plot(bins, 1 / (np.sqrt(np.pi * self.No)) * np.exp(- (bins) ** 2 / self.No),
                        linewidth=2, color='r')
            plt.title('AWGN')
            plt.xlabel('Noise, n')
            plt.ylabel('Density')
            plt.legend(['Theoretical', 'RNG'])
            plt.draw()
        return s

    def show_noise(self):
        """
        Trigger showing the gaussian noise. Only works if ``plot_noise`` is True.
        """
        plt.show()