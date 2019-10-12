#!/usr/bin/env python

"""
Simulates an AWGN channel by adding gaussian noise with double-sided noise power.
This class updates PolarCode.likelihoods with the log-likelihood ratios for each bit in PolarCode.u.
For puncturing, the likelihoods for the punctured bits in punct_set_lookup will be set to zero.
For shortening, the likelihoods for the shortened bits in punct_set_lookup will be set to infinity.
Currently only BPSK modulation is supported.
"""

import matplotlib.pyplot as plt
import numpy as np

class AWGN:
    def __init__(self, myPC, Eb_No, plot_noise = False):
        """
        Constructor arguments:
            myPC -- a polar code created using the PolarCode class.
            Eb_No -- the design SNR in decibels.
            plot_noise -- a flag to view the modeled noise.
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
                self.myPC.likelihoods[self.myPC.punct_set_lookup == 0] = np.inf
            elif self.myPC.punct_type == 'punct':
                self.myPC.likelihoods[self.myPC.punct_set_lookup == 0] = 0

    def LLR(self, y):
        """
        Description:
            Finds the log-likelihood ratio of a received signal.
            LLR = Pr(y=0)/Pr(y=1).
        Arguments:
            y -- a modulated signal received from the AWGN channel.
        Returns:
            Log-likelihood ratio.
        """
        return -2 * y * np.sqrt(self.Es) / self.No

    def get_likelihoods(self, y):
        """
        Description:
            Finds the log-likelihood ratio of an ensemble of received signals using LLR().
        Arguments:
            y -- an ensemble of received signals.
        Returns:
            A vector of log-likelihood ratios.
        """
        return [self.LLR(y[i]) for i in range(len(y))]

    def modulation(self, x):
        """
        Description:
            BPSK modulation for a bit field.
            "1" maps to +sqrt(E_s) and "0" maps to -sqrt(E_s).
        Arguments:
            x -- an ensemble of received signals.
        Returns:
            Modulated signal.
        """
        return 2 * (x - 0.5) * np.sqrt(self.Es)

    def noise(self, N):
        """
        Description:
            Generate gaussian noise with a specified noise power.
            For a noise power No, the double-side noise power is No/2.
        Arguments:
            N -- the noise power.
        Returns:
            White gaussian noise vector.
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
        Description:
            Trigger showing the gaussian noise. Only works if self.plot_noise = True.
        """
        plt.show()