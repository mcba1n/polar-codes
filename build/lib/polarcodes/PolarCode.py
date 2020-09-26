#!/usr/bin/env python

"""
An object that encapsulates all of the parameters required to define a polar code.
This object must be given to the following classes: `AWGN`, `Construct`, `Decode`, `Encode`, `GUI`, `Shorten`.
"""

import numpy as np
from polarcodes.utils import *
from polarcodes.Construct import Construct
from polarcodes.Shorten import Shorten
from polarcodes.Encode import Encode
from polarcodes.Decode import Decode
from polarcodes.AWGN import AWGN
import json
import matplotlib.pyplot as plt
import threading
import tkinter as tk

class PolarCode:
    """
    Attributes
    ----------
    N: int
        the mothercode block length
    M: int
        the block length (after puncturing)
    K: int
        the code dimension
    n: int
        number of bits per index
    s: int
        number of shortened bit-channels
    reliabilities: ndarray<int>
        reliability vector (least reliable to most reliable)
    frozen: ndarray<int>
        the frozen bit indices
    frozen_lookup: ndarray<int>
        lookup table for the frozen bits
    x: ndarray<int>
        the uncoded message with frozen bits
    construction_type: string
        the mothercode construction type
    message_received: ndarray<int>
        the decoded message received from a channel
    punct_flag: bool
        whether or not the code is punctured
    simulated_snr: ndarray<float>
        the SNR values simulated
    simulated_fer: ndarray<float>
        the FER values for the SNR values in ``simulated_snr`` using `simulate`
    simulated_ber: ndarray<float>
        the BER values for the SNR values in ``simulated_snr`` using `simulate`
    punct_type: string
        'punct' for puncturing, and 'shorten' for shortening
    punct_set: ndarray<int>
        the coded punctured indices
    punct_set_lookup: ndarray<int>
        lookup table for ``punct_set``
    source_set: ndarray<int>
        the uncoded punctured indices
    source_set_lookup: ndarray<int>
        lookup table for ``source_set``
    punct_algorithm: string
        the name of a puncturing algorithm. Options: {'brs', 'wls', 'bgl', 'perm'}
    update_frozen_flag: bool
        whether or not to update the frozen indices after puncturing
    recip_flag: bool
        True if ``punct_set`` equals ``source_set``

    """

    def __init__(self, M, K, punct_params=('', '', [], [], None,)):
        """
        Parameters
        ----------
        M: int
            the block length (after puncturing)
        K: int
            the code dimension
        punct_params: tuple
            a tuple to completely specify the puncturing parameters (if required).
            The syntax is (``punct_type``, ``punct_algorithm``, ``punct_set``, ``source_set``, ``update_frozen_flag``)
        """

        self.initialise_code(M, K, punct_params)
        self.status_bar = None  # set by the GUI so that the simulation progress can be tracked
        self.gui_widgets = []

    def initialise_code(self, M, K, punct_params):
        """
        Initialise the code with a set of parameters the same way as the constructor.
        Call this any time you want to change the code rate.
        """

        # mothercode parameters
        self.M = M
        self.N = int(2**(np.ceil(np.log2(M))))
        self.n = int(np.log2(self.N))
        self.F = arikan_gen(self.n)
        self.K = K
        self.s = self.N - self.M
        self.reliabilities = np.array([])
        self.frozen = np.array([])
        self.frozen_lookup = np.array([])
        self.x = np.zeros(self.N, dtype=int)
        self.u = np.zeros(self.N, dtype=int)
        self.construction_type = 'bb'
        self.message_received = np.array([])
        self.punct_flag = False if self.M == self.N else True
        self.simulated_snr = np.array([])
        self.simulated_fer = np.array([])
        self.simulated_ber = np.array([])
        self.FERestimate = 0
        self.T = None

        # puncturing parameters
        self.punct_type = punct_params[0]
        self.punct_set = np.array(punct_params[2])
        self.punct_set_lookup = self.get_lut(punct_params[2])
        self.source_set = np.array(punct_params[3])
        self.source_set_lookup = self.get_lut(punct_params[3])
        self.punct_algorithm = punct_params[1]
        self.update_frozen_flag = punct_params[4]
        self.recip_flag = np.array_equal(np.array(punct_params[2]), np.array(punct_params[3]))

    def __str__(self):
        """
        A string definition of PolarCode. This allows you to print any PolarCode object and see all of its
        relevant parameters.

        Returns
        ----------
        string
            a stringified version of PolarCode

        """

        output = '=' * 10 + " Polar Code " + '=' * 10 + '\n'
        output += "N: " + str(self.N) + '\n'
        output += "M: " + str(self.M) + '\n'
        output += "K: "+ str(self.K) + '\n'
        output += "Mothercode Construction: " + self.construction_type + '\n'
        output += "Ordered Bits (least reliable to most reliable): " + str(self.reliabilities) + '\n'
        output += "Frozen Bits: " + str(self.frozen) + '\n'
        output += "Puncturing Flag: " + str(self.punct_flag) + '\n'
        output += "Puncturing Parameters: {punct_type: " + str(self.punct_type) + '\n'
        output += "                        punct_algorithm: " + str(self.punct_algorithm) + '\n'
        output += "                        punct_set: " + str(self.punct_set) + '\n'
        output += "                        source_set: " + str(self.source_set) + '\n'
        output += "                        update_frozen_flag: " + str(self.update_frozen_flag) + "}" + '\n'
        return output

    def set_message(self, m):
        """
        Set the message vector to the non-frozen bits in ``x``. The frozen bits in ``frozen`` are set to zero.

        Parameters
        ----------
        m: ndarray<int>
            the message vector

        """

        self.message = m
        self.x[self.frozen_lookup == 1] = m
        self.u = self.x.copy()

    def get_codeword(self):
        """
        Get the codeword that was last encoded in this `PolarCode` object. Note that this codeword is not always
        the same as `myPC.u`, since punctured bits are simply set to zero in this variable as if they were
        frozen bits, and then decoded using the corresponding puncturing table likelihoods.

        Returns
        -------
        ndarray<float>
            the codeword for the last encoded message using `myPC.u`, or None.

        """
        if self.punct_flag == False:
            return self.u
        else:
            return self.u[np.where(self.source_set_lookup == 1)]

    def get_normalised_SNR(self, design_SNR):
        """
        Normalise E_b/N_o so that the message bits have the same energy for any code rate.

        Parameters
        ----------
        design_SNR: float
            E_b/N_o in decibels

        Returns
        ----------
        float
            normalised E_b/N_o in linear units

        """

        Eb_No_dB = design_SNR
        Eb_No = 10 ** (Eb_No_dB / 10)  # convert dB scale to linear
        Eb_No = Eb_No * (self.K / self.M)  # normalised message signal energy by R=K/M (M=N if not punctured)
        return Eb_No

    def get_lut(self, my_set):
        """
        Convert a set into a lookup table.

        Parameters
        ----------
        my_set: ndarray<int>
            a vector of indices

        Returns
        ----------
        ndarray<int>
            a LUT with "0" for an index in ``my_set``, else "1"

        """

        my_lut = np.ones(self.N, dtype=int)
        my_lut[my_set] = 0
        return my_lut

    def save_as_json(self, sim_filename):
        """
        Save all the important parameters in this object as a JSON file.

        Parameters
        ----------
        sim_filename: string
            directory and filename to save JSON file to (excluding extension)

        """
        data = {
            'N': self.M,
            'n': self.n,
            'K': self.K,
            'frozen': self.frozen.tolist(),
            'construction_type': self.construction_type,
            'punct_flag': self.punct_flag,
            'punct_type': self.punct_type,
            'punct_set': self.punct_set.tolist(),
            'source_set': self.source_set.tolist(),
            'punct_algorithm': self.punct_algorithm,
            'update_frozen_flag': self.update_frozen_flag,
            'BER': self.simulated_ber.tolist(),
            'FER': self.simulated_fer.tolist(),
            'SNR': self.simulated_snr.tolist()
        }
        with open(sim_filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def run_simulation(self, Eb_No, max_iter, min_errors, min_iters):
        frame_error_count = 0
        bit_error_count = 0
        num_blocks = 0
        for i in range(1, max_iter + 1):
            # simulate random PC in an AWGN channel
            self.set_message(np.random.randint(2, size=self.K))
            Encode(self)
            AWGN(self, Eb_No)
            Decode(self)

            # detect errors
            error_vec = self.message ^ self.message_received
            num_errors = sum(error_vec)
            frame_error_count = frame_error_count + (num_errors > 1)
            bit_error_count = bit_error_count + num_errors

            # early stopping condition
            num_blocks = i
            if frame_error_count >= min_errors and i >= min_iters:
                break
        return frame_error_count, bit_error_count, num_blocks

    def simulate(self, save_to, Eb_No_vec, design_SNR=None, max_iter=100000, min_iterations=1000, min_errors=30, sim_seed=1729, manual_const_flag=True):
        """
        Monte-carlo simulation of the performance of this polar code.
        The simulation has an early stopping condition of when the number of errors is below min_errors.
        Each E_b/N_o simulation has an additional early stopping condition using the minimum iterations
        and the minimum number of errors. The results are saved in a JSON file using :func:`save_as_json`.

        Parameters
        ----------
        save_to: string
            directory and filename to save JSON file to (excluding extension)
        Eb_No_vec: ndarray<float>
            the range of SNR values to simulate
        design_SNR: float
            the construction design SNR, E_b/N_o
        max_iter: int
            maximum number of iterations per SNR
        min_iterations: int
            the minimum number of iterations before early stopping is allowed per SNR
        min_errors: int
            the minimum number of frame errors before early stopping is allowed per SNR
        sim_seed: int
            pseudo-random generator seed, default is 1729 ('twister' on MATLAB)
        manual_const_flag: bool
            a flag that decides if construction should be done before simulating.
            Set to False if mothercode and/or puncturing constructions are manually set by the user.

        """

        # initialise simulation
        np.random.seed(sim_seed)
        frame_error_rates = np.zeros(len(Eb_No_vec))
        bit_error_rates = np.zeros(len(Eb_No_vec))

        # do construction if not done already
        if not manual_const_flag:
            if self.punct_flag and self.punct_type == 'shorten':
                Shorten(self, design_SNR)
            else:
                Construct(self, design_SNR)

        print(self)
        print('=' * 10, "Simulation", '=' * 10)
        for i in range(len(Eb_No_vec)):
            # run simulation for the current SNR
            frame_error_count, bit_error_count, num_blocks = self.run_simulation(Eb_No_vec[i], max_iter, min_errors, min_iterations)

            # calculate FER and BER
            frame_error_rate = frame_error_count / num_blocks
            bit_error_rate = bit_error_count / (self.K * num_blocks)
            frame_error_rates[i] = frame_error_rate
            bit_error_rates[i] = bit_error_rate
            print("Eb/No:", round(Eb_No_vec[i], 5), "  FER:", round(frame_error_rate, 3), "  BER:", round(bit_error_rate, 5))
            print('# Iterations:', num_blocks, '  # Frame Errors:', frame_error_count, ' # Bit Errors:', bit_error_count)
            print('='*20)

            # update GUI (if used)
            if self.status_bar != None:
                self.status_bar.set("Simulation progress: " + str(i + 1) + "/" + str(len(Eb_No_vec)))

            # early stopping condition
            if frame_error_count < min_errors:
                break

        # write data to JSON file
        self.simulated_snr = Eb_No_vec
        self.simulated_ber = bit_error_rates
        self.simulated_fer = frame_error_rates
        self.save_as_json(save_to)

        # update GUI construction fields (if used)
        if self.status_bar != None:
            self.gui_widgets[3].delete("1.0", tk.END)
            self.gui_widgets[6].delete("1.0", tk.END)
            self.gui_widgets[3].insert(tk.INSERT, ",".join(map(str, self.frozen)))
            self.gui_widgets[6].insert(tk.INSERT, ",".join(map(str, self.punct_set)))

        # update console and GUI
        print("Successfully completed simulation.\n")
        if self.status_bar != None:
            self.status_bar.set("Simulation progress: Done.")

    def plot_helper(self, new_plot, sim_filenames, dir, plot_title = 'Polar Code Performance'):
        # plot the FER and BER from file list
        new_plot.cla()
        for sim_filename in sim_filenames:
            with open(dir + sim_filename + '.json') as data_file:
                data_loaded = json.load(data_file)
            new_plot.plot(data_loaded['SNR'], data_loaded['FER'], '-o', markersize=6, linewidth=3, label=sim_filename)

        # format the plots
        new_plot.set_title(plot_title)
        new_plot.set_ylabel("Frame Error Rate")
        new_plot.set_xlabel("$E_b/N_o$ (dB)")
        new_plot.grid(linestyle='-')
        new_plot.set_yscale('log')
        new_plot.legend(loc='lower left')

    # call this for manual plotting
    def plot(self, sim_filenames, dir):
        """
        Plot multiple sets of FER data from the same directory on the same axes.

        Parameters
        ----------
        sim_filenames: ndarray<string>
            a list of all filenames to plot in a common root directory
        dir: string
            the root directory for the specified filenames

        """

        fig = plt.figure()
        new_plot = fig.add_subplot(111)
        self.plot_helper(new_plot, sim_filenames, dir)
        fig.show()

    # used by the GUI class for automated plotting
    def gui_plot_handler(self, gui_dict, fig):
        sim_filenames = gui_dict['filenames']
        dir = gui_dict['file_dir']
        self.plot_helper(fig, sim_filenames, dir)

    # used by the GUI class for simulating a new code
    def gui_sim_handler(self, gui_dict):
        # updated Polar Code from user
        punct_type = 'shorten' if gui_dict['punct_type'] == True else 'punct'
        shortening_params = (punct_type, gui_dict['punct_algo'], np.array(gui_dict['shortened_set'], dtype=int),
                             np.array(gui_dict['shortened_set'], dtype=int), False)
        self.initialise_code(gui_dict['N'], gui_dict['K'], shortening_params)
        self.construction_type = gui_dict['construction_algo']
        self.frozen = gui_dict['frozen_set']

        # simulation parameters from user
        iterations = gui_dict['iterations']
        min_frame_errors = gui_dict['min_frame_errors']
        file_dir = gui_dict['file_dir']
        save_to = gui_dict['save_to']
        manual_const_flag = gui_dict['manual_const_flag']
        design_SNR = gui_dict['design_SNR']
        Eb_No_vec = gui_dict['snr_values']

        # run simulation in another thread to avoid GUI freeze
        th = threading.Thread(name='sim_thread', target=self.simulate, args=(save_to, Eb_No_vec, design_SNR, iterations, 1000, min_frame_errors, 1729, manual_const_flag,))
        th.setDaemon(True)
        th.start()
