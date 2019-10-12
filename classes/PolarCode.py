#!/usr/bin/env python

"""
An object that encapsulates all of the parameters required to define a polar code.
This object must be given to the following classes: AWGN, Construct, Decode, Encode, GUI, Shorten.
"""

import numpy as np
from classes.Math import Math
from classes.Construct import Construct
from classes.Encode import Encode
from classes.Decode import Decode
from classes.AWGN import AWGN
import json
import matplotlib.pyplot as plt
import threading
import tkinter as tk

class PolarCode(Math):
    def __init__(self, M, K, punct_type = 'None'):
        """
        Constructor arguments:
            M -- the block length.
            K -- the code dimension.
            punct_type -- 'punct' for puncturing, 'shorten' for shortening
        """

        self.initialise_code(M, K, punct_type)
        self.status_bar = None  # set by the GUI so that the simulation progress can be tracked
        self.gui_widgets = []

    def initialise_code(self, M, K, punct_type = 'None'):
        """
        Description:
            Initialise the code with a set of parameters the same way as the constructor.
            Call this any time you want to change the code rate.
        """

        # mothercode parameters
        self.M = M                              # the block length.
        self.N = int(2**(np.ceil(np.log2(M))))  # the mothercode block length (N=M if not punctured).
        self.n = int(np.log2(self.N))
        self.K = K                              # the code dimension.
        self.frozen = []                        # the frozen indices.
        self.frozen_lookup = []                 # LUT: "0" => frozen, "1" => information.
        self.x = np.zeros(self.N, dtype=int)    # the uncoded message with parity bits
        self.construction_type = 'bb'           # the mothercode construction type.
        self.construction_algos = {"Bhattacharyya Bounds": 'bb', "Gaussian Approximation": 'ga'}
        self.message_received = []              # the decoded message from a channel.
        
        # puncturing parameters
        self.punct_type = punct_type            # 'punct' for puncturing, 'shorten' for shortening
        self.punct_flag = False if self.M == self.N else True   # is True if punctured
        self.punct_set = []                     # the coded puncturing indices.
        self.punct_set_lookup = []              # LUT: "0" => punctured, "1" => information.
        self.s = self.N - self.M                # size of punct_set
        self.punct_algorithm = 'None'
        self.shortening_algos = {"None": '', "BRS": 'brs', "WLS": 'wls', "Permutation": 'perm'}
        self.recip_flag = False                 # is True if coded punctured bits equal uncoded punctured bits

    def __str__(self):
        """
        Description:
            A string definition of PolarCode.
            This allows you to print any PolarCode object and see all of its relevant parameters.
        Returns:
            A stringified version of PolarCode.
        """

        output = ""
        output += "N: " + str(self.N) + '\n'
        output += "M: " + str(self.M) + '\n'
        output += "K: "+ str(self.K) + '\n'
        output += "Mothercode Construction: " + self.construction_type + '\n'
        output += "Frozen Bits: " + str(self.frozen) + '\n'
        output += "Puncturing Type: " + self.punct_type + '\n'
        output += "Puncturing Flag: " + str(self.punct_flag) + '\n'
        output += "Puncturing Algorithm: " + self.punct_algorithm + '\n'
        output += "Punctured Bits: " + str(self.punct_set)
        return output

    def set_message(self, m):
        """
        Description:
            Set the message vector to the non-frozen bits in self.x.
            The frozen bits are set to zero.
        """

        self.message = m
        self.x[self.frozen_lookup == 1] = m
        self.u = self.x.copy()

    def get_normalised_SNR(self, design_SNR):
        """
        Description:
            Normalise E_b/N_o so that the message bits have the same energy for any code rate.
        Arguments:
            design_SNR -- E_b/N_o in decibels.
        Returns:
            Normalised E_b/N_o in decibels.
        """

        Eb_No_dB = design_SNR
        Eb_No = 10 ** (Eb_No_dB / 10)  # convert dB scale to linear
        Eb_No = Eb_No * (self.K / self.M)  # normalised message signal energy by R=K/M (M=N if not punctured)
        return Eb_No

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

    def simulate(self, sim_filename, Eb_No_vec, design_SNR = 5.0, max_iter = 100000, manual_const_flag = False,
                 min_iterations=1000, min_errors=30, sim_seed = 1729):
        """
        Description:
            Monte-carlo simulation of the performance of PolarCode.
            The simulation has an early stopping condition of when the number of errors is below min_errors.
            Each E_b/N_o simulation has an additional early stopping condition of one thousand minimum iterations
            and also satisfying the minimum number of errors. The results are saved in a JSON file.
        Arguments:
            sim_filename -- the name of the file to save the simulations results to. Format: JSON.
            design_SNR -- the construction design E_b/N_o.
            max_iter -- max. number of iterations.
            min_errors -- the min. number of frame errors before early stopped is allowed.
            min_iterations -- the min. number of iterations before early stopping is allowed.
            manual_const_flag -- a flag that decides if Construct should be used.
                                Set to False if self.frozen (and self.frozen_lookup),
                                 and self.punct_set (and self.punct_set_lookup) are manually set by the user.
            Eb_No_vec -- a np.array() of the Eb_No values to simulate.
        """

        # initialise simulation
        np.random.seed(sim_seed)
        frame_error_rates = np.zeros(len(Eb_No_vec))
        bit_error_rates = np.zeros(len(Eb_No_vec))

        # construction using design_SNR
        Construct(self, design_SNR, manual_const_flag)
        print('=' * 10, "Polar Code", '=' * 10)
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
        data = {'N': self.M,
                'K': self.K,
                'SNR': Eb_No_vec.tolist(),
                'BER': bit_error_rates.tolist(),
                'FER': frame_error_rates.tolist()
                }
        with open(sim_filename + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # update GUI construction fields (if used)
        if self.status_bar != None:
            self.gui_widgets[3].delete("1.0", tk.END)
            self.gui_widgets[6].delete("1.0", tk.END)
            self.gui_widgets[3].insert(tk.INSERT, ",".join(map(str, self.frozen)))
            self.gui_widgets[6].insert(tk.INSERT, ",".join(map(str, self.punct_set)))

        # update console and GUI
        print("Successfully completed simulation.")
        if self.status_bar != None:
            self.status_bar.set("Simulation progress: Done.")

    def plot_helper(self, new_plot, sim_filenames, dir, plot_title = 'PC Performance'):
        # plot the FER and BER from file list
        new_plot.cla()
        for sim_filename in sim_filenames:
            with open(dir + sim_filename + '.json') as data_file:
                data_loaded = json.load(data_file)
            new_plot.plot(data_loaded['SNR'], data_loaded['FER'], '-o', markersize=3, linewidth=1, label=sim_filename)

        # format the plots
        new_plot.set_title(plot_title)
        new_plot.set_ylabel("Frame Error Rate")
        new_plot.set_xlabel("Design SNR, $E_b/N_o$ (dB)")
        new_plot.grid(linestyle='-')
        new_plot.set_yscale('log')
        new_plot.legend(loc='lower left')

    # call this for manual plotting
    def plot(self, sim_filenames, dir):
        """
        Description:
            Plot multiple sets of FER data from the same directory on the same axes.
        Arguments:
            sim_filenames -- a list of all filenames to plot in a common root directory.
            dir -- the root directory for the specified filenames.
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
        self.initialise_code(gui_dict['N'], gui_dict['K'], punct_type)
        self.construction_type = self.construction_algos[gui_dict['construction_algo']]
        self.punct_algorithm = self.shortening_algos[gui_dict['punct_algo']]
        self.frozen = gui_dict['frozen_set']
        self.punct_set = gui_dict['shortened_set']

        # simulation parameters from user
        iterations = gui_dict['iterations']
        min_frame_errors = gui_dict['min_frame_errors']
        file_dir = gui_dict['file_dir']
        save_to = gui_dict['save_to']
        manual_const_flag = gui_dict['manual_const_flag']
        design_SNR = gui_dict['design_SNR']
        Eb_No_vec = gui_dict['snr_values']

        # run simulation in another thread to avoid GUI freeze
        x = threading.Thread(name='sim_thread', target=self.simulate, args=(save_to, Eb_No_vec, design_SNR, iterations, manual_const_flag))
        x.setDaemon(True)
        x.start()

