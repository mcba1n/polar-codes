from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from polarcodes.PolarCode import PolarCode

class GUI:
    def __init__(self):
        # initialise GUI global vars
        self.master = self.create_window()
        self.my_widget_ptrs = []
        self.fig = Figure(figsize=(10, 5), dpi=60)
        self.new_plot = self.fig.add_subplot(111)
        self.myPC = PolarCode(100, 64)

        # generate section frames
        first_frame = self.make_inputs()
        footer_frame = self.make_footer()
        self.myPC.gui_widgets = self.my_widget_ptrs

        # section layout
        first_frame.pack()
        self.make_plot(self.fig)
        footer_frame.pack(side=BOTTOM, fill=X)
        self.master.mainloop()

    def create_window(self):
        root = Tk()
        root.geometry("800x710")
        root.title('Polar Codes Simulator')
        return root

    def make_plot(self, fig):
        # add plot to tkinter canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas.draw()
        graph_widget = self.canvas.get_tk_widget()
        graph_widget.pack(pady=10)
        return graph_widget

    def make_inputs(self):
        my_frame = Frame(self.master)

        # labels
        titleFont = ("Arial", 10, "bold")
        pc_title_label = Label(my_frame, text="Polar Code", font=titleFont)
        puncturing_title_label = Label(my_frame, text="Puncturing/Shortening", font=titleFont)
        sim_label = Label(my_frame, text="Simulation", font=titleFont)
        plot_label = Label(my_frame, text="Plot", font=titleFont)
        n_label = Label(my_frame, text="N")
        k_label = Label(my_frame, text="K")
        const_title_label = Label(my_frame, text="Mothercode Construction")
        design_label = Label(my_frame, text="Design Eb/No (dB)")
        frozen_label = Label(my_frame, text="Frozen Bits")
        punct_algorithm_label = Label(my_frame, text="Algorithm")
        punctured_label = Label(my_frame, text="Punctured/Shortened Bits")
        iter = Label(my_frame, text="Iterations")
        min_frame_errors = Label(my_frame, text="Min. Frame Errors")
        save_dir_label = Label(my_frame, text="Directory")
        filenames_label = Label(my_frame, text="Filename(s)")
        save_label = Label(my_frame, text="Save to")
        Eb_No_values_label = Label(my_frame, text="Eb/No Values (dB)")

        # entries and text fields
        n_entry = Entry(my_frame)
        k_entry = Entry(my_frame)
        design_entry = Entry(my_frame)
        iter_entry = Entry(my_frame)
        min_frame_errors_entry = Entry(my_frame)
        file_dir_entry = Entry(my_frame)
        filenames_entry = Entry(my_frame)
        savefilename_entry = Entry(my_frame)
        frozen_bits = Text(my_frame, width=30, height=5)
        shortened_bits = Text(my_frame, width=30, height=5)
        Eb_No_values_entry = Entry(my_frame)

        n_entry.insert(END, '100')
        k_entry.insert(END, '64')
        design_entry.insert(END, '5')
        iter_entry.insert(END, '10000')
        min_frame_errors_entry.insert(END, '30')
        savefilename_entry.insert(END, 'data/pc_sim')
        file_dir_entry.insert(END, 'data/')
        filenames_entry.insert(END, 'pc_sim')
        Eb_No_values_entry.insert(END, '1,2,3,4')

        # menus
        const_options = ['bb', 'ga']
        const_opt_val = StringVar(my_frame)
        const_opt_val.set(const_options[0])
        const_opt = OptionMenu(my_frame, const_opt_val, *const_options)

        shortening_options = ['brs', 'wls', 'bgl']
        shortening_opt_val = StringVar(my_frame)
        shortening_opt_val.set(shortening_options[0])
        shortening_opt = OptionMenu(my_frame, shortening_opt_val, *shortening_options)

        # check and radio buttons
        manual_construction_flag = IntVar()
        punct_type_var = IntVar()
        punct_type_var.set(1)

        c = Checkbutton(my_frame, text="Manual Construction", variable=manual_construction_flag)
        p_opt = Radiobutton(my_frame, text="Puncturing", value=0, variable=punct_type_var)
        s_opt = Radiobutton(my_frame, text="Shortening", value=1, variable=punct_type_var)

        # buttons
        button_1 = Button(my_frame, text="Simulate",
                          command=lambda: [self.update_dict(), self.myPC.gui_sim_handler(self.dict)])
        button_2 = Button(my_frame, text="Plot",
                          command=lambda: [self.update_dict(), self.myPC.gui_plot_handler(self.dict, self.new_plot),
                                           self.canvas.draw()])

        # grid layout
        pc_title_label.grid(row=0, columnspan=2)
        n_label.grid(row=1, sticky=E)
        k_label.grid(row=2, sticky=E)
        n_entry.grid(row=1, column=1, sticky="nsew")
        k_entry.grid(row=2, column=1, sticky="nsew")
        design_label.grid(row=5, column=0, sticky=E)
        design_entry.grid(row=5, column=1, sticky="nsew")
        frozen_bits.grid(row=4, column=1)
        frozen_label.grid(row=4, column=0, sticky=E)
        const_title_label.grid(row=3, column=0, sticky=E)
        const_opt.grid(row=3, column=1, sticky="nsew")
        c.grid(row=6, column = 1)
        puncturing_title_label.grid(row=0, column=2, columnspan=2)
        shortening_opt.grid(row=2, column=3, sticky="nsew")
        punct_algorithm_label.grid(row=2, column=2, sticky=E)
        p_opt.grid(row=1, column=2)
        s_opt.grid(row=1, column=3)
        shortened_bits.grid(row=4, column=3)
        punctured_label.grid(row=4, column=2, sticky=E)
        save_label.grid(row=11, column=0, sticky=E)
        savefilename_entry.grid(row=11, column=1, sticky="nsew")
        save_dir_label.grid(row=9, column=2, sticky=E)
        file_dir_entry.grid(row=9, column=3, sticky="nsew")
        sim_label.grid(row=8, columnspan=2, column=0)
        plot_label.grid(row=8, columnspan=2, column=2)
        iter_entry.grid(row=9, column=1, sticky="nsew")
        min_frame_errors_entry.grid(row=10, column=1, sticky="nsew")
        iter.grid(row=9, column=0, sticky=E)
        min_frame_errors.grid(row=10, column=0, sticky=E)
        filenames_label.grid(row=10, column=2, sticky=E)
        filenames_entry.grid(row=10, column=3, sticky="nsew")
        Eb_No_values_label.grid(row=12, column=0, sticky=E)
        Eb_No_values_entry.grid(row=12, column=1, sticky="nsew")
        button_1.grid(row=13, column=0, columnspan=2, sticky="nsew")
        button_2.grid(row=13, column=2, columnspan=4, sticky="nsew")

        # save data widget pointers
        my_widget_ptrs = [n_entry, k_entry, const_opt_val, frozen_bits, punct_type_var, shortening_opt_val, shortened_bits, iter_entry,
                          min_frame_errors_entry, savefilename_entry, file_dir_entry, filenames_entry, design_entry, manual_construction_flag,
                          Eb_No_values_entry]
        self.my_widget_ptrs = my_widget_ptrs
        return my_frame

    def make_footer(self):
        my_frame = Frame(self.master)
        my_text_var = StringVar(my_frame)
        my_text_var.set("Simulation updates here...")
        status = Label(my_frame, relief=SUNKEN, anchor=W, textvariable=my_text_var)
        status.pack(fill=X)
        self.myPC.status_bar = my_text_var
        return my_frame

    def update_dict(self):
        # create GUI parameters object
        self.dict = {
            'N': int(self.my_widget_ptrs[0].get()),
            'K': int(self.my_widget_ptrs[1].get()),
            'construction_algo': self.my_widget_ptrs[2].get(),
            'frozen_set': [int(x) for x in self.my_widget_ptrs[3].get("1.0", END).strip().split(",") if x != ''],
            'punct_type': int(self.my_widget_ptrs[4].get()),
            'punct_algo': self.my_widget_ptrs[5].get(),
            'shortened_set': [int(x) for x in self.my_widget_ptrs[6].get("1.0", END).strip().split(",") if x != ''],
            'iterations': int(self.my_widget_ptrs[7].get()),
            'min_frame_errors': int(self.my_widget_ptrs[8].get()),
            'save_to': self.my_widget_ptrs[9].get(),
            'file_dir': self.my_widget_ptrs[10].get(),
            'filenames': self.my_widget_ptrs[11].get().strip().replace(' ', '').split(","),
            'design_SNR': int(self.my_widget_ptrs[12].get()),
            'manual_const_flag': self.my_widget_ptrs[13].get(),
            'snr_values': np.array([int(x) for x in self.my_widget_ptrs[14].get().strip().split(',')])
        }
