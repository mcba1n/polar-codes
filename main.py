from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from polarcodes.GUI import GUI
from polarcodes.PolarCode import PolarCode

root = Tk()
root.geometry("800x710")
root.title('Polar Codes Simulator')
myPC = PolarCode(100, 64)
myGUI = GUI(root, myPC)
root.mainloop()