from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from classes.GUI import GUI
from classes.PolarCode import PolarCode

root = Tk()
root.geometry("800x710")
root.title('Polar Codes Simulator')
myPC = PolarCode(100, 64)
myGUI = GUI(root, myPC)
root.mainloop()