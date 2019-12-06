import numpy as np
from polarcodes.Math import Math
from polarcodes.Construct import Construct

class Puncture(Construct):
    def __init__(self, myPC, design_SNR, manual=False):
        """
        In the future, this class will contain common puncturing algorithms.

        :param myPC:
        :param design_SNR:
        :param manual:
        """

        super().__init__(myPC, design_SNR, True)
        if manual:
            return