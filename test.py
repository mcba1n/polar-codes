from polarcodes.PolarCode import PolarCode
import numpy as np
import json

# simulate test code
myPC = PolarCode(64, 32)
myPC.simulate(save_to='data/pc_sim', Eb_No_vec=np.arange(1,5), design_SNR=5.0, manual_const_flag=False)
myPC.plot(['pc_sim'], 'data/')

# read FER and BER data from test code
with open('data/pc_sim.json') as json_file:
    data = json.load(json_file)
    FER_data = np.array(data['FER'])
    BER_data = np.array(data['BER'])

# compare test code with known correct values
FER_test_data = np.array([
        0.313,
        0.126,
        0.03,
        0.004125412541254125
    ])
BER_test_data = np.array([
        0.09709375,
        0.03740625,
        0.00815625,
        0.0010184612211221122
    ])

check_cond = np.logical_and(np.abs(FER_test_data-FER_data) < 1e-3, np.abs(BER_test_data-BER_data) < 1e-3)
if np.sum(check_cond) == check_cond.shape[0]:
    print("The library is ready to go!")