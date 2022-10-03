import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sovaflow.flow import createRaw, organize_channels, set_montage
from sovaharmony.p_entropy import p_entropy
from sovaharmony.pme import Amplitude_Modulation_Analysis

fnameCE=r"E:\Academico\Universidad\Posgrado\Tesis\Datos\BASESDEDATOS\BIOMARCADORES_BIDS\derivatives\sovaharmony\sub-CTR001\ses-V0\eeg\sub-CTR001_ses-V0_task-CE_desc-norm_eeg"
raw_data=mne.read_epochs(fnameCE + '.fif', verbose='error')
data = raw_data.get_data()
(e, c, t) = data.shape
new_data = np.transpose(data.copy(),(1,2,0))
for e in range(data.shape[0]):
    for c in range(data.shape[1]):
        assert np.all(data[e,c,:] == new_data[c,:,e])
        pme = Amplitude_Modulation_Analysis(new_data,raw_data.info['sfreq'])
        print(pme)

'''mean_channels = []
for channel in range(data.shape[1]):
    segment = []
    for epoch in range(data.shape[0]):
        # Por segmento
        entropy_segment = p_entropy(new_data[channel,:,epoch])
        segment.append(entropy_segment)
        # Por canal
    mean_channels.append(np.mean(segment))
print(len(segment))
print(len(mean_channels))'''
