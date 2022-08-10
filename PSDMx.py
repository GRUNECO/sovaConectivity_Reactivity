import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sovaflow.flow import createRaw

#Importación 
fnameE=r"E:\Academico\Universidad\Posgrado\Tesis\Datos\BASESDEDATOS\BIOMARCADORES_BIDS\derivatives\sovaharmony\sub-CTR001\ses-V0\eeg\sub-CTR001_ses-V0_task-OE_desc-norm_eeg"
raw=mne.read_epochs(fnameE + '.fif', verbose='error')
data=raw.copy()
(e, c, t) = raw._data.shape
da_eeg_cont = np.reshape(data,(c,e*t),order='F')
default_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
signal_ch = createRaw(da_eeg_cont,data.info['sfreq'],ch_names=data.info['ch_names'])
data_signal = signal_ch.copy()


def psdMax (data):   
    #periodograma de Welch
    nblock = data.info['sfreq']
    overlap = data.info['sfreq']/2
    win = signal.hamming(int(nblock),True)
        
    f, Pxx = signal.welch(data[i,:], data.info['sfreq'], window=win, noverlap=overlap, nfft=nblock, return_onesided=True)
    plt.plot(f[f<20],Pxx[f<20])
    plt.ylabel('Amplitud ')
    plt.xlabel('Frecuencia')
    plt.title('Senal')
    plt.grid(True)
    plt.show()
        
    position = np.where((f >= 8) & (f <= 13)) 
    maxValue = np.max(Pxx[position[0]])
    
    return maxValue

for i in range(len(data.info['ch_names'])):
    ssvepValue = psdMax(signal_ch)

