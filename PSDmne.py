import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sovaflow.flow import createRaw, organize_channels, set_montage

#Importación 
fnameE=r"E:\Academico\Universidad\Posgrado\Tesis\Datos\BASESDEDATOS\BIOMARCADORES_BIDS\derivatives\sovaharmony\sub-CTR001\ses-V0\eeg\sub-CTR001_ses-V0_task-OE_desc-norm_eeg"
raw=mne.read_epochs(fnameE + '.fif', verbose='error')
#correct_montage = ['OZ','O1','O2']
#raw,correct_montage= organize_channels(raw,correct_montage)
#raw,montage = set_montage(raw,'standard_1005')
#if correct_montage is not None:
#    assert correct_montage == set(raw.info['ch_names'])

raw_welch=mne.time_frequency.psd_array_welch(raw._data, sfreq=256) #Revisar fs
#raw_psd=mne.time_frequency.psd_array_multitaper(raw._data, sfreq=1000)

Pxx=raw_welch[0]
f=raw_welch[1]
#Pxx=raw_psd[0]
#f=raw_psd[1]
for i in range(len(Pxx)):
  plt.plot(f,Pxx[i][0])
  position = np.where((f>= 8) & (f<= 13))
  maxValue = np.max(Pxx[i][position[0]])
  position_end= np.where(Pxx[i]==maxValue)
  print('P:',position_end[0][0],'Max',maxValue)

plt.xlim(4,15)
plt.ylabel('Amplitud ')
plt.xlabel('Frecuencia')
plt.title('Señal')
plt.grid(True)
plt.show()
