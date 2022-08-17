import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sovaflow.flow import createRaw, organize_channels, set_montage

#Importación 
#fnameE=r"E:\Academico\Universidad\Posgrado\Tesis\Datos\BASESDEDATOS\BIOMARCADORES_BIDS\derivatives\sovaharmony\sub-CTR001\ses-V0\eeg\sub-CTR001_ses-V0_task-OE_desc-norm_eeg"
fnameOE=r"D:\TDG\filesSaved\BIOMARCADORES\derivatives\sovaharmony\sub-CTR001\ses-V0\eeg\sub-CTR001_ses-V0_task-OE_desc-norm_eeg"
fnameCE=r"D:\TDG\filesSaved\BIOMARCADORES\derivatives\sovaharmony\sub-CTR001\ses-V0\eeg\sub-CTR001_ses-V0_task-CE_desc-norm_eeg"

raw_OE=mne.read_epochs(fnameCE + '.fif', verbose='error')
raw_CE=mne.read_epochs(fnameOE + '.fif', verbose='error')

roi_LH=['T3','T5','F3','C3','P3','O1'] #Left hemisphere 
roi_RH=['T4','T6','F4','C4','P4','O2'] #Right hemisphere
roi_O=['O1','O2']# Occipital  
roi_F=['F3','F4','F0']# Frontal 
roi_reactividad=['O1','O2','OZ']

def organize_rois(raw,correct_montage):   
  raw,correct_montage= organize_channels(raw,correct_montage)
  raw,montage = set_montage(raw,'standard_1005')
  if correct_montage is not None:
    assert correct_montage == set(raw.info['ch_names'])
    return raw

raw_organize_OE=organize_rois(raw_OE,roi_reactividad)
raw_organize_CE=organize_rois(raw_CE,roi_reactividad)

raw_welch_OE=mne.time_frequency.psd_array_welch(raw_organize_OE._data, sfreq=256) #Revisar fs
raw_welch_CE=mne.time_frequency.psd_array_welch(raw_organize_CE._data, sfreq=256) #Revisar fs

def average_psd_v1(raw_welch):
  Pxx=raw_welch[0]
  f=raw_welch[1]
  max= []
  for epochs in range(len(Pxx)):
    for ch in range(len(Pxx[1])):
      position = np.where((f>= 8) & (f<= 13))
      maxValue = np.max(Pxx[epochs][ch][position[0]])
      position_end= np.where(Pxx[epochs]==maxValue)
      max.append(maxValue)
  average=np.average(max)
  return Pxx, f, average

Pxx_CE, f_CE,average_CE=average_psd_v1(raw_welch_CE)
Pxx_OE, f_OE,average_OE=average_psd_v1(raw_welch_OE)
reactivity_v1= (average_CE-average_OE)/average_CE

def average_psd_v2(raw_welch):
  Pxx=raw_welch[0]
  f=raw_welch[1]
  psd= []
  for epochs in range(len(Pxx)):
    position = np.where((f>= 8) & (f<= 13))
    psd.append(Pxx[epochs][0,:][position[0]]) 
  average=np.average(psd)
  maxValue = np.max(average)
  return maxValue

maxValue_CE=average_psd_v2(raw_welch_CE)
maxValue_OE=average_psd_v2(raw_welch_OE)
reactivity_v2= (maxValue_CE-maxValue_OE)/maxValue_CE

print(reactivity_v1,reactivity_v2)

plt.plot(f_CE,Pxx_CE[0][0,:],linestyle='--',label='CE')
plt.plot(f_OE,Pxx_OE[0][0,:],linestyle='solid',label='OE')


plt.xlim(0,20)
plt.ylabel('Amplitud ')
plt.xlabel('Frecuencia')
plt.title('Señal')
plt.grid(True)
plt.show()
plt.legend()



#raw_psd=mne.time_frequency.psd_array_multitaper(raw._data, sfreq=1000)
#Pxx=raw_psd[0]
#f=raw_psd[1]