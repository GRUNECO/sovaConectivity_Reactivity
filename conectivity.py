from sovaflow.utils import cfg_logger,get_spatial_filter
import mne
import os
from bids import BIDSLayout
from bids.layout import parse_file_entities
from datetime import datetime
import numpy as np
from sovaharmony.info import info as info_dict
import pandas as pd
from time import time
from sl import get_sl
import matplotlib.pyplot as plt
import numpy as np
from bids.layout import parse_file_entities
from sovaharmony.getDataframes import get_information_data
from sovaharmony.processing import get_derivative_path

#BIO CTR, DTA, DCL: eeg_file[82:88]+'_'+eeg_file[115:117]+'_'+eeg_file[123:125]
#BIO G1, G2: eeg_file[82:87]+'_'+eeg_file[113:115]+'_'+eeg_file[121:123]
#CHBMP eeg_file[82:94]
#SRM eeg_file[82:89]+eeg_file[93:96]


def load(path):
    raw_data = mne.read_epochs(path, verbose='error')
    data = raw_data.get_data()
    new_data = np.transpose(data.copy(),(1,2,0))
    for e in range(data.shape[0]):
        for c in range(data.shape[1]):
            assert np.all(data[e,c,:] == new_data[c,:,e])
    return new_data, raw_data.info['sfreq']

def sl_connectivity(THE_DATASET,fast_mode=False):
    layout,task,runlabel,name,group_regex,session_set=get_information_data(THE_DATASET)
    sl_mean = []
    sl_subject = []
    sl_columns = []
    # Dataset dependent inputs
    input_path = THE_DATASET.get('input_path',None)
    default_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    channels = THE_DATASET.get('channels',default_channels)
    layout_dict = THE_DATASET.get('layout',None)

    # Static Params
    pipeline = 'sovaharmony'
    pipelabel = '['+THE_DATASET.get('run-label', '')+']'
    layout = BIDSLayout(input_path)
    bids_root = layout.root
    output_path = os.path.join(bids_root,'derivatives',pipeline)

    eegs = layout.get(**layout_dict)

    derivatives_root = os.path.join(layout.root,'derivatives',pipeline)
    log_path = os.path.join(derivatives_root,'code')
    os.makedirs(log_path, exist_ok=True)
    logger,currentdt = cfg_logger(log_path)
    e = 0
    archivosconerror = []
    description = layout.get_dataset_description()
    desc_pipeline = "sovaharmony, a harmonization eeg pipeline using the bids standard"
    description['GeneratedBy']=[info_dict]
    num_files = len(eegs)
    list_studies=[name]*len(eegs)
    list_info=[parse_file_entities(eegs[i]) for i in range(len(eegs))]
    list_subjects=[info['subject'] for info in list_info]
    for i,eeg_file in enumerate(eegs):
        #process=str(i)+'/'+str(num_files)
        try:
            logger.info(f"File {i+1} of {num_files} ({(i+1)*100/num_files}%) : {eeg_file}")
            norm_path = get_derivative_path(layout,eeg_file,'norm','eeg','.fif',bids_root,derivatives_root)
            #capture the start time
            star_time = time()
            #Read input data
            data, fs = load(norm_path)
            #data, fs = load_epoch(path)
            #For default values
            sl = get_sl(data, fs)
            for mean in range(len(sl[0])):
                mean_ch = np.mean(sl[mean])
                sl_mean.append(mean_ch)
            sl_subject.append(sl_mean)
            if i <= 203:
                sl_columns.append(eeg_file[82:89]+eeg_file[93:96])
                if not os.path.exists(r'sovaConectivity_Reactivity\Conectivity\SL_'+ eeg_file[82:89]+eeg_file[93:96] +'.png'):
                    plt.pcolor(sl,cmap=plt.cm.Blues)
                    plt.colorbar()
                    top=0.88
                    bottom=0.11
                    left=0.125
                    right=0.9
                    hspace=0.2
                    wspace=0.2
                    plt.title('Conectivity SL ' + eeg_file[82:89]+eeg_file[93:96])
                    plt.xticks(np.arange(0, 58, 1),channels,rotation=75,fontsize=4)
                    plt.yticks(np.arange(0, 58, 1),channels,fontsize=4)
                    plt.ylabel('Channels')
                    plt.xlabel('Channels')
                    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom,hspace=hspace,wspace=wspace)
                    plt.tight_layout()
                    plt.savefig(r'sovaConectivity_Reactivity\Conectivity\SL_{name_group}.png'.format(name_group= eeg_file[82:89]+eeg_file[93:96]),dpi=500)
                    #plt.show()

                    plt.close()
                    #show the execution time
                    print("The execution time [seconds]:")
                    print(time()-star_time)
                else:
                    print("ya existe "+eeg_file[82:89]+eeg_file[93:96])
                    pass
            else:
                sl_columns.append(eeg_file[82:89]+eeg_file[93:96])
                if not os.path.exists(r'sovaConectivity_Reactivity\Conectivity\SL_'+ eeg_file[82:89]+eeg_file[93:96] +'.png'):
                    plt.pcolor(sl,cmap=plt.cm.Blues)
                    plt.colorbar()
                    top=0.88
                    bottom=0.11
                    left=0.125
                    right=0.9
                    hspace=0.2
                    wspace=0.2
                    plt.title('Conectivity SL ' + eeg_file[82:89]+eeg_file[93:96])
                    plt.xticks(np.arange(0, 58, 1),channels,rotation=75,fontsize=4)
                    plt.yticks(np.arange(0, 58, 1),channels,fontsize=4)
                    plt.ylabel('Channels')
                    plt.xlabel('Channels')
                    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom,hspace=hspace,wspace=wspace)
                    plt.tight_layout()
                    plt.savefig(r'sovaConectivity_Reactivity\Conectivity\SL_{name_group}.png'.format(name_group= eeg_file[82:89]+eeg_file[93:96]),dpi=500)
                    #plt.show()

                    plt.close()
                    #show the execution time
                    print("The execution time [seconds]:")
                    print(time()-star_time)
                else:
                    print("ya existe "+eeg_file[82:89]+eeg_file[93:96])
                    pass
            sl_mean = []
        except Exception as error:
            e+=1
            logger.exception(f'Error for {eeg_file}')
            archivosconerror.append(eeg_file)
            print(error)
            pass
    
    return sl_subject,sl_columns   
