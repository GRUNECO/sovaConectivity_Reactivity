#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marcos L. David Usuga
"""
####################### External libraries ####################################
import os
import scipy.io as sio
from glob import glob
import pandas as pd
import numpy as np
import itertools
from sklearn import svm
from sklearn.model_selection import LeaveOneOut

####################### Internal libraries ####################################
import amplitude_modulation
#import scalograms

############################## Functions ######################################
def epochs_information(path, path_save, signal_key, id_group="g"):
    """
    Extrae informacion de todas las señales en un directorio (numero de canales, 
    epocas y muestras) y deja un registro en un archivo de texto. Retorna el 
    menor y maximo numero de epocas encontrado en uno de los registro del  
    directorio, y el numero de registros que contiene el directorio.  
    """
    # Obtener los archivos en la direccion
    files = glob(path)
    if files:
        txt_file = open(path_save+id_group+'_file_information.txt','w')
        num_epochs = [] # Lista que va a contener el numero de epocas de cada una de las señales
        txt_file.write('\n*****************************************************************\n')
        txt_file.write('\nFiles on '+path+' :\n')
        num_files = 0 # Contador de registros en la direccion
        # Recorrer cada uno de los archivos en la direccion
        for file in files:
            # Trabajar solo los archivos .mat
            if os.path.splitext(file)[1] == '.mat':
                # Obtener el nombre del registro
                file_name = os.path.splitext(os.path.basename(file))[0]
                txt_file.write('\n'+file_name+'\n')
                # Importar la señal
                signal = sio.loadmat(file)[signal_key]
                if signal.ndim==1:
                    channels = 1
                    values = signal.size
                    epochs = 0
                if signal.ndim==2:
                    channels, values = signal.shape
                    epochs = 0
                if signal.ndim==3:
                    channels, values, epochs = signal.shape
                txt_file.write('\nChannels: '+str(channels)+
                              '\nValues: '+str(values)+
                              '\nEpochs: '+str(epochs))
                # Guardar el numero de epocas de la señal
                num_epochs.append(epochs)
                del signal, channels, values, epochs
                # Incrementar el contador de registros
                num_files += 1
                txt_file.write('\n\n')
        txt_file.write('\nThe number of files on the path are '+str(num_files)+'\n')
        # Obtener el menor numero de epocas de todas las señales
        num_epochs = np.array(num_epochs)
        max_epochs = np.max(num_epochs)
        min_epochs = np.min(num_epochs)
        txt_file.write('\nThe smallest number of epochs is: '+str(min_epochs)+
                      '\nThe greatest number of times is: '+str(max_epochs))
        txt_file.close()
        return min_epochs, max_epochs, num_files
    else:
        return False

def min_epochs_number(files_paths, path_save, signal_key, id_group=''):
    """
    Encuentra el menor numero de epocas que posee uno de los registros entre
    varios directorios. Retorna el el menor numero de epocas entre todos los 
    directorios y una lista con el menor numero de epocas de cada directorio.
    """
    num_epochs = []
    for file in range(0,len(files_paths)):
        path = files_paths[file]
        num_epochs.append(epochs_information(path, path_save, signal_key, id_group=id_group+"g"+str(file+1))[0])
    num_epochs = np.array(num_epochs)
    return num_epochs, np.min(num_epochs)

def calculate_pme(path, path_save, name_montages, sampling_frequency, signal_key, 
                  bands, name_bands, new_sampling_frequency, num_epochs=100, 
                  id_group="g", method='hamming'):
    """
    Calcula el PME a cada uno de los registros en un directorio, guarda cada matriz
    de PME en un archivo csv.
    """
    #################### Calcular y guardar todos los PME #####################
    sampling_frequency = float(sampling_frequency)
    # Obtener los archivos en la direccion
    files = glob(path)  
    if files:
        txt_file = open(path_save+id_group+'_process_information.txt','w')
        txt_file.write('\n*****************************************************************\n')
        txt_file.write('\nFiles on '+path+' :\n')
        num_files = 0 # Contador de registros en la direccion
        # Recorrer cada uno de los archivos en la direccion
        for file in files:
            # Trabajar solo los archivos .mat
            if os.path.splitext(file)[1]  == '.mat':
                # Obtener el nombre del registro
                file_name = os.path.splitext(os.path.basename(file))[0]
                txt_file.write('\n'+file_name+'\n')
                # Importar la señal
                signal = np.squeeze(sio.loadmat(file)[signal_key])
                # Remuestrear la señal
                signal = scalograms.resampling(signal, sampling_frequency, new_sampling_frequency)
                signal = signal[:,:,0:num_epochs]
                txt_file.write('\nMontages: '+str(signal.shape[0])+
                              '\nValues: '+str(signal.shape[1])+
                              '\nEpochs: '+str(signal.shape[2])+
                              '\nSampling Frequency: '+str(new_sampling_frequency))
                # Calcular el PME de la señal, por metodo de la ventana Hamming
                pme = amplitude_modulation.Algorithm_Amplitude_Modulation_Analysis(signal,
                                                                  new_sampling_frequency, Bands=bands, Method=method)
                # Guardar el PME 
                amplitude_modulation.Save_PME(pme, 
                             Path_Name_File=path_save+file_name+'_'+method, Name_Bands=name_bands,
                             Name_Channels=name_montages)
                del pme, signal
                # Incrementar el contador de registros
                num_files += 1
                txt_file.write('\n\n')
                print("\nFinish with "+file_name)
        txt_file.write('\n\n')
        txt_file.write('\nThe number of files on the path are '+str(num_files)+'\n')
        txt_file.close()    
                   
def group_pme(path, path_save, name_montages, name_bands, id_group="G", id_file='v'):
    """
    Toma todos los archivos de csv con las matrices de PME de varios directorios, 
    las organiza en un dataset con todos los resultados y los guarda en un rchivo
    MAT donde cada directorio esta separado por un identificativo. 
    """
    ######################### Agrupar PME de cada grupo #######################
    # Organizar los PME en un mismo archivo y guardarlo 
    adict = {}
    # Abrir PME de cada grupo y los respectivos metodos de calculo
    for file, path in enumerate(path):
        adict['PME_'+id_group+str(file+1)] = amplitude_modulation.Load_Table_PME(path, name_montages, Name_Bands=name_bands)[0]
    sio.savemat(path_save+'PME_'+id_file+'.mat', adict)

############################ Classifier functions #############################
def dataset_pme_cpot(pme_path, cpot_paths, path_save, name_bands, name_montages, features, id_dataset='Data'):
    """
    Crea un dataset con las caracteristicas (valores PME y potencia), parte de 
    los resultados en CSV de las matrices de PME y archivos MAT con los valores 
    de potencia, asocia cada sujeto y los agrupa como un grupo en un archivo CSV.
    """
    txt_file = open(path_save+id_dataset+'_Dataset_Information.txt','w')
    for i_path, path in enumerate(pme_path):
        pme_signals_list = []
        cpot_signals_list = []
        for root, dirs, files in os.walk(path):  
            if files:
                for file in files:
                    file_name, ext = os.path.splitext(file)
                    name_signal = file_name.split("_Channel_")[0]
                    if((ext=='.csv') and (not name_signal in pme_signals_list)):
                        pme_signals_list.append(name_signal)
                        cpot_signals_list.append(name_signal.split("_2s_")[0]+'_2s_qeeg_allchannels.mat')       
        del root, dirs, files, file, file_name, ext, name_signal
        num_errors = 0
        list_dataset = []
        txt_file.write('\n**************************************************************************\n')
        txt_file.write('\nFiles on '+path+' and '+cpot_paths[i_path]+':\n')
        for file, signal_name in enumerate(pme_signals_list):
            txt_file.write('\n** File name:\n-> PME: '+signal_name+'\n-> Cpot: '+cpot_signals_list[file]+'\n\n')
            theta_alpha = np.squeeze(np.squeeze(sio.loadmat(cpot_paths[i_path]+cpot_signals_list[file])['cpot'])[:,[1,3]])
            for channel, channel_name in enumerate(name_montages):
                try:
                    pme = amplitude_modulation.Load_PME(Path_Name_File=path+signal_name+"_Channel_"+channel_name+'.csv', Name_Bands=name_bands).values
                    features_list = []
                    for band in range(0, pme.shape[0]):
                        for m_band in range(0, band+1):
                            features_list.append(pme[band, m_band])
                    del pme, band, m_band 
                    cpot =list(theta_alpha[channel])
                    cpot.append(float(theta_alpha[channel,0]/theta_alpha[channel,1]))
                    features_list += cpot
                    list_dataset.append(features_list)
                    txt_file.write('\n'+channel_name+' channel finished...\n')
                    del features_list, cpot
                except:
                    num_errors += 1
                    txt_file.write('\nError with: '+path+signal_name+"_Channel_"+channel_name+'.csv\n')
                    print('\nError with: '+path+signal_name+"_Channel_"+channel_name+'.csv\n')
            del theta_alpha, channel, channel_name
        txt_file.write('\nTotal Errors: '+str(num_errors)+'\n')
        print('\nTotal Errors: '+str(num_errors)+'\n')
        del num_errors, file, signal_name, pme_signals_list, cpot_signals_list
    del i_path, path
    dataset = np.array(list_dataset)
    del list_dataset
    file_dataset = pd.DataFrame(dataset, columns=features)
    file_dataset.to_csv(path_save+"Dataset_"+id_dataset+".csv")
    del file_dataset, dataset
    txt_file.close()

def pipeline(files_paths_v1, filter_path_save_v1, hamming_path_save_v1,
             files_paths_v2, filter_path_save_v2, hamming_path_save_v2,
             name_montages, bands, name_bands, new_sampling_frequency,
             path_save, paths_cpot_ACr, paths_cpot_Ctrl, features):
    """
    Flujo de trabajo completo para calcular los PME de dos grupos de estudio,
    y crear los dataset a utilizar en un modelo de clasificacion de caracteristicas.
    """
    ################ Obtner el pme de cada sujeto por grupo ###################
    # V1
    calculate_pme(files_paths_v1[0], filter_path_save_v1[0], hamming_path_save_v1[0], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=120, id_group="g1")
    calculate_pme(files_paths_v1[1], filter_path_save_v1[1], hamming_path_save_v1[1], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=120, id_group="g2")
    calculate_pme(files_paths_v1[2], filter_path_save_v1[2], hamming_path_save_v1[2], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=90, id_group="g3")
    calculate_pme(files_paths_v1[3], filter_path_save_v1[3], hamming_path_save_v1[3], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=90, id_group="g4")
    # V2
    calculate_pme(files_paths_v2[0], filter_path_save_v2[0], hamming_path_save_v2[0], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=120, id_group="g1")
    calculate_pme(files_paths_v2[1], filter_path_save_v2[1], hamming_path_save_v2[1], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=120, id_group="g2")
    calculate_pme(files_paths_v2[2], filter_path_save_v2[2], hamming_path_save_v2[2], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=90, id_group="g3")
    calculate_pme(files_paths_v2[3], filter_path_save_v2[3], hamming_path_save_v2[3], name_montages, bands, name_bands, new_sampling_frequency, num_epochs=90, id_group="g4")
    
    ############# Agrupar todos los pme en un mismo archivo ###################
    # V1
    group_pme(filter_path_save_v1, hamming_path_save_v1, path_save, name_montages, name_bands, id_group="G", id_file='v1')
    # V2
    group_pme(filter_path_save_v2, hamming_path_save_v2, path_save, name_montages, name_bands, id_group="G", id_file='v2')

    ########################## Realizar los Datasets ##############################
    ########## Dataset del grupo de portadores asintomaticos (G1) #################
    # V1
    # Filter
    dataset_pme_cpot([filter_path_save_v1[0]], [paths_cpot_ACr[0]], path_save, name_bands, name_montages, features, id_dataset='ACr_v1_filter')
    # Hamming
    dataset_pme_cpot([hamming_path_save_v2[0]], [paths_cpot_ACr[0]], path_save, name_bands, name_montages, features, id_dataset='ACr_v1_hamming')
    # V2
    # Filter
    dataset_pme_cpot([filter_path_save_v2[0]], [paths_cpot_ACr[1]], path_save, name_bands, name_montages, features, id_dataset='ACr_v2_filter')
    # Hamming
    dataset_pme_cpot([hamming_path_save_v2[0]], [paths_cpot_ACr[1]], path_save, name_bands, name_montages, features, id_dataset='ACr_v2_hamming')
    
    ####### Dataset del grupo control - no portadores asintomaticos (G2) ##########
    # V1
    # Filter
    dataset_pme_cpot([filter_path_save_v1[1]], [paths_cpot_Ctrl[0]], path_save, name_bands, name_montages, features, id_dataset='Ctrl_v1_filter')
    # Hamming
    dataset_pme_cpot([hamming_path_save_v2[1]], [paths_cpot_Ctrl[0]], path_save, name_bands, name_montages, features, id_dataset='Ctrl_v1_hamming') 
    # V2
    # Filter
    dataset_pme_cpot([filter_path_save_v2[1]], [paths_cpot_Ctrl[1]], path_save, name_bands, name_montages, features, id_dataset='Ctrl_v2_filter')
    # Hamming
    dataset_pme_cpot([hamming_path_save_v2[1]], [paths_cpot_Ctrl[1]], path_save, name_bands, name_montages, features, id_dataset='Ctrl_v2_hamming')

def model_svc():
    """
    Creacion y prueba de un modelo de clasificacion.
    """
    Features = ['Theta-M_Delta', 'Theta-M_Theta',
                'Alpha-M_Delta', 'Alpha-M_Theta', 'Alpha-M_Alpha']
    Features = ['Alpha1-M_Theta', 
                'Alpha2-M_Theta', 
                'Alpha-M_Theta', 'Theta', 'Alpha']
    
    dataset = []
    labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
    clf = svm.SVC(kernel='poly', degree=2)
    loo = LeaveOneOut()
    
    def loocv(X, Y):
        err = 0
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            err += (y_test - pred)**2
        return err
    
    def normalize(dataset):
        max_value = np.max(dataset, axis=0)
        min_value = np.min(dataset, axis=0)
        norm_dataset = (max_value-dataset)/(max_value-min_value)
        return norm_dataset
    
    #All possible combination of 8 features without repeat
    comb = [list(x) for x in itertools.combinations(Features, 5)]
    poblacion1_f = '/home/marcos/Documents/Local_Repositories/Alzheimer_Classification/Python/Dataset_ACr.csv'
    poblacion2_f = '/home/marcos/Documents/Local_Repositories/Alzheimer_Classification/Python/Dataset_Ctrl.csv'
    #Dataset Completos
    dataset1 = pd.read_csv(poblacion1_f)
    dataset2 = pd.read_csv(poblacion2_f)
    #Creo un dataset para guardar los resultados
    i = 1
    #Realizo pruebas sobre el dataset con los features, sobre combinación sin repeticion
    #24310 posibilidades
    for feat_comb in comb:
        print('Procesando combinación %i/%i'%(i, len(comb)))
        Xs1 = dataset1.loc[:, feat_comb].values
        Xs2 = dataset2.loc[:, feat_comb].values
        Xs1 = normalize(Xs1)
        Xs2 = normalize(Xs2)
        Y1 = np.zeros([len(Xs1)])#class 1 
        Y2 = np.ones([len(Xs2)])#class 2
        X = np.append(Xs1, Xs2, axis=0)
        Y = np.append(Y1, Y2, axis=0)
        d = dict(zip(labels, feat_comb))
        d['Error']=loocv(X,Y)
        dataset.append(d)
        i = i + 1
    
    results = pd.DataFrame(dataset)
    min_err = results['Error'].min()
    menores = results[(results['Error'] == min_err[0])]
    print(menores)

######################### IMPLEMENTACION DEL MODULO ###########################
if __name__=='__main__':
    # Ruta del grupo 1
    g1_path = "C:/Users/Marcos/Documents/pruebas/Group_1/*Group_1.mat"
    g1_save_path = "C:/Users/Marcos/Documents/pruebas/Group_1/Output/amplitude_modulation/"
    # Ruta del grupo 2
    g2_path = "C:/Users/Marcos/Documents/pruebas/Group_2/*Group_2.mat"
    g2_save_path = "C:/Users/Marcos/Documents/pruebas/Group_2/Output/amplitude_modulation/"
    # Ruta donde se guarda el dataset final con ambos grupos unidos 
    g1_g2_save_path = "C:/Users/Marcos/Documents/pruebas/Output/amplitude_modulation/"
    # Clave para ubicar la señal en los archivos mat
    signal_key = 'data'
    # Frecuencia de muestreo original de las señales
    sampling_frequency = 1000
    # Nombres de todos los montages que conforman la señal
    name_montages = ["(FP1+FPZ+FP2)-(P1+PZ+P2)",
                   "1.2(FP1+FPZ+FP2)-(2.2P1+2.7PZ+2.4P2)",
                   "FPZ-PZ",
                   "1.2FPZ-2.7PZ",
                   "PZ",
                   "PZ-(TP8+TP7)"]
    # Bandas de frecuencia en las cuales se va a realizar el analisis
    bands = [(4,8), (8,10), (10,13), (8,13)]
    # Nombres de las bandas utilizadas para el analisis
    name_bands = ['Theta', 'Alpha1', 'Alpha2','Alpha']
    # La frecuencia de remuestreo a utilizar 
    new_sampling_frequency = 500
    ####### verificar el menor numero de epocas entre todas las señales #######
    # Identificativo para el nombre de los archivos de texto con la informacion de las señales
    id_group='min_g1_g2'
    num_epochs, min_epochs = min_epochs_number(files_paths=[g1_path, g2_path], 
                                               path_save=g1_g2_save_path, 
                                               signal_key=signal_key,
                                               id_group=id_group)
    print('\nMenor numero de epocas encontrado en el Grupo 1: ',num_epochs[0],
          '\nMenor numero de epocas encontrado en el Grupo 2: ',num_epochs[1],
          '\nMenor numero de epocas: ',min_epochs)
    ################ Obtner el pme de cada sujeto por grupo ###################
    # Identificativo para el nombre del archivo de texto con la informacion del proceso
    id_group = ['g1', 'g2']
    # Aplicar al grupo 1
    calculate_pme(path=g1_path,  
                  path_save=g1_save_path,
                  sampling_frequency=sampling_frequency,
                  signal_key=signal_key,
                  name_montages=name_montages, 
                  bands=bands, 
                  name_bands=name_bands, 
                  new_sampling_frequency=new_sampling_frequency, 
                  num_epochs=min_epochs, 
                  id_group=id_group[0]) 
    # Aplicar al grupo 2
    calculate_pme(path=g2_path,  
                  path_save=g2_save_path, 
                  sampling_frequency=sampling_frequency,
                  signal_key=signal_key,
                  name_montages=name_montages, 
                  bands=bands, 
                  name_bands=name_bands, 
                  new_sampling_frequency=new_sampling_frequency, 
                  num_epochs=min_epochs, 
                  id_group=id_group[1]) 
    ############# Agrupar todos los pme en un mismo archivo ###################
    # Identificativo para cada grupo y que va en la clave del archivo mat
    id_group = 'g'
    # Identificativo para el nombre del archivo mat que contiene todos los pme
    id_file = 'g1_g2'
    group_pme(path=[g1_save_path, g2_save_path], 
              path_save=g1_g2_save_path, 
              name_montages=name_montages, 
              name_bands=name_bands, 
              id_group=id_group, 
              id_file=id_file)