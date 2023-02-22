from __future__ import division
from numpy import genfromtxt
import os
from scipy.signal import find_peaks
from scipy.integrate import simps
import pickle
import datetime
from collections import OrderedDict
import numpy as np



## Only thing that needs to be changed between runs is the source path (and the file management stuff)

## ordered dictionary to remember the order in which keys are inserted
data_dict = OrderedDict()

## keeping track of beginning and end time:
print("start time is: {}".format(datetime.datetime.now()))
print(' ')


##total time to collet traces need to be in folder name next to voltage

#source_path = r''
folders = os.listdir(source_path)

for folder in folders:
    

    folder_path = (source_path + '\\' + folder)
    folder_path_inside = folder_path
#    file_names = os.listdir(folder_path)                                ## uncomment the lines based on the way picoscope saved the data
#    folder_path_inside = folder_path + '\\' + file_names[0]     
    file_names = os.listdir(folder_path_inside)
    
    
    
    this_folder_voltage = "{}{}".format(folder_path[folder_path.index('kV')-2], folder_path[folder_path.index('kV')-1])         ##current folder voltage (in kV) value extracted from folder name has to be double digits the way its implemented now, can be sxtracted many other ways
    this_folder_capture_time = float(folder_path.split('_')[-1])      ## the total time (in sec) to capture all traces at this voltage (extracted from folder name)
    
    
    this_folder_num_peaks = 0
#    this_folder_num_errors = 0
    this_folder_amplitudes = []   ### amplitudes for this specific votlage, e.g. amplitudes at 15kV for all the traces at this voltage value (forthis specidif run of course, not all runs) 
    this_folder_energy_values = []  ### now each value is energy of whole trace, instead of individual peak energies
    
    
    #### add something that reads actual voltage value
    
    ## each file is a trace
    for name in file_names:
        current_file_path = (folder_path_inside + '\\' + name)
        current_file = genfromtxt(current_file_path, delimiter=',')
        current_file = current_file[2:,:]
    
        
        time = current_file[:, 0]
        channel_A = current_file[:, 1]


        indices = channel_A > 50
        shift_array = indices * channel_A
        channel_A = channel_A - np.mean(shift_array)
    
        ## finding the glitch peaks in this trace:
        current_peak_indices, _ = find_peaks((-1*channel_A), prominence=50, height = 90, distance =50)
#        current_peak_values = channel_A[current_peak_indices]
        this_folder_num_peaks += len(current_peak_indices)
    
        

        for index in current_peak_indices:
            
            this_folder_amplitudes.append(channel_A[index])
            
        
        ## y first then x, for these integration methods
        this_trace_energy = simps(channel_A**2, time, even='avg')
        this_folder_energy_values.append(this_trace_energy)

## this section used to evalulate energy for each individual peak, now getting replaced with seciton ^ that gets energy of whole trace        
#####################################################################################################################
                ## peak edges finder:
#            try:
#                left_index = index
#                right_index = index
#                
#                
#                
#                while channel_A[left_index] < -5:   #voltage value in mV
#                    if left_index == 0:
#                        break
#                    else:
#                        left_index -= 1
#                
#                while channel_A[right_index] < -5:
#                    if right_index == (len(channel_A)-1):
#                        break
#                    else:
#                        right_index += 1
#                    
#                    
#                peak_window_time = time[left_index : right_index]
#                peak_window_voltage = channel_A[left_index : right_index]
#        
#                voltage_squared = np.square(peak_window_voltage)
#                result = simps(peak_window_time, voltage_squared)
#        
#        
#                
#                this_folder_energy_values.append(result)
#            
#            except:
#                this_folder_num_errors += 1
#                print ("ERROR!")
#                print ("folder: {}".format(folder_path))
#                print ("file : {}".format(name))
#                print ("number of errors in this folder so far: {}".format(this_folder_num_errors))
#                print(' ')
####################################################################################################################################   
    
    ## the estimator of poisson rate mean, and variance (or std dev, I think not necessirily just the sqrt of it, double check) on the estimator
    this_folder_rate = this_folder_num_peaks/this_folder_capture_time    ##outside of file for loop, because capture time is for all the traces in a specific voltage folder
    
    this_folder_rate_error = np.sqrt(this_folder_rate)
    ## storing data in dictionary of ditionaries, one outer key for each folder
    

    
    
    data_dict['voltage_{}kV'.format(this_folder_voltage)] = {'amplitudes':this_folder_amplitudes, 'energies':this_folder_energy_values, 'rate':this_folder_rate, 'rate_error':this_folder_rate_error, 'num_peaks':this_folder_num_peaks, 'capture_time':this_folder_capture_time}
f = open(source_path + '\\' + 'this_run_data.pkl', 'wb')
pickle.dump(data_dict, f)
f.close()

print("Finished this run")
print("Finsih time is: {}".format(datetime.datetime.now()))
print(' ')
