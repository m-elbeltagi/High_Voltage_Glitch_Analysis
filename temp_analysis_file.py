from __future__ import division
import numpy as np
from numpy import genfromtxt
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simps
from scipy.integrate import trapz
import sys
import pickle
import datetime
import time


save_path = 'C:\\Users\\mo_em\\Desktop\\HV analysis\\images2\\'


print(datetime.datetime.now())
print (' ')

## add outside loop to go through the different files 

folder_path = 'C:\\Users\\mo_em\\Desktop\\HV analysis\\3rd_run_csv_files\\43kV_19'
#file_names = os.listdir(folder_path)
#folder_path = folder_path + '\\' + file_names[0]
file_names = os.listdir(folder_path)

#print (file_names)

#test_dict = {'a': {'c':4, 'r':9}, 'b': 2}
#
#a_file = open('C:\\Users\\mo_em\\Desktop\\HV analysis\\5th_run_csv_files\\test_file_delete_me.pkl', 'wb')
#pickle.dump(test_dict, a_file)
#a_file.close()
#print(test_dict)
#
#a_file = open('C:\\Users\\mo_em\\Desktop\\HV analysis\\5th_run_csv_files\\test_file_delete_me.pkl', 'rb')
#loaded_dict = pickle.load(a_file)
#print(loaded_dict)
#print(loaded_dict['a']['r'])
#a_file.close()
#
#
#
#time.sleep(3)
#print(datetime.datetime.now())
#
#sys.exit("FORCE STOPPED!")

#global n_peaks
#global this_file_amplitudes
#global this_folder_energy_values
#global capture_time
n_peaks = 0
this_folder_amplitudes = []   ### amplitudes for this specific votlage, e.g. amplitudes at 15kV for all the traces at this voltage value (forthis specidif run of course, not all runs) 
this_folder_energy_values = []

## save this at the top of each file, read from there
capture_time = 22      ## the total capture time (in seconds) for the all the pulses in this sample, taken from picoscope file





for name in file_names:
    current_file_path = (folder_path + '\\' + name)
    current_file = genfromtxt(current_file_path, delimiter=',')
    current_file = current_file[2:,:]
    
    
    current_peak_value = 0
#    peak_window_size = 1.88     ##chosen manually for this particular sample run
#    time_inc = 0.005       ##roughly 0.005ms between each time sample, so divides the window into 376 pieces
    
    
#     print (current_file)
    time = current_file[:, 0]
    channel_A = current_file[:, 1]
    
    indices = channel_A > 50

    shift_array = indices * channel_A
    
#    print (np.mean(shift_array))
    
    channel_A = channel_A - np.mean(shift_array)
    
    ## distance because sometimes double counts
    current_peak_indices, _ = find_peaks((-1*channel_A), prominence=50, height = 90, distance=50)
    current_peak_values = channel_A[current_peak_indices]
    n_peaks += len(current_peak_indices)
    print("number of peaks in this trace: {}".format(len(current_peak_indices)))
#    print(current_peak_indices)
    print(current_peak_values)
    
    
    ### peak edges finder:
        


#    for index in current_peak_indices:
#        print(index)
#        print('time of this index is {}'.format(time[index]))
#        
#        
##        peak_window_time = time[index-188:index+188]
##        peak_window_voltage = channel_A[index-188:index+188]
#        
#        left_index = index
#        right_index = index
#        
#        while channel_A[left_index] < -10:   #voltage value in mV
#            left_index -= 1
#        
#        while channel_A[right_index] < -10:
#            right_index += 1
#            
#        
#        peak_window_time = time[left_index : right_index]
#        peak_window_voltage = channel_A[left_index : right_index]
#        
##        print (peak_window_time)
#
#        voltage_squared = np.square(peak_window_voltage)
#
#        result = simps(voltage_squared, peak_window_time)
#
#        print ('integration result is {:f}'.format(result))
#
#
#
#        this_folder_amplitudes.append(channel_A[index])
#        this_folder_energy_values.append(result)
#        plt.axvline(x=time[left_index], color = 'r')
#        plt.axvline(x=time[right_index], color = 'r')
#        
    
    
        
    
        
#    print(len(this_folder_amplitudes))
#    print(len(this_folder_energy_values))
    
#    print(n_peaks)
#    plt.plot(time, channel_A)
#    plt.ylabel('TPC line voltage (mV)')       ##channel A is the TPC line voltage
#    plt.xlabel('Time (ms)')
#    
#    plt.axvline(x=time[left_index], color = 'r')
#    plt.axvline(x=time[right_index], color = 'r')
##    plt.savefig(save_path + 'peak_window_sample_plot.png', bbox_inches='tight', dpi=200)
#    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(channel_A)
    plt.plot(current_peak_indices, channel_A[current_peak_indices], 'x')
    plt.ylabel('TPC line voltage (mV)')
    plt.xlabel('Order')
#    plt.savefig(save_path + 'peak_selection_sample_plot.png', bbox_inches='tight', dpi=200)
#    plt.show()
    
    ####
    print ('signal integration result is {}'.format(trapz(channel_A**2, time)))
#    print (simps(channel_A**2, time))
    

    break

# plt.plot(peak_window_time, peak_window_voltage)
# plt.ylabel('TPC line voltage')      
# plt.xlabel('Time')
# plt.show()

# print (current_peak_value)
# print (index)
# print (time[index])
# print (peak_window)

# print (current_peak_indices)
# print(len(current_peak_indices))
# print (channel_A[current_peak_indices])
# print(n_peaks)
    
    

# print (time[1]-time[0])
# print (time[2]-time[1])
# print (time[100]-time[99])
    
#####################################################
    

         