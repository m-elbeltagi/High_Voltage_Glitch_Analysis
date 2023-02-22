from __future__ import division
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.stats import norm
from scipy import stats
plt.style.use('seaborn-whitegrid')

## this section opens the raw data dictionary and save it to a proccesed data dict, it uses the simple formula assuming underlying gaussian dist for std devs:
save_path = 'C:\\Users\\mo_em\\Desktop\\HV analysis\\images2\\'

source_path = 'C:\\Users\\mo_em\\Desktop\\HV analysis\\10th_run_csv_files'
data_path = (source_path + '\\' + 'this_run_data.pkl')
f = open(data_path, 'rb')
raw_data = pickle.load(f)
f.close()
#keys = raw_data.keys()

#print(keys)

test = raw_data['voltage_43kV']['amplitudes']
##
#print (len(test))
print (np.mean(test))
#print(np.std(test, ddof = 1))
#
#

(mu, sigma) = norm.fit(test)
n, bins = np.histogram(test, bins='auto')

index_of_max = np.where(n == max(n))
print (index_of_max[0][-1])

upper_cut_indices = test < bins[index_of_max[0][-1]] + 100
lower_cut_indices = test > bins[index_of_max[0][-1]] - 100
cut_indices = upper_cut_indices * lower_cut_indices

test = test * cut_indices
test = np.delete(test, np.where(test == 0))
(mu, sigma) = norm.fit(test)
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(test, bins='auto', range = (bins[index_of_max[0][-1]] - 150, bins[index_of_max[0][-1]] + 100), normed=1)

plt.ylabel('Normalized Counts')
plt.xlabel('Amplitude (mV)')
y = stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=2, label = r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
plt.legend(loc='upper left', frameon=True, edgecolor='k')
#plt.savefig(save_path + 'new_analsis_sample.png', bbox_inches='tight', dpi=200)
plt.show()
print (mu, sigma)



storing processed data in a dictionary
processed_data = OrderedDict()

for key in keys:
   # ddof =1 for np.std, to calculate sample mean, and not popualtion mean
   processed_data[key] = {'amplitude_mean': np.mean(raw_data[key]['amplitudes']), 'amplitude_error': np.std(raw_data[key]['amplitudes'], ddof=1), 'energy_mean':np.mean(raw_data[key]['energies']), 'energy_error' : np.std(raw_data[key]['energies'], ddof=1), 'rate': raw_data[key]['rate'], 'rate_error': raw_data[key]['rate_error']}
 
 
now pickling the dictionary of processed data:
a = open(source_path + '\\' + 'processed_data.pkl', 'wb')
pickle.dump(processed_data, a)
a.close()


#####################################################

## putting the means & std devs caculated above into arrays:


loading processed data:
processed_data_path = (source_path + '\\' + 'processed_data.pkl')
f1 = open(processed_data_path, 'rb')
data = pickle.load(f1)
f1.close()
data_keys = data.keys()

amplitudes_array = []
amplitudes_errors_array = []
energies_array = []
energies_errors_array = []
rates_array = []
voltage_axis = []

#######################################################


for key in data_keys:
   amplitudes_array.append(data[key]['amplitude_mean'])
   amplitudes_errors_array.append(data[key]['amplitude_error'])
   energies_array.append(data[key]['energy_mean'])
   energies_errors_array.append(data[key]['energy_error'])
   rates_array.append(data[key]['rate'])
   
   voltage_axis.append(float('{}{}'.format(key[key.index('_')+1], key[key.index('_')+2])))
    
    
##################################################################
    
## section to find mean & std dev using integration of histogram: (put these in sections in functions or classes to clean up the code!)   
## the confidence intervals will follow the simple ordering rule of equal area surrounding the mean, so for 1sigma, the total area covered 68.27%, so each area around the mean will be 34.135%
    
for key in keys:
    # first calculating the mean  for each histogram:
    
   mean = simps(np.multiply())
    
    
    
    
    
##################################################################

## plotting section:
    
plt.plot(voltage_axis,amplitudes_array)
plt.errorbar(voltage_axis, amplitudes_array, yerr=amplitudes_errors_array, fmt='.k')
plt.ylabel('Amplitudes (mV)')       
plt.xlabel('Voltage (kV)')
   
#############

def exp_func(x, a, b):
   return a * np.exp(b*x)

fit_parameters, pcov = curve_fit(exp_func, voltage_axis, energies_array, sigma=energies_errors_array)    

print (fit_parameters)

plt.plot(voltage_axis, energies_array)
plt.errorbar(voltage_axis, energies_array, yerr=energies_errors_array, fmt='.k', label='data')
plt.ylabel('Energy')       
plt.xlabel('Voltage (kV)')
plt.plot(voltage_axis, exp_func(np.array(voltage_axis), *fit_parameters), label='fit')
    
 
 ####################################################
 
obs_path = r'C:\Users\mo_em\Desktop\HV analysis\5th_run_csv_files\39kV_20\voltages\39kV_20_013.csv'

spice_path = r'C:\Users\mo_em\Desktop\HV analysis\spice_scope_voltage2.txt'


obs_data = genfromtxt(obs_path, delimiter=',')
obs_data = obs_data[2:,:]

time1 = obs_data[:, 0]-2.6
channel_A = obs_data[:, 1]-100

spice_data = genfromtxt(spice_path)
spice_data = spice_data[2:,:]*1000

time2 = spice_data[:, 0]-15
scope = spice_data[:, 1]

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(time1[5500:-4000], channel_A[5500:-4000], label = 'observed glitch')
plt.legend(loc='upper left', frameon=True, edgecolor='k')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')



#print (len(time2))
#print (len(scope))

plt.subplot(212)
plt.plot(time2[100: 240], scope[100: 240], label = 'simulated glitch', color = 'r')
plt.legend(loc='upper left', frameon=True, edgecolor='k')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
##plt.savefig(save_path + 'double_glitch_comparison.png', bbox_inches='tight', dpi=200)
plt.show()


########################################


plt.figure(figsize=(10, 6))

plt.subplot(111)
plt.plot(time1[5500:-4000], channel_A[5500:-4000], label = 'observed glitch')

plt.subplot(111)
plt.plot(time2[100: 280], scope[100: 280], label = 'simulated glitch', color = 'r')
plt.legend(loc='upper left', frameon=True, edgecolor='k')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
#plt.savefig(save_path + 'glitch_comparison_1.png', bbox_inches='tight', dpi=200)
plt.show()
