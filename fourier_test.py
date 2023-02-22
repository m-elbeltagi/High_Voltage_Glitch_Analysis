from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.fftpack import rfft, rfftfreq, irfft
from numpy import genfromtxt
import pickle as pkl
plt.style.use('seaborn-whitegrid')


source_path = r''



current_file = genfromtxt(source_path, delimiter=',')
current_file = current_file[2:,:]


time = current_file[:, 0]
channel_A = current_file[:, 1]
time = np.array(time)
channel_A = np.array(channel_A)

N = len(time)
data_step = (time[-1]-time[0])/N
xf = rfftfreq(N, data_step)

plt.figure(figsize=(8, 7))
plt.plot(time, channel_A, color='k')
plt.xlabel('Time (ms)')
plt.ylabel('(original trace) Voltage (mV)')
plt.show()
#
#
#
trace_spectrum = rfft(channel_A)

# loading the background trace extraceted below, from a 10k sample trace, from 10kV voltage trace
f = open(r'', 'rb')
background_spectrum = pickle.load(f)
f.close()

##loading peak spectrum also form that same 10kV file:
f = open(r'C:\Users\mo_em\Desktop\HV analysis\peak_spectrum.pkl', 'rb')
peak_spectrum = pickle.load(f)
f.close()

## subtract either the background, or the peak, then background

#clean_spectrum = trace_spectrum - background_spectrum
####
clean_spectrum = trace_spectrum - peak_spectrum
clean_spectrum = trace_spectrum - clean_spectrum


clean_trace = irfft(clean_spectrum)
plt.figure(figsize=(8, 7))
plt.plot(time, clean_trace, color='k')
plt.xlabel('Time (ms)')
plt.ylabel('(clean trace) Voltage (mV)')
plt.show()


## plotting og trace - clean trace
difference = channel_A - clean_trace
plt.figure(figsize=(8, 7))
plt.plot(time, difference, color='k')
plt.xlabel('Time (ms)')
plt.ylabel('(difference) Voltage (mV)')
plt.show()



###########################################################

## code used to get the background (or the peak spectrum) spectrum to use as a filter (for traces with 10k samples):

# picking out the peak
new_indices1 = time > -3
new_indices2 = time < 5

new_indices = new_indices1 * new_indices2

peak_trace = new_indices * channel_A

plt.figure(figsize=(8, 7))
plt.plot(time, peak_trace, color='k')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()


# subtracting peak from trace to get the background
trace_background = channel_A - peak_trace

plt.figure(figsize=(8, 7))
plt.plot(time, trace_background, color='k')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()


# transforming the full trace:
yf_trace = rfft(channel_A)
#yf_trace = np.abs(yf_trace)

plt.figure(figsize=(8, 7))
plt.plot(xf, yf_trace)
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude')
plt.show()

#

# transforming background trace
yf_background = rfft(trace_background)
#yf_background = np.abs(yf_background)

plt.figure(figsize=(8, 7))
plt.plot(xf, yf_background)
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude')
plt.show()


# trasforming peak trace
yf_peak = rfft(peak_trace)


# subtracting both

yf_clean = yf_trace - yf_peak
yf_clean = yf_trace - yf_clean
plt.figure(figsize=(8, 7))
plt.plot(xf, yf_clean)
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude (clean)')
plt.show()


inverse_clean = irfft(yf_clean)
plt.figure(figsize=(8, 7))
plt.plot(time, inverse_clean, color='k')
plt.xlabel('Time (ms)')
plt.ylabel('(clean) Voltage (mV)')
plt.show()


#f = open(r'', 'wb')
#pickle.dump(yf_background, f)
#f.close()
    
    
#f = open(r'', 'wb')
#pickle.dump(yf_peak, f)
#f.close()
