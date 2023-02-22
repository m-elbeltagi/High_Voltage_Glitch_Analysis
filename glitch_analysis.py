from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.integrate import trapz
from scipy import stats
from numpy import genfromtxt
from scipy.stats import norm
plt.style.use('seaborn-whitegrid')


save_path = 'C:\\Users\\mo_em\\Desktop\\HV analysis\\images2\\'

## loads the dictionary saved by the extracct_glitch_data file if option 0, if option 1 loads the processed dictionary, option 2 is the final processed dictionary
def open_dictionary(source_path, option):
    if option == 0: 
        data_path = (source_path + '\\' + 'this_run_data.pkl')
        f = open(data_path, 'rb')
        raw_data = pickle.load(f)
        f.close()
        return raw_data
    if option == 1:
        processed_data_path = (source_path + '\\' + 'processed_data.pkl')
        f = open(processed_data_path, 'rb')
        processed_dict = pickle.load(f)
        f.close()
        return processed_dict
    if option == 2:
        final_data = (source_path + '\\' + 'final_data.pkl')
        f = open(final_data, 'rb')
        final_data = pickle.load(f)
        f.close()
        return final_data


## saves the processed data in another dictionary (needs to be updated for non-normal distributions)
def process_data_simple(raw_data):
    keys = raw_data.keys()
    processed_data = OrderedDict()
    for key in keys:
        # ddof =1 for np.std, to calculate sample mean (using nanmean to ignore nan values), and not popualtion mean
        processed_data[key] = {'amplitude_mean': np.nanmean(raw_data[key]['amplitudes']), 'amplitude_error': np.std(raw_data[key]['amplitudes'], ddof=1), 'energy_mean':np.nanmean(raw_data[key]['energies']), 'energy_error' : np.std(raw_data[key]['energies'], ddof=1), 'rate': raw_data[key]['rate'], 'rate_error': raw_data[key]['rate_error'], 'total_energy_for_this_kV': np.sum(raw_data[key]['energies']), 'num_peaks' : raw_data[key]['num_peaks']}
    # now pickling the dictionary of processed data:
    f = open(source_path + '\\' + 'processed_data.pkl', 'wb')
    pickle.dump(processed_data, f)
    f.close()


def process_data_hist(raw_data):
    keys = raw_data.keys()
    processed_data = OrderedDict()
    for key in keys:
        current_amps = raw_data[key]['amplitudes']
        (mu, sigma) = norm.fit(current_amps)
        n, bins = np.histogram(current_amps, bins='auto')
        
        index_of_max = np.where(n == max(n))
        
        upper_cut_indices = current_amps < bins[index_of_max[0][-1]] + 100
        lower_cut_indices = current_amps > bins[index_of_max[0][-1]] - 100
        cut_indices = upper_cut_indices * lower_cut_indices
        
        current_amps = current_amps * cut_indices
        current_amps = np.delete(current_amps, np.where(current_amps == 0))
        (mu, sigma) = norm.fit(current_amps)
        
        processed_data[key] = {'amplitude_mean': mu, 'amplitude_error': sigma, 'energy_mean':np.nanmean(raw_data[key]['energies']), 'energy_error' : np.std(raw_data[key]['energies'], ddof=1), 'rate': raw_data[key]['rate'], 'rate_error': raw_data[key]['rate_error'], 'total_energy_for_this_kV': np.sum(raw_data[key]['energies']), 'num_peaks' : raw_data[key]['num_peaks']}

# now pickling the dictionary of processed data:
    f = open(source_path + '\\' + 'processed_data.pkl', 'wb')
    pickle.dump(processed_data, f)
    f.close()


## returns data ready for plotting
def final_data(source_path):
    # openeing the processed dictionary:
    data = open_dictionary(source_path, 1)
    data_keys = data.keys()
    
    amplitudes_array = []
    amplitudes_errors_array = []
    mean_energies_array = []
    mean_energies_errors_array = []
    rates_array = []
    rates_errors_array = []
    total_energy_per_glitch = []
    num_peaks_array = []
    this_run_energy_sum = []
    voltage_axis = []



    for key in data_keys:
        amplitudes_array.append(data[key]['amplitude_mean'])
        amplitudes_errors_array.append(data[key]['amplitude_error'])
        mean_energies_array.append(data[key]['energy_mean'])
        mean_energies_errors_array.append(data[key]['energy_error'])
        rates_array.append(data[key]['rate'])
        rates_errors_array.append(data[key]['rate_error'])
        total_energy_per_glitch.append(data[key]['total_energy_for_this_kV']/data[key]['num_peaks'])
        num_peaks_array.append(data[key]['num_peaks'])
        this_run_energy_sum.append(data[key]['total_energy_for_this_kV'])
        voltage_axis.append(float('{}{}'.format(key[key.index('_')+1], key[key.index('_')+2])))



    data = OrderedDict()
    data['amplitudes_array'] = amplitudes_array
    data['amplitudes_errors_array'] = amplitudes_errors_array
    data['mean_energies_array'] = mean_energies_array
    data['mean_energies_errors_array'] = mean_energies_errors_array
    data['rates_array'] = rates_array
    data['rates_errors_array'] = rates_errors_array
    data['total_energy_per_glitch'] = total_energy_per_glitch
    data['voltage_axis'] = voltage_axis
    
    f = open(source_path + '\\' + 'final_data.pkl', 'wb')
    pickle.dump(data, f)
    f.close()





def plot_all_seperate():
    breakdowns = np.array([39, 35, 43, 40, 40, 38, 45.5, 29, 37, 46])
    for i in range(1,11):
        data = eval('run'+str(i))
        
        plt.figure(figsize=(24, 6))
        

        
        plt.subplot(141)
        plt.errorbar(data['voltage_axis'], data['mean_energies_array'], yerr=data['mean_energies_errors_array'], fmt='.k')
        plt.ylabel(r'Energy ($mV^2$.s)')   ## the r before means this is raw string, and not to treat baclslashes as escapes, should probably use this for path strings
        plt.xlabel('Voltage (kV)')
        
        plt.subplot(142)
        plt.errorbar(data['voltage_axis'], data['rates_array'], yerr=data['rates_errors_array'], fmt ='.k')
        plt.ylabel('Rate (Hz)')       
        plt.xlabel('Voltage (kV)')
        
        plt.subplot(143)
        plt.errorbar(np.array(data['voltage_axis']), -1*np.array(data['amplitudes_array']), yerr=np.array(data['amplitudes_errors_array']), fmt='.k')
        plt.ylabel('|Amplitudes| (mV)')
        plt.xlabel('Voltage (kV)')
        
        plt.subplot(144)
        plt.scatter(data['voltage_axis'], data['total_energy_per_glitch'], c ='k')
        plt.ylabel(r'Total Energy per Glitch ($mV^2$.s)')
        plt.xlabel('Voltage (kV)')
        
        plt.suptitle('Run ' + str(i) + ' , breakdown/max voltage = {}'.format(breakdowns[i-1]))
        plt.savefig(save_path + 'separate_plots_run_' + str(i)  + '.png', bbox_inches='tight', dpi=300)
        plt.show()
        print('-----------------------------------------------------------------------------------------------')


def plot_fit_vs_breakdown():
    breakdowns = np.array([39, 35, 0, 40, 40, 0, 45.5, 29, 37, 46])
    params0 = []
    params1 = []
    breakdown_axis = []
    
    def exp_func(x, a, b):
        return a * np.exp(b*x)
    
    
    for i in range(1,11):
        if i == 3 or i == 6:
            continue
        else:
            data = eval('run'+str(i))
            breakdown_axis.append(breakdowns[i-1])
            x_axis = np.array(data['voltage_axis'])
            x_axis = x_axis.astype(np.float)
            y_axis = np.array(data['rates_array'])
            y_err = np.array(data['rates_errors_array'])
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(x_axis, y_axis, yerr = y_err, fmt ='.k')
            
            fit_parameters, pcov = curve_fit(exp_func, xdata=x_axis, ydata=y_axis, sigma=y_err, method = 'trf')   ## for some reason the default 'lm' method fails for the 10th run specifically when I include the rrors for that run, otherwise 'trf' seems to match 'lm', but works for run10 with errors
            plt.plot(x_axis, exp_func(x_axis, *fit_parameters), label='fit')
            plt.legend(loc='upper left', frameon=True, edgecolor='k')
            plt.ylabel('Rate (Hz)')
            plt.xlabel('Voltage (kV)')
            plt.title('Run ' + str(i))
#            plt.savefig(save_path + 'sample_rate_exp_fit', bbox_inches='tight', dpi=200)
            plt.show()
            params0.append(fit_parameters[0])
            params1.append(fit_parameters[1])
            
            break
            
            
##################################################################################################
## attempt tp fit linear comination of fit parameters to a line with the breakdown voltage, but didn't work, best line I could find has 0.699 correlation

#    breakdown_axis = np.array(breakdown_axis)
#    params0 = np.array(params0)
#    params1 = np.array(params1)
#            
#        
#    def lin_combo_func(x, m, b):
#        return m * x + b
#    
#    default_corr = 0.5
#    
#    for i in np.linspace(1, 1000, 100):
#        for j in np.linspace(1, 1000, 100):
#            
#            current_x_data = i * params0 + j * params1
#            
#            combo_fit_params, combo_pcov = curve_fit(lin_combo_func, xdata=current_x_data, ydata=breakdown_axis, method='trf')
#    
#            current_corr, _ = stats.pearsonr(breakdown_axis, lin_combo_func(current_x_data, *combo_fit_params))
#    
#            if current_corr > default_corr:
#                default_corr = current_corr
#                
#                return_combo = [i, j]
#                return_combo_fit_params = combo_fit_params
#                
#    
#    plt.scatter(x=breakdown_axis, y=return_combo_fit_params[0] * (return_combo[0] * params0 + return_combo[1] * params1) * return_combo_fit_params[1], c='k')
#    print('correlation is {}'.format(default_corr))
#    
##    plt.scatter(x=breakdown_axis, y=params0, c='k', label='runs')
##    plt.ylabel('fit parameter')
##    plt.xlabel('Breakdowns (kV)')
##    plt.legend(loc='upper left', frameon=True, edgecolor='k')
##    plt.show()
            
##########################################################################################################


def plot_all_together(form, parameter):
    breakdowns = np.array([39, 35, 44, 40, 40, 39, 45.5, 29, 37, 46])   ## breakdown voltages in kV, no breakdown in run 3 & 6, but using last votlage + 1kV reached as breakdown for them, but mention this on graphs
    colors = ['k','b','tab:brown','r','g','xkcd:salmon','y','m', 'c', 'tab:orange']
    plt.figure(figsize=(10, 6))
    
    if parameter == 'rate':
        for i in range(1,11):
            data = eval('run'+str(i))
            voltage_axis = np.array(data['voltage_axis'])
            rates_array = np.array(data['rates_array'])
            rates_errors_array = np.array(data['rates_errors_array'])
            if i == 3 or i == 6:
                if form == 'scatter':
                    plt.errorbar(voltage_axis/voltage_axis[-1], rates_array, yerr=rates_errors_array, fmt = '.{}'.format(colors[i-1]), label = 'run {},Max @ {}kV, no BD'.format(i, breakdowns[i-1]))
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), rates_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], rates_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
            else:
                if form == 'scatter':
                    plt.errorbar(voltage_axis/voltage_axis[-1], rates_array, yerr=rates_errors_array, fmt = '.{}'.format(colors[i-1]), label = 'run {},BD @ {}kV'.format(i, breakdowns[i-1]))
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), rates_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], rates_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
    
                        
        plt.ylabel('Rate (Hz)')
        plt.xlabel('Normalized Voltage')
    #    ax = plt.gca()
    #    ax.set_facecolor('xkcd:salmon')
        plt.legend(loc='upper left', frameon=True, edgecolor='k')
        plt.savefig(save_path + 'all_rates_plot.png', bbox_inches='tight', dpi=200)
        plt.show()
        
    if parameter == 'energy':
        for i in range(1,11):
            data = eval('run'+str(i))
            voltage_axis = np.array(data['voltage_axis'])
            mean_energies_array = np.array(data['mean_energies_array'])
            mean_energies_errors_array = np.array(data['mean_energies_errors_array'])
            if i == 3 or i == 6:
                if form == 'scatter':
                    plt.errorbar(voltage_axis/voltage_axis[-1], mean_energies_array, yerr=mean_energies_errors_array, fmt = '.{}'.format(colors[i-1]), label = 'run {},Max @ {}kV, no BD'.format(i, breakdowns[i-1]))
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), mean_energies_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], mean_energies_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
            else:
                if form == 'scatter':
                    plt.errorbar(voltage_axis/voltage_axis[-1], mean_energies_array, yerr=mean_energies_errors_array, fmt = '.{}'.format(colors[i-1]), label = 'run {},BD @ {}kV'.format(i, breakdowns[i-1]))
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), mean_energies_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], mean_energies_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
    
                        
        plt.ylabel(r'Energy ($mV^2$.s)')
        plt.xlabel('Normalized Voltage')
    #    ax = plt.gca()
    #    ax.set_facecolor('xkcd:salmon')
        plt.legend(loc='upper left', frameon=True, edgecolor='k')
        plt.savefig(save_path + 'all_energies_plot.png', bbox_inches='tight', dpi=200)
        plt.show()
        
    if parameter == 'amplitude':
        for i in range(1,11):
            data = eval('run'+str(i))
            voltage_axis = np.array(data['voltage_axis'])
            amplitudes_array = -1*np.array(data['amplitudes_array'])
            amplitudes_errors_array = np.array(data['amplitudes_errors_array'])
            if i == 3 or i == 6:
                if form == 'scatter':
                    plt.errorbar(voltage_axis/voltage_axis[-1], amplitudes_array, yerr=amplitudes_errors_array, fmt = '.{}'.format(colors[i-1]), label = 'run {},Max @ {}kV, no BD'.format(i, breakdowns[i-1]))
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), rates_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], amplitudes_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
            else:
                if form == 'scatter':
                    plt.errorbar(voltage_axis/voltage_axis[-1], amplitudes_array, yerr=amplitudes_errors_array, fmt = '.{}'.format(colors[i-1]), label = 'run {},BD @ {}kV'.format(i, breakdowns[i-1]))
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), rates_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], amplitudes_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
    
                        
        plt.ylabel('|Amplitudes| (mV)')
        plt.xlabel('Normalized Voltage')
    #    ax = plt.gca()
    #    ax.set_facecolor('xkcd:salmon')
        plt.legend(loc='upper center', frameon=True, edgecolor='k')
        plt.savefig(save_path + 'all_amplitudes_plot.png', bbox_inches='tight', dpi=200)
        plt.show()

    if parameter == 'energy_density':
        for i in range(1,11):
            data = eval('run'+str(i))
            voltage_axis = np.array(data['voltage_axis'])
            density_array = np.array(data['total_energy_per_glitch'])
            if i == 3 or i == 6:
                if form == 'scatter':
#                        plt.errorbar(voltage_axis/voltage_axis[-1], density_array, yerr=, fmt = '.{}'.format(colors[i-1]), label = 'run {},Max @ {}kV, no BD'.format(i, breakdowns[i-1]))
                    pass
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), density_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], density_array, colors[i-1], label = 'run {},Max @  {}kV, no BD'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
            else:
                if form == 'scatter':
#                        plt.errorbar(voltage_axis/voltage_axis[-1], density_array, yerr=, fmt = '.{}'.format(colors[i-1]), label = 'run {},BD @ {}kV'.format(i, breakdowns[i-1]))
                    pass
                if form == 'line':
#                    plt.plot(((abs(voltage_axis-breakdowns[i-1]))/breakdowns[i-1]), density_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))    ## "reduced" voltage axis
                    plt.plot(voltage_axis/voltage_axis[-1], density_array, colors[i-1], label = 'run {},BD @  {}kV'.format(i, breakdowns[i-1]))          #normalized voltage axis, comment the one you're not using
    
                        
        plt.ylabel(r'Total Energy per Glitch ($mV^2$.s)')
        plt.xlabel('Normalized Voltage')
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0,46000))
    #    ax = plt.gca()
    #    ax.set_facecolor('xkcd:salmon')
        plt.legend(loc='upper left', frameon=True, edgecolor='k')
        plt.savefig(save_path + 'all_energy_densities_plot.png', bbox_inches='tight', dpi=200)
        plt.show()



def plot_energy_vs_rate(form):
    for i in range(1,11):
        plt.figure(figsize=(8, 6))
        data = eval('run'+str(i))
        rates_array = np.array(data['rates_array'])
        rates_errors_array = np.array(data['rates_errors_array'])
        mean_energies_array = np.array(data['mean_energies_array'])
        mean_energies_errors_array = np.array(data['mean_energies_errors_array'])
        if form == 'error':
            plt.errorbar(x=rates_array, y=mean_energies_array, yerr=mean_energies_errors_array, xerr=rates_errors_array, fmt ='.k')
        if form == 'scatter':
            plt.scatter(x=rates_array, y=mean_energies_array)
        plt.ylabel(r'Energy ($mV^2$.s)')
        plt.xlabel('Rate (Hz)')
        plt.title('Run ' + str(i))
        plt.show()
        corr, _ = stats.pearsonr(rates_array, mean_energies_array)
        print('Correlation coefficient is {}'.format(corr))
        print('-------------------------------')




def plot_max_vs_breakdown(parameter):
    breakdowns = np.array([39, 35, 0, 40, 40, 0, 45.5, 29, 37, 46, 41, 43, 43])   ## breakdown voltages in kV, no breakdown in run 3 & 6, add run 10 (with breakdown 46kV), after adjusting Volt to mV
    x_axis = []
    y_axis = []
    y_err = []
    plt.figure(figsize=(8, 6))
    for i in range(1,11):
        if i == 3 or i == 6:
                continue
        else:
            data = eval('run'+str(i))
            x_axis.append(breakdowns[i-1])
            if parameter == 'energy':
                y_axis.append(data['mean_energies_array'][-1])
                y_err.append(data['mean_energies_errors_array'][-1])
            if parameter == 'rate':
                y_axis.append(data['rates_array'][-1])
                y_err.append(data['rates_errors_array'][-1])
            if parameter == 'amplitude':
                y_axis.append(data['amplitudes_array'][-1] * -1)
                y_err.append(data['amplitudes_errors_array'][-1])
            if parameter == 'energy_density':
                y_axis.append(data['total_energy_per_glitch'][-1])
                y_err.append(0.01) ## place holder till I make the errors array for energy density
            

    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    y_err = np.array(y_err)
    x_axis = x_axis.astype(np.float)
    corr, _ = stats.pearsonr(x_axis, y_axis)
    plt.errorbar(x_axis, y_axis, yerr = y_err, fmt ='.k', label = 'runs, correlation: {0:.2f}'.format(corr))
    plt.xlabel('Breakdown Voltage (kV)')
    if parameter == 'energy':
        plt.ylabel(r'Max Energy ($mV^2$.s)')
    if parameter == 'rate':
        plt.ylabel('Max Rate (Hz)')
    if parameter == 'amplitude':
        plt.ylabel('Max |Amplitudes| (mV)')
    if parameter == 'energy_density':
        plt.ylabel(r'Max Total Energy per Glitch ($mV^2$.s)')
    
    def exp_func(x, a, b):
        return a * np.exp(b*x)

    def lin_func(x,a,b):
        return a*x + b
    

    fit_parameters, pcov = curve_fit(lin_func, xdata=x_axis, ydata=y_axis, sigma=y_err)
    plt.plot(x_axis, lin_func(np.array(x_axis), *fit_parameters), label='fit')
    
    
    
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    if parameter == 'rate':
        plt.savefig(save_path + 'breakdown_vs_max_rate_plot.png', bbox_inches='tight', dpi=200)
    if parameter == 'energy':
        plt.savefig(save_path + 'breakdown_vs_max_energy_plot.png', bbox_inches='tight', dpi=200)
    if parameter == 'amplitude':
        plt.savefig(save_path + 'breakdown_vs_max_amplitude_plot.png', bbox_inches='tight', dpi=200)
    if parameter == 'energy_density':
        plt.savefig(save_path + 'breakdown_vs_max_energy_density.png', bbox_inches='tight', dpi=200)
        
    plt.show()
    
    print('Correlation coefficient is {}'.format(corr))
    print('-------------------------------------------')


def plot_aggregate():
    breakdowns = np.array([39, 35, 40, 40, 45.5, 29, 37, 46])  ## 10 runs excluding runs 3, and 6 with no breakdowns, 41, 43, 43 are the breakdowns of the remaining 3 if you want to include later
    breakdown_stem_weights = np.array([40000, 40000, 40000, 39000, 40000, 40000, 40000, 40000])
    ## opening the processed data, not the final data
    run1 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\1st_run_csv_files', 1)
    run2 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\2nd_run_csv_files', 1)
    run3 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\3rd_run_csv_files', 1)
    run4 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\4th_run_csv_files', 1)
    run5 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\5th_run_csv_files', 1)
    run6 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\6th_run_csv_files', 1)
    run7 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\7th_run_csv_files', 1)
    run8 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\8th_run_csv_files', 1)
    run9 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\9th_run_csv_files', 1)
    run10 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\10th_run_csv_files', 1)
    run11 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\11th_run_csv_files', 1)
    run12 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\12th_run_csv_files', 1)
    run13 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\13th_run_csv_files', 1)


    check_voltages = [v for v in range(10, 44, 2)]
    total_energy_across_runs = [0 for i in range(len(check_voltages))]
    total_n_peaks_across_runs = [0 for i in range(len(check_voltages))]


    for i in range(1,11):  ##only using original 10 runs right now
        if i==1 or i==2 or i==3 or i==6 or i==7:
            data = eval('run'+str(i))
            keys = [float('{}{}'.format(key[key.index('_')+1], key[key.index('_')+2])) for key in data.keys()]
            keys = np.array(keys)
            keys = keys.astype(int)
            keys = list(keys)
            
    #        if i == 1:
    #            print not(math.isnan(data['voltage_{}kV'.format(10)]['total_energy_for_this_kV']))
    #            print (data['voltage_{}kV'.format(10)]['num_peaks'])
            
            for k in range(len(check_voltages)):
                for j in range(len(keys)):
                    if 0 <= keys[j]-check_voltages[k] <= 1 and (not(math.isnan(keys[j]-check_voltages[k]))):
                        total_energy_across_runs[k] += data['voltage_{}kV'.format(keys[j])]['total_energy_for_this_kV']
                        total_n_peaks_across_runs[k] += data['voltage_{}kV'.format(keys[j])]['num_peaks']
        else:
            continue


    total_energy_across_runs = np.array(total_energy_across_runs)
    total_n_peaks_across_runs = np.array(total_n_peaks_across_runs)
    
    total_energy_density_across_runs = total_energy_across_runs/total_n_peaks_across_runs
    errors_for_this = (total_energy_across_runs/(total_n_peaks_across_runs)**2) * 0.5**2 * np.sqrt(10)  ## very rough estimate, see notes
    
    # manually adding the empty values on eith side to make it look better:
    total_energy_density_across_runs = np.insert(total_energy_density_across_runs, 0, 0)
    total_energy_density_across_runs = np.insert(total_energy_density_across_runs, 0, 0)
    total_energy_density_across_runs = np.insert(total_energy_density_across_runs, len(total_energy_density_across_runs), 0)
    total_energy_density_across_runs = np.insert(total_energy_density_across_runs, len(total_energy_density_across_runs), 0)
    total_energy_density_across_runs = np.insert(total_energy_density_across_runs, len(total_energy_density_across_runs), 0)
    total_energy_density_across_runs = np.insert(total_energy_density_across_runs, len(total_energy_density_across_runs), 0)
    total_energy_density_across_runs[2] = 0
    
    check_voltages = np.array(check_voltages)
    check_voltages = np.insert(check_voltages, 0, 9)
    check_voltages = np.insert(check_voltages, 0, 8)
    check_voltages = np.insert(check_voltages, len(check_voltages), 43)
    check_voltages = np.insert(check_voltages, len(check_voltages), 44)
    check_voltages = np.insert(check_voltages, len(check_voltages), 45)
    check_voltages = np.insert(check_voltages, len(check_voltages), 46)
    
    
#    print (total_energy_across_runs)
#    print (len(total_energy_across_runs))
#    print (total_n_peaks_across_runs)
#    print (len(total_n_peaks_across_runs))
#    
#    print(total_energy_density_across_runs)
#    print (len(total_energy_density_across_runs))
#    print(len(errors_for_this))
#    print (errors_for_this)
    print (check_voltages)
    print (total_energy_density_across_runs[2])
#    print (len(check_voltages))

    
    plt.figure(figsize=(10, 6))
#    plt.bar(check_voltages, total_energy_density_across_runs, bottom=None, label = r'aggregate run data')
    plt.step(check_voltages, total_energy_density_across_runs, where='mid', label = r'aggregate run data, 2 runs no BD')
#    plt.stem(breakdowns, breakdown_stem_weights, basefmt=' ', markerfmt='rD', linefmt='k-', label = 'breakdowns')
    plt.stem([39, 35, 40, 45.5], [40000, 40000, 40000, 40000], basefmt=' ', markerfmt='rD', linefmt='k-', label = 'breakdowns')
    plt.stem([39, 44], [40000, 40000], basefmt=' ', markerfmt='yD', linefmt='k-', label = 'max voltage')
#    plt.fill_between(check_voltages, total_energy_density_across_runs-errors_for_this, total_energy_density_across_runs+errors_for_this, where=None, step='mid')
    plt.fill_between(check_voltages, np.zeros(len(check_voltages)), total_energy_density_across_runs, where=None, step='mid', alpha=0.3, color='b')
    plt.xlabel(r'Voltage (kV)')
    plt.ylabel(r'Total Energy/Total # glitches, across all runs $(mV^2.s)$')
    plt.legend(loc='upper left', frameon=True, edgecolor='k')
    plt.title('Feedthrough Runs Excluded')
#    plt.savefig(save_path + 'aggregate_energy_density_plot.png', bbox_inches='tight', dpi=200)
    plt.show()

################################################################################################################


#source_path = 'C:\\Users\\mo_em\\Desktop\\HV analysis\\13th_run_csv_files'


run1 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\1st_run_csv_files', 2)
run2 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\2nd_run_csv_files', 2)
run3 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\3rd_run_csv_files', 2)
run4 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\4th_run_csv_files', 2)
run5 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\5th_run_csv_files', 2)
run6 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\6th_run_csv_files', 2)
run7 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\7th_run_csv_files', 2)
run8 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\8th_run_csv_files', 2)
run9 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\9th_run_csv_files', 2)
run10 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\10th_run_csv_files', 2)
run11 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\11th_run_csv_files', 2)
run12 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\12th_run_csv_files', 2)
run13 = open_dictionary('C:\\Users\\mo_em\\Desktop\\HV analysis\\13th_run_csv_files', 2)




#plot_aggregate()