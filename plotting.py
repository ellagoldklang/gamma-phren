from neuron import h
from neuron.units import ms, mV
h.load_file('stdrun.hoc')
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time 
from scipy.signal import butter, filtfilt, hilbert, spectrogram


start = time.time()

def run_simulation(c4):

    class Cell():
        def __init__(self):
            self._setup_morphology()
            self.all = self.soma.wholetree() # get all sections in the cell
            self.all2 = self.dend.wholetree() # get all sections in the dendrite
            self._setup_biophysics()


    # excitatory neuron code 
    class Excitatory_neuron(Cell):   ### IMPORTANT - if it doesn't work, check these values
        name = 'ExcitatoryNeuron'

        def _setup_morphology(self):
            self.soma = h.Section(name='soma') #soma section
            self.soma.L = 30 # length in micrometers
            self.soma.diam = 30 # diameter in micrometers
            self.soma.nseg = 1
            self.soma.insert('extracellular')


            self.dend = h.Section(name='dend') # dendritic section 
            self.dend.L = 500 # length in micrometers
            self.dend.diam = 1 # diameter in micrometers
            self.dend.nseg = 21
            self.dend.insert('extracellular')

            # connect the dendrite to the soma 
            self.dend.connect(self.soma(1))
        
        def _setup_biophysics(self): # biophysics for the soma
            for sec in self.all:
                sec.cm = 1    # µF/cm²
                sec.Ra = 150  # Ohm·cm (axial resistance)

            for sec in self.all2:
                sec.cm = 1
                sec.Ra = 150

            self.soma.insert('hh')
            for seg in self.soma:
                seg.hh.gnabar = 0.1  # Na⁺ conductance (S/cm²)
                seg.hh.gkbar = 0.02  # K⁺ conductance (S/cm²)   # possibly change the value to 0.3 - 0.4 in case
                seg.hh.gl = 0.0003   # leak conductance (S/cm²)
                seg.hh.el = -67     # leak reversal (mV)

            self.dend.insert('pas')  # passive properties for dendrite
            for seg in self.dend:
                seg.pas.g = 0.001    # S/cm²
                seg.pas.e = -67       # mV

            # creating the synpase on the dendrite 
        
            self.syn = h.ExpSyn(self.dend(0.5))

        #IMPORTANT - add some noise into the system 
        #add more dendrites onto the soma - figure out later


    # GENERAL - duplicate the neuron network a bunch of times 

        
    # inhibitory interneuron code

    class inhibitory_neuron(Cell):
        name = 'InhibitoryNeuron'

        def _setup_morphology(self):
            self.soma = h.Section(name='soma_inhibit') #soma section
            self.soma.L = 20 # length in micrometers
            self.soma.diam = 10 # diameter in micrometers
            self.soma.nseg = 1
            self.soma.insert('extracellular')


            self.dend = h.Section(name='dend_inhibit') # dendritic section 
            self.dend.L = 300 # length in micrometers
            self.dend.diam = 1 # diameter in micrometers
            self.dend.nseg = 5
            self.dend.insert('extracellular')

            # connect the dendrite to the soma 
            self.dend.connect(self.soma(1))
        
        def _setup_biophysics(self): # biophysics for the soma. ### IMPORTANT - if it doesn't work, check these values
            for sec in self.all:
                sec.cm = 1    # µF/cm²
                sec.Ra = 100  # Ohm·cm (axial resistance)

            for sec in self.all2:
                sec.cm = 1
                sec.Ra = 150

            self.soma.insert('hh')
            for seg in self.soma:
                seg.hh.gnabar = 0.035  # Na⁺ conductance (S/cm²)
                seg.hh.gkbar = 0.009  # K⁺ conductance (S/cm²)
                seg.hh.gl = 0.0001    # leak conductance (S/cm²)
                seg.hh.el = -67     # leak reversal (mV)

            self.dend.insert('pas')  # passive properties for dendrite
            for seg in self.dend:
                seg.pas.g = 0.0001    # S/cm²
                seg.pas.e = -65       # mV

            # creating the synpase on the dendrite 
        
            self.syn = h.ExpSyn(self.dend(0.5))

        #IMPORTANT - add some noise into the system 
        #add more dendrites onto the soma? - figure out later
        # you probably have to edit the biophysics values soon, they look way to off - check later !!\

    def remove_synapse(conn_list, percent):
        n_remove = int(len(conn_list) * percent)
        if n_remove == 0:
            return conn_list
        remove_indices = set(random.sample(range(len(conn_list)), n_remove))
        return [conn for idx, conn in enumerate(conn_list) if idx not in remove_indices]


    # n = round(sigmoid * number)

    dt = 0.1 # in ms
    time = 1000 # in ms
    nt = int(time / dt)
    t_points = np.arange(0, time, dt)


    # simulating the amount of pairs (n)

    # Instantiate both neurons
    e_cell = [Excitatory_neuron() for _ in range(n)]
    i_cell = [inhibitory_neuron() for _ in range(n)]

    # connection from excitatory to inhibiton and the others together as well
    e_to_i = []
    i_to_e = []


    for i in range(n):
        # randomness variables
        ei_probability = random.uniform(0.45, 0.5)
        ie_probability = random.uniform(0.4, 0.6)

        # excitatory to inhibitory
        if random.random() < ei_probability:
            nc_ei = h.NetCon(e_cell[i].soma(0.5)._ref_v, i_cell[i].syn, sec=e_cell[i].soma)
            nc_ei.threshold = 0 # in mV, the threshold for the connection to trigger 
            nc_ei.delay = random.normalvariate(3.5, 0.581)    # in ms
            nc_ei.weight[0] = 0.25  # in μS, the amount of current given to inhibition # make sure to convert this to a for loop if you decide to add multiple neurons 
            e_cell[i].syn.tau = 3 * ms  
            e_to_i.append(nc_ei)

        # inhibitory to excitatory
        if random.random() < ie_probability:
            e_cell[i].inh_syn = h.ExpSyn(e_cell[i].soma(0.5))
            e_cell[i].inh_syn.tau = 9 * ms
            e_cell[i].inh_syn.e = -75      # inhibitory reversal potential in mV
            nc_ie = h.NetCon(i_cell[i].soma(0.5)._ref_v, e_cell[i].inh_syn, sec=i_cell[i].soma)
            nc_ie.threshold = 0        # mV, spike detection threshold for inhibitory neuron
            nc_ie.delay = random.normalvariate(9.5, 0.97)            # ms, can adjust as needed
            nc_ie.weight[0] = 0.25     # μS, synaptic strength (experiment with values 0.01 - 0.1)
            i_to_e.append(nc_ie)

    # excitatory to excitatory 
    e_to_e = []
    for pre_idx in range(n):
        ee_probability = random.uniform(0.06, 0.1)
        for post_idx in range(n):
            if pre_idx == post_idx:
                continue  # skip self-connection
            if random.random() < ee_probability:  # Only make a connection with given probability
                nc = h.NetCon(e_cell[pre_idx].soma(0.5)._ref_v, e_cell[post_idx].syn, sec=e_cell[pre_idx].soma)
                nc.threshold = 0
                nc.delay = random.normalvariate(3.5, 0.581)
                nc.weight[0] = 0.25
                e_cell[i].syn.tau = 3 * ms 
                e_to_e.append(nc)

    # inhibitory to inhibitory
    i_to_i = []
    for pre_idx in range(n):
        ii_probability = random.uniform(0.3, 0.4)
        for post_idx in range(n):
            if pre_idx == post_idx:
                continue
            if random.random() < ii_probability:
                i_cell[post_idx].inh_syn = h.ExpSyn(i_cell[post_idx].soma(0.5))
                i_cell[post_idx].inh_syn.tau = 9 * ms
                i_cell[post_idx].inh_syn.e = -75      # inhibitory reversal potential in mV
                nc_ie = h.NetCon(i_cell[pre_idx].soma(0.5)._ref_v, i_cell[post_idx].inh_syn, sec=i_cell[pre_idx].soma)
                nc_ie.threshold = 0        # mV, spike detection threshold for inhibitory neuron
                nc_ie.delay = random.normalvariate(9.5, 0.97)            # ms, can adjust as needed
                nc_ie.weight[0] = 0.25     # μS, synaptic strength (experiment with values 0.01 - 0.1)
                i_to_i.append(nc_ie)


    # removing the synapses
    percent_remove = (((100) / (1 + (math.e ** (-0.05555556 * (c4 - 45))))) / 100)
    percent_remove_inhibitory = percent_remove / 4

    e_to_i = remove_synapse(e_to_i, percent_remove)
    i_to_e = remove_synapse(i_to_e, percent_remove_inhibitory)
    e_to_e = remove_synapse(e_to_e, percent_remove)
    i_to_i = remove_synapse(i_to_i, percent_remove_inhibitory)


    # ploting extracellular measurements





    # adding noise and injecting current into the neuron
    # comments about noise: 1] only 1 neuron gets the noise, 2] the excitatory neuron does not recieve the noise 



    baseline = 0.5 # in nA
    noise_strength = 0.1 # in nA
    e_noise = []
    i_noise = []

    for i in range(n):
        # fix the noise stuff later 
        stim = h.IClamp(e_cell[i].soma(0.5))
        stim.delay = 0
        stim.dur = time  # Always on
        stim.amp = 0.5
        e_noise.append(stim)





    # #LFP section

    # 1. Record synaptic currents (this part is good)
    t_vec = h.Vector().record(h._ref_t)
    v_e = [h.Vector().record(cell.soma(0.5)._ref_v) for cell in e_cell]
    v_i = [h.Vector().record(cell.soma(0.5)._ref_v) for cell in i_cell]


    # 2. Run the simulation
    h.finitialize(-65)
    h.continuerun(time)


    # Convert recorded vectors to numpy arrays for easier handling
    v_i_mat = np.array([np.array(vec) for vec in v_i])

    # LFP = mean across all excitatory (and optionally, inhibitory) neurons at each time point
    # This is the "network LFP"
    v_e_mat = np.array([np.array(vec) for vec in v_e])
    lfp = np.mean(v_e_mat, axis=0)

    return lfp, t_vec


n = 1 # number of pairs
num_runs = 5
c4_levels = [105]

for c4 in c4_levels:
    lfp_all = []

    for run in range(num_runs):
        print(f"Running simulation {run+1}/{num_runs} with C4 = {c4} mg/dL")
        lfp, t_vec = run_simulation(c4)
        lfp_all.append(lfp)

    lfp_all = np.array(lfp_all)  # shape (num_runs, n_timepoints)
    avg_lfp = np.mean(lfp_all, axis=0)

    transient_ms = 50
    dt = 0.1 
    start_idx = int(transient_ms / dt)
    lfp_valid = avg_lfp[start_idx:]
    lfp_all_valid = lfp_all[:, start_idx:]
    t_vec_valid = np.array(t_vec)[start_idx:]






 



    # --- Fourier/Powerspectrum Analysis of Simulated LFP ---

    # 1. Remove initial transient (e.g., first 50 ms)

    # 2. Remove DC offset (mean subtraction)
    lfp_valid = lfp_valid - np.mean(lfp_valid)

    # 3. FFT
    lfp_fft = np.fft.fft(lfp_valid)
    freqs = np.fft.fftfreq(len(lfp_valid), d=dt/1000)  # dt is in ms; convert to seconds
    lfp_fft_all = np.fft.fft(lfp_all_valid, axis=1)
    freqs_all = np.fft.fftfreq(lfp_all_valid.shape[1], d=dt/1000)


    # 4. Only use positive frequencies
    pos_mask = freqs > 0
    pos_mask_all = freqs_all > 0
    freqs_pos = freqs[pos_mask]
    freqs_pos_all = freqs_all[pos_mask_all]
    power = np.square(np.abs(lfp_fft[pos_mask]))
    power_all = np.square(np.abs(lfp_fft_all[:, pos_mask_all]))




    # Average amplitude code - dont touch this lowk

    dt = 0.1  # ms
    fs = 1000 / dt  # Hz (since dt is in ms)

    # Bandpass filter
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        return filtfilt(b, a, data)

    # Sampling rate

    lfp_gamma = bandpass_filter(lfp_valid, 30, 80, fs)


    # Compute average amplitude
    analytic_signal = hilbert(lfp_gamma)
    amplitude_envelope = np.abs(analytic_signal)
    average_amplitude = np.mean(amplitude_envelope)
    print(f"Average gamma amplitude for C4 = {c4} mg/dL: {average_amplitude:.4f} mV")


    # Spectogram

   

## IMPORTANT ##
## UPLOAD THE PROPER FILE FORM GOOGLE DRIVE FIRST INTO THE VIRTUAL ENVIRONMENT ##

# Plot LFP Averages 

data_lfp = np.loadtxt("lfp_average_trial105")  ## CHANGE TO FILE NAME ##
print(data_lfp)
print(data_lfp.shape)
plt.figure(figsize=(10,4))
plt.plot(t_vec_valid, data_lfp, color='red', linewidth=2, label='Average LFP')
plt.xlabel('Time (ms)')
plt.ylabel('Average LFP (mV)')
plt.title(f'Average LFP Across 5 Trials for 15 mg/dL of C4 levels') ## CHANGE TITLE FOR THE APPROPRIATE C4 LEVELS ##
plt.legend()
plt.tight_layout()
plt.show()

#Plot Average Fourier Transform of LFP 
power_data = np.loadtxt("power_avg15")  ## CHANGE TO FILE NAME ##
freqs_data = np.loadtxt("freqs_pos_avg105")
plt.figure(figsize=(10,4))
plt.plot(freqs_data, power_data, color='purple')
plt.xlim(0, 150)  # adjust if you want to see a different range
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.title(f'LFP Power Spectrum (Fourier Transform) for 105 mg/dL of C4 levels') ## CHANGE TITLE FOR THE APPROPRIATE C4 LEVELS ##
plt.tight_layout()
plt.show()

# Plot Average Spectrogram of LFP
## CHANGE TO FILE NAME ##


plt.figure(figsize=(15, 10))
f_spec, t_spec, Sxx = spectrogram(data_lfp, fs=((1000/dt)), nperseg=400, noverlap=100) 
plt.subplot(3, 2, 1)
plt.pcolormesh(t_spec * 1000, f_spec, Sxx, shading='gouraud', cmap="jet",vmin=np.min(Sxx), vmax=(np.max(Sxx)/2.5))
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylim(0, 100)
plt.xlim(0, 1000)  # adjust if you want to see a different range
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title(f' Average Spectrogram Trial for 15 mg/dL of C4)') ## CHANGE TITLE FOR THE APPROPRIATE C4 LEVELS ##
plt.suptitle(f'Spectrograms (C4 = 15 mg/dL)', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")
