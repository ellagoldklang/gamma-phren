from neuron import h
from neuron.units import ms, mV
h.load_file('stdrun.hoc')
import random
import numpy as np
import math
from fractions import Fraction
import matplotlib.pyplot as plt

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


        self.dend = h.Section(name='dend') # dendritic section 
        self.dend.L = 500 # length in micrometers
        self.dend.diam = 1 # diameter in micrometers
        self.dend.nseg = 21

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


        self.dend = h.Section(name='dend_inhibit') # dendritic section 
        self.dend.L = 300 # length in micrometers
        self.dend.diam = 1 # diameter in micrometers
        self.dend.nseg = 5

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

n = 500 # number of pairs
c4 = 45  # concentration of c4 protein in mg/dL


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


# #LFP section

# 1. Record synaptic currents (this part is good)
t_vec = h.Vector().record(h._ref_t)
v_e = [h.Vector().record(cell.soma(0.5)._ref_v) for cell in e_cell]
v_i = [h.Vector().record(cell.soma(0.5)._ref_v) for cell in i_cell]


# 2. Run the simulation
h.finitialize(-65)
h.continuerun(time)

# Convert recorded vectors to numpy arrays for easier handling
v_e_mat = np.array([np.array(vec) for vec in v_e])
v_i_mat = np.array([np.array(vec) for vec in v_i])

# LFP = mean across all excitatory (and optionally, inhibitory) neurons at each time point
# This is the "network LFP"
v_e_mat = np.array([np.array(vec) for vec in v_e])
lfp = np.mean(v_e_mat, axis=0)

# Plot LFP
plt.figure(figsize=(10,4))
plt.plot(t_vec, lfp, color='black', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Simulated LFP (mV)')
plt.title('LFP (Mean of Excitatory Soma Voltages)')
plt.tight_layout()
plt.show()


# # --- FOURIER TRANSFORM OF THE LFP (after 50 ms) ---

# # 1. Remove any DC offset (recommended)
# lfp_valid_detrended = lfp_valid - np.mean(lfp_valid)

# # 2. Compute the FFT
# lfp_fft = np.fft.fft(lfp_valid_detrended)
# freqs = np.fft.fftfreq(len(lfp_valid), d=dt/1000)  # dt in ms, so divide by 1000 for seconds

# # 3. Only use positive frequencies (one-sided spectrum)
# pos_mask = freqs > 0
# freqs_pos = freqs[pos_mask]
# lfp_fft_pos = np.abs(lfp_fft[pos_mask])

# # 4. Plot the power spectrum
# plt.figure(figsize=(10,4))
# plt.plot(freqs_pos, lfp_fft_pos, color='purple')
# plt.ylim(0, 100)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude (a.u.)')
# plt.title('LFP Power Spectrum (Fourier Transform, after 50 ms)')
# plt.tight_layout()
# plt.show()




#testing code section




# # 5. Record voltages and time

# h.finitialize(-65)
# h.continuerun(time)



# 6. Run simulation (e.g., 200 ms)

# 7. Plot both neuron voltages

# plt.figure(figsize=(10,5))
# for i in range(n):
#     plt.plot(t_vec, v_e[i], alpha=0.6, label=f"E{i}")
#     plt.plot(t_vec, v_i[i], alpha=0.6, label=f"I{i}")
# plt.xlabel("Time (ms)")
# plt.ylabel("Membrane Potential (mV)")
# plt.title("PING Test: E→I and I→E Connections")
# plt.legend()
# plt.tight_layout()
# plt.show()
