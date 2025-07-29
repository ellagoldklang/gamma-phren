from neuron import h
from neuron.units import ms, mV
h.load_file('stdrun.hoc')
import random
import numpy as np

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
    # add external drives ? 

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

number = 2  # number of pairs
c4 = 5  # concentration of c4 protein
n = 

# simulating the amount of pairs (n)

# Instantiate both neurons
e_cell = [Excitatory_neuron() for _ in range(n)]
i_cell = [inhibitory_neuron() for _ in range(n)]

# connection from excitatory to inhibiton
e_to_i = []
i_to_e = []

for i in range(n):
    # excitatory to inhibitory
    nc_ei = h.NetCon(e_cell[i].soma(0.5)._ref_v, i_cell[i].syn, sec=e_cell[i].soma)
    nc_ei.threshold = 0 # in mV, the threshold for the connection to trigger 
    nc_ei.delay = random.normalvariate(2, 5)    # in ms
    nc_ei.weight[0] = 0.25  # in μS, the amount of current given to inhibition # make sure to convert this to a for loop if you decide to add multiple neurons 
    e_cell[i].syn.tau = 3 * ms  
    e_to_i.append(nc_ei)

    # inhibitory to excitatory
    e_cell[i].inh_syn = h.ExpSyn(e_cell[i].soma(0.5))
    e_cell[i].inh_syn.tau = 9 * ms
    e_cell[i].inh_syn.e = -75      # inhibitory reversal potential in mV
    nc_ie = h.NetCon(i_cell[i].soma(0.5)._ref_v, e_cell[i].inh_syn, sec=i_cell[i].soma)
    nc_ie.threshold = 0        # mV, spike detection threshold for inhibitory neuron
    nc_ie.delay = random.normalvariate(7, 12)            # ms, can adjust as needed
    nc_ie.weight[0] = 0.25     # μS, synaptic strength (experiment with values 0.01 - 0.1)
    i_to_e.append(nc_ie)

# adding noise into the neuron
# comments about noise: 1] only 1 neuron gets the noise, 2] the excitatory neuron does not recieve the noise 

dt = 0.1 # in ms
time = 100 # in ms
nt = int(time / dt)
t_points = np.arange(0, time, dt)

baseline = 0.5 # in nA
noise_strength = 0.1 # in nA
e_noise = []
i_noise = []



for i in range(n):
    # noise for excitatory cell
    stim = h.IClamp(e_cell[i].soma(0.5))
    stim.delay = 0
    stim.dur = time  # Always on
    noise_and_baseline = 0.5 + np.random.normal(0, noise_strength, nt)
    noise_vec = h.Vector(noise_and_baseline)
    noise_vec.play(stim._ref_amp, dt)
    e_noise.append(stim)


#testing code section


import matplotlib.pyplot as plt



# 5. Record voltages and time
t_vec = h.Vector().record(h._ref_t)
v_e = [h.Vector().record(cell.soma(0.5)._ref_v) for cell in e_cell]
v_i = [h.Vector().record(cell.soma(0.5)._ref_v) for cell in i_cell]


# 6. Run simulation (e.g., 200 ms)
h.finitialize(-65)
h.continuerun(time)

# 7. Plot both neuron voltages
plt.figure(figsize=(10,5))
for i in range(n):
    plt.plot(t_vec, v_e[i], alpha=0.6, label=f"E{i}")
    plt.plot(t_vec, v_i[i], alpha=0.6, label=f"I{i}")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("PING Test: E→I and I→E Connections")
plt.legend()
plt.tight_layout()
plt.show()
