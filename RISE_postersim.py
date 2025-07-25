from neuron import h
from neuron.units import ms, mV
h.load_file('stdrun.hoc')

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
    # you probably have to edit the biophysics values soon, they look way to off - check later !!

# Instantiate both neurons
e_cell = Excitatory_neuron()
i_cell = inhibitory_neuron()

# connection from excitatory to inhibiton
e_to_i = h.NetCon(e_cell.soma(0.5)._ref_v, i_cell.syn, sec=e_cell.soma)
e_to_i.threshold = 0 # in mV, the threshold for the connection to trigger 
e_to_i.delay = 5    # in ms
e_to_i.weight[0] = 0.25  # in μS, the amount of current given to inhibition # make sure to convert this to a for loop if you decide to add multiple neurons 
e_cell.syn.tau = 3 * ms  


# connection from inhibition to excitatory
e_cell.inh_syn = h.ExpSyn(e_cell.soma(0.5))
e_cell.inh_syn.tau = 9 * ms
e_cell.inh_syn.e = -75      # inhibitory reversal potential in mV


i_to_e = h.NetCon(i_cell.soma(0.5)._ref_v, e_cell.inh_syn, sec=i_cell.soma)
i_to_e.threshold = 0        # mV, spike detection threshold for inhibitory neuron
i_to_e.delay = 5            # ms, can adjust as needed
i_to_e.weight[0] = 0.25     # μS, synaptic strength (experiment with values 0.01 - 0.1)



#testing code section


import matplotlib.pyplot as plt
import numpy as np

# 4. Inject current into excitatory neuron only
e_stim = h.IClamp(e_cell.soma(0.5))
e_stim.delay = 10         # ms
e_stim.dur = 100         # ms (long enough for a few spikes)
e_stim.amp = 1.5         # nA (adjust for repetitive spiking)

# No current to inhibitory neuron

# 5. Record voltages and time
t_vec = h.Vector().record(h._ref_t)
v_e = h.Vector().record(e_cell.soma(0.5)._ref_v)
v_i = h.Vector().record(i_cell.soma(0.5)._ref_v)

# 6. Run simulation (e.g., 200 ms)
h.finitialize(-65)
h.continuerun(100)

# 7. Plot both neuron voltages
plt.figure(figsize=(10,5))
plt.plot(t_vec, v_e, label="Excitatory neuron (soma)", color='C0')
plt.plot(t_vec, v_i, label="Inhibitory neuron (soma)", color='C1')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("PING Test: E→I and I→E Connections")
plt.legend()
plt.tight_layout()
plt.show()