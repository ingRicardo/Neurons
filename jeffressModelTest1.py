import numpy as np
import matplotlib.pyplot as plt

import spiking
import plotting

neurons = 11
timesteps = 100

ax_delays = np.array([0, 5, 15, 25, 0, 25, 15, 5, 0, 0, 0])

threshold = np.array([1]*neurons)
t_current = np.array([5]*neurons)
t_membrane = np.array([10]*neurons)
eta_reset = np.array([2]*neurons)

model = spiking.SRM_X(neurons=neurons, threshold=threshold, t_current=t_current,
                      t_membrane=t_membrane, eta_reset=eta_reset, ax_delay=ax_delays)

weights = np.zeros((neurons, neurons))

# Connect input layer
weights[0, (1, 2, 3)] = 1
weights[4, (5, 6, 7)] = 1

# Connect to output layer
weights[(1, 5), 8] = 1.1
weights[(2, 6), 9] = 1.1
weights[(3, 7), 10] = 1.1

spiketrain = np.zeros((neurons, timesteps), dtype=bool)

spiketrain[0, (0, 5, 10)] = 1
spiketrain[4, (20, 25, 30)] = 1


for t in range(timesteps):
    model.check_spikes(spiketrain, weights, t)
    
psth = plotting.PSTH(spiketrain, binsize=5)
psth.show_plot(neuron_indices=[8, 9, 10])
plt.show()