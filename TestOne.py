import numpy as np
import spiking
#https://neurons.readthedocs.io/en/latest/intro/srm_network.html
model = spiking.SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, eta_reset=5)

weights = np.array([[0, 0, 1.],
                    [0, 0, 1.],
                    [0, 0, 0]])  

spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                       [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

for time in range(10):
    total_potential = model.check_spikes(spiketrain, weights, time)
    
print("Spiketrain:")
print(spiketrain)