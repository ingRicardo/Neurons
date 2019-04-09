import numpy as np
import spiking
import learning
 
srm_model = spiking.SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, eta_reset=5)

stdp_model = learning.STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=5)

weights = np.array([[0, 0, 1.], [0, 0, 1.], [0, 0, 0]])

spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                       [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

for time in range(10):
    srm_model.check_spikes(spiketrain, weights, time)
    stdp_model.weight_change(spiketrain, weights, time)
    
print("Final Spiketrain:")
print(spiketrain)

print("Final weights:")
print(weights)