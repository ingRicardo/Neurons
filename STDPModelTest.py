import numpy as np
import learning

stdp_model = learning.STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=5)

weights = np.array([[0, 0, 1.],
                    [0, 0, 1.],
                    [0, 0, 0]])

spiketrain = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                       [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=bool)

for time in range(10):
    stdp_model.weight_change(spiketrain, weights, time)
    
print("Weights after")
print(weights)