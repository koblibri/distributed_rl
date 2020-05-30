import numpy as np
prob_weights = np.array([0.1,0.1,0.2,0.2,0.4])
action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
print(action)
