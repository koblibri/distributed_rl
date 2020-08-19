# import numpy as np
# prob_weights = np.array([0.1,0.1,0.2,0.2,0.4])
# action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)
# print(action)

import time
time_start = time.time()
time_end = time.time()
time_c = time_end - time_start
print('time cost', time_c, 's')
