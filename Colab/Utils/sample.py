import numpy as np

def sample_log_uniform(low=0.01, high=2.0, size=1):
#     return np.exp(np.random.uniform(np.log(low), np.log(high), size))
    # return np.random.uniform(low, high, size)
    if low <= 0:
        return np.random.uniform(low, high, size)
    else:
        return np.exp(np.random.uniform(np.log(low), np.log(high), size))
