import numpy as np
from numpy.random import random



def resample_from_index(particles, weights, indices):
    particles[:] = particles[indices]
    weights.resize(len(particles))
    weights.fill(1.0/len(weights))
    particles = particles + (random(particles.shape) - 0.5)
    return particles, weights



def systematic_resampling(particles, weights):
    n = len(weights)
    positions = (random() + np.arange(n)) / n
    indices = np.zeros(n, 'i')
    cumulative_sum = np.cumsum(weights)

    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1

    return resample_from_index(particles, weights, indices)
