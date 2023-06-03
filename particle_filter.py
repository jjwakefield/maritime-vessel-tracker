import numpy as np
from numpy.random import uniform, multivariate_normal
from numpy.linalg import inv
from distance_metrics import mahalanobis
from resampling import systematic_resampling

class ParticleFilter:
    
    def __init__(self, xrange, yrange, std, dt, motion_model, damping, n_particles, seed=42):
        '''Initialize the particle filter
        
        Parameters
        ----------
        x_range : tuple
            Range of the x-axis
        y_range : tuple
            Range of the y-axis
        std : float
            Standard deviation of the noise
        n_particles : int
            Number of particles
        seed : int
            Seed for the random number generator
        '''
        np.random.seed(seed)

        self.n = n_particles
        self.std = std
        self.dt = dt
        self.motion_model = motion_model
        self.name = 'Particle Filter'
        self.abrev = 'pf'
        self.motion_model = motion_model
        self.xrange = xrange
        self.yrange = yrange

        self.particles = np.zeros((self.n, 4), dtype='float64')
        self.particles[:, :2] = uniform(low=[xrange[0], yrange[0]], high=[xrange[1], yrange[1]], size=(self.n, 2))
        self.particles[:, 2:] = multivariate_normal(np.zeros(2), np.diag([std, std]))

        self.weights = np.ones(self.n) / self.n

        if motion_model == 'cv':
            self.F = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype='float64')

            self.Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                               [0, (dt**4)/4, 0, (dt**3)/2],
                               [(dt**3)/2, 0, dt**2, 0],
                               [0, (dt**3)/2, 0, dt**2]], dtype='float64') * self.std**2
        elif motion_model == 'iou':
                    # Evolution/transition matrix
                    F1 = 1/damping * (1 - np.exp(-damping * dt))
                    F2 = np.exp(-damping * dt)
                    self.F = np.array([[1, 0, F1, 0],
                                       [0, 1, 0, F2],
                                       [0, 0, F2, 0],
                                       [0, 0, 0, F2]], dtype='float64')
                    
                    # Noise covariance matrix
                    Q1 = (dt - 2/damping * (1 - np.exp(-damping * dt)) + 1/(2*damping) * (1 - np.exp(-2*damping * dt))) / damping**2
                    Q2 = (1/damping * (1 - np.exp(-damping * dt)) - 1/(2*damping) * (1 - np.exp(-2*damping * dt))) / damping
                    Q3 = (1 - np.exp(-2*damping * dt)) / (2*damping)
                    self.Q = np.array([[Q1, 0, Q2, 0],
                                      [0, Q1, 0, Q2],
                                      [Q2, 0, Q3, 0],
                                      [0, Q2, 0, Q3]], dtype='float64') * self.std**2
        else:
            raise ValueError('Invalid motion model')
        

    def predict(self):
        '''Predict the next position of the vessel using the constant velocity model'''
        for i in range(self.n):
            noise = multivariate_normal(np.zeros(4), self.Q)
            self.particles[i] = self.F @ self.particles[i] + noise


    def update(self, measurement): 
        '''Update the particles based on the measurement
        
        Parameters
        ----------
        measurement : array_like
            Measurement of the vessel's position (x, y)
        '''
        cov_inv = inv(np.cov(self.particles[:, :2].T))
        for i in range(self.n):
            mahal = mahalanobis(self.particles[i, :2], measurement, cov_inv)
            self.weights[i] = np.exp(-0.5 * mahal) * self.weights[i]
        self.weights /= np.sum(self.weights)


    def resample(self):
        '''Resample the particles
        
        Returns
        -------
        bool : True if resampling was performed, False otherwise
        '''
        N_eff = 1 / np.sum(self.weights**2)
        if N_eff < self.n / 2:
            self.particles[:, :2], self.weights = systematic_resampling(self.particles[:, :2], self.weights)
            return True


    def estimate(self):
        '''Estimate the current position of the vessel
        
        Returns
        -------
        est : array_like
            Estimated position based on the weighted average of the particles (x, y)
        '''
        self.est = np.average(self.particles[:, :2], weights=self.weights, axis=0)
        return self.est
    