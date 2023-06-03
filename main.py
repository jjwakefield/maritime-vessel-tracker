import numpy as np
import pandas as pd
from particle_filter import ParticleFilter
from simulation import run_simulation
from animation import generate_animation



if __name__ == '__main__':
    # Load the data
    path = './data'
    df = pd.read_csv(f'{path}/SolentAIS_complete.csv')
    all_mmsi = df['MMSI'].unique()

    offset = 1000 # Offset for plotting

    for mmsi in all_mmsi[:4]:
        measurements = df[df['MMSI'] == mmsi][['Time', 'X_utm', 'Y_utm']].to_numpy()
        np.save(f'./beta_data/{mmsi}_measurements.npy', measurements)
        
        # Determine surveillance area
        xrange = (measurements[:, 1].min()-offset, measurements[:, 1].max()+offset)
        yrange = (measurements[:, 2].min()-offset, measurements[:, 2].max()+offset)

        # Tracker parameters
        std = 1
        tracker_dt = 1
        motion_model = 'iou' # 'cv' or 'iou'
        damping = 1e-5
        n_particles = 256
        init_state=measurements[0, 1:]

        # Simulation parameters
        sim_delta = 1 # Simulation time step in seconds

        # Initialise particle filter
        tracker = ParticleFilter(xrange, yrange, std, tracker_dt, motion_model, damping, n_particles)

        # Run simulation
        run_simulation(mmsi, measurements, tracker, sim_delta)

        generate_animation(mmsi, tracker, xrange, yrange, fps=5)

        print(f'Finished {mmsi}')


