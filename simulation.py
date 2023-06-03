import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Sonar parameters
SOURCE_LEVEL = 133.9 # dB
NOISE_LEVEL = 59.7 # dB
ARRAY_GAIN = 11.5 # dB
DETECTION_THRESHOLD = 13.8 # dB
PROBA_FA = 0.05 # Probability of false alarm


def _check_for_detection(vessel_pos, sensor_pos):
    '''Check if the vessel is detected by the sensor

    Parameters
    ----------
    vessel_pos : numpy.ndarray
        Vessel position
    sensor_pos : numpy.ndarray
        Sensor position

    Returns
    -------
    bool : True if the vessel is detected, False otherwise
    '''
    # Compute the transmission loss
    distance = np.linalg.norm(vessel_pos - sensor_pos)
    transmission_loss = 10 * np.log10(distance)

    # Compute the signal-to-noise ratio
    snr = SOURCE_LEVEL - transmission_loss - (NOISE_LEVEL - ARRAY_GAIN)
    snr = 10**(0.1*snr)
    
    if snr > DETECTION_THRESHOLD:
        # Calculate the probability of detection
        proba_d = PROBA_FA**(1 / (1 + snr))
        if np.random.rand() < proba_d:
            return True



def _calculate_bearing(target_pos, sensor_pos):
    '''Calculate the range and bearing of the vessel from the sensor
    
    Parameters
    ----------
    target_pos : numpy.ndarray
        Vessel position
    sensor_pos : numpy.ndarray
        Sensor position
        
    Returns
    -------
    target_bearing_deg : float
        Bearing of the vessel from the sensor in degrees
    '''
    target_bearing_rad = np.arctan2(target_pos[:, 0] - sensor_pos[:, 0], target_pos[:, 1] - sensor_pos[:, 1])
    target_bearing_deg = np.degrees(target_bearing_rad)
    return target_bearing_deg



def calc_sensor_track(n_theta, xrange, yrange):
    '''Calculate the sensor track'''
    center = ((xrange[1] + xrange[0])/2, (yrange[1] + yrange[0])/2)  # Center coordinates (x, y)
    width = (xrange[1] - xrange[0]) / 3  # Width
    height = (yrange[1] - yrange[0]) / 3  # Height
    angle = 0  # Rotation angle in degrees
    # Generate theta values from 0 to 2*pi
    theta = np.linspace(0, 2*np.pi, n_theta)
    # Compute x and y coordinates of the ellipse
    x = center[0] + width/2 * np.cos(theta) * np.cos(np.radians(angle)) - height/2 * np.sin(theta) * np.sin(np.radians(angle))
    y = center[1] + width/2 * np.cos(theta) * np.sin(np.radians(angle)) + height/2 * np.sin(theta) * np.cos(np.radians(angle))
    return np.array([x, y]).T



def run_simulation(mmsi, measurements, tracker, sim_delta):
    '''Run the simulation
    
    Parameters
    ----------
    measurements : pandas.DataFrame
        AIS measurements
    sensor_track : numpy.ndarray
        Sensor track
    tracker : Tracker
        Tracker object, e.g. KalmanFilter, ParticleFilter
    sim_delta : datetime.timedelta
        Simulation time step
    '''
    # Create a log file
    logger.addHandler(logging.FileHandler(f'./logs/{mmsi}_{tracker.abrev}_{tracker.motion_model}.log', mode='w'))

    # Initialise output data for plotting
    out_estimates = np.empty((0, 2)) # Estimated target positions
    out_particles = np.empty((0, tracker.n, 2)) # Particle states
    out_weights = np.empty((0, tracker.n)) # Particle weights
    out_measurements = np.empty((0, 2)) # Measured target positions

    # Initialise simulation clock
    sim_clock = datetime.strptime(measurements[0, 0], '%Y-%m-%d %H:%M:%S.%f')

    # Initialise AIS measurement index 
    ais_idx = 0
    ais_time = datetime.strptime(measurements[ais_idx, 0], '%Y-%m-%d %H:%M:%S.%f')
    ais_time_final = datetime.strptime(measurements[-1, 0], '%Y-%m-%d %H:%M:%S.%f')

    pbar = tqdm(total=len(measurements)-1, desc=f'{mmsi} | Simulation progress', colour='green', ncols=120)

    while True:
        # Break if the simulation clock exceeds the last AIS measurement time
        if sim_clock > ais_time_final:
            break

        # Make a prediction
        tracker.predict()

        # Check if there is an AIS measurement available
        if sim_clock >= ais_time:
            # Get the AIS measurement
            meas_pos = measurements[ais_idx, 1:]

            # Update the tracker
            tracker.update(meas_pos)

            tracker.resample()

            # Increment the AIS index
            ais_idx += 1

            # Next AIS measurement time
            ais_time = datetime.strptime(measurements[ais_idx, 0], '%Y-%m-%d %H:%M:%S.%f')

            pbar.update(1)

        # Estimate the target position
        est = tracker.estimate()
        out_estimates = np.concatenate((out_estimates, est[np.newaxis, :]), axis=0)

        out_particles = np.concatenate((out_particles, tracker.particles[np.newaxis, :, :2]), axis=0)
        out_weights = np.concatenate((out_weights, tracker.weights[np.newaxis, :]), axis=0)

        out_measurements = np.concatenate((out_measurements, meas_pos[np.newaxis, :]), axis=0)

        # Increment the simulation clock
        sim_clock += timedelta(seconds=sim_delta)

    # Calculate the sensor track
    out_sensor_track = calc_sensor_track(len(out_estimates), tracker.xrange, tracker.yrange)

    # Calculate the bearings
    out_bearings = _calculate_bearing(out_sensor_track, out_estimates)

    # Save data
    path = f'./data/{mmsi}_{tracker.abrev}_{tracker.motion_model}'
    np.save(f'{path}_estimates_plot.npy', out_estimates)
    np.save(f'{path}_particles_plot.npy', out_particles)
    np.save(f'{path}_weights_plot.npy', out_weights)
    np.save(f'{path}_measurements_plot.npy', out_measurements)
    np.save(f'{path}_sensor_track_plot.npy', out_sensor_track)
    np.save(f'{path}_bearings_plot.npy', out_bearings)
