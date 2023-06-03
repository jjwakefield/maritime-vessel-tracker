import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import seaborn as sns
sns.set_theme(style='whitegrid')

def _update(frame, ax, tracker, measurements, estimates, particles, weights, xrange, yrange, pbar):
    # Clear the previous frame
    ax.clear()
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.ticklabel_format(axis='both', style='plain', useOffset=True)
    # Plot the data
    ax.plot(measurements[frame, 0], measurements[frame, 1], 'ro', markersize=10)
    ax.plot(measurements[:frame, 0], measurements[:frame, 1], 'r--', label='AIS Measurements')
    ax.plot(estimates[frame, 0], estimates[frame, 1], 'k*', markersize=10)
    ax.plot(estimates[:frame, 0], estimates[:frame, 1], 'k--', label='Estimated track')
    ax.scatter(particles[frame, :, 0], particles[frame, :, 1], marker='.', c='g', alpha=0.5, s=weights[frame]*10000, label='Particles')
    # Set the axis limits
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    # Set the plot title
    ax.set_title('Frame {}'.format(frame))
    # Add labels
    ax.set_xlabel('X UTM (m)')
    ax.set_ylabel('Y UTM (m)')
    # Add a legend & title
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.27), ncol=2)
    ax.set_title(f'{tracker.name}', fontsize=14)

    plt.subplots_adjust(bottom=0.2)

    # Update the progress bar
    pbar.update(1) 


def generate_animation(mmsi, tracker, xrange, yrange, fps):
    data_path = f'./data/{mmsi}_{tracker.abrev}_{tracker.motion_model}'
    anim_path = './animations'

    # Load the data
    measurements = np.load(f'{data_path}_measurements_plot.npy', allow_pickle=True)
    estimates = np.load(f'{data_path}_estimates_plot.npy', allow_pickle=True)
    particles = np.load(f'{data_path}_particles_plot.npy', allow_pickle=True)
    weights = np.load(f'{data_path}_weights_plot.npy', allow_pickle=True)

    total_frames = estimates.shape[0]

    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    fig.canvas.manager.set_window_title(f'Vessel Tracker | MMSI: {mmsi}')

    # Set the plot title
    fig.suptitle(f'Vessel Tracker | MMSI: {mmsi} | Tracker: {tracker.name} | Motion Model: {tracker.motion_model.upper()}', fontsize=16)

    # Set the number of frames per second in the animation
    fps = 5

    # Create the progress bar using tqdm
    pbar = tqdm(total=total_frames, desc=f'{mmsi} |  Animation progress', colour='blue', ncols=120)

    # Create the animation
    animation = FuncAnimation(fig, lambda i: _update(i, ax, tracker, measurements, estimates, particles, weights, xrange, yrange, pbar), frames=total_frames, interval=1000/fps, blit=False, repeat=False)

    # Save the animation
    animation.save(f'{anim_path}/{mmsi}_{tracker.abrev}_{tracker.motion_model}.mp4', writer='ffmpeg', fps=fps)

    pbar.n = total_frames
    pbar.refresh()
