import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import this for 3D plotting
import os
from matplotlib.animation import FuncAnimation

# Get the mat path
mat_path = os.path.expanduser(
    "~/git/action_classification/utd_mhad_dataset/a21_pickup_throw/a21_s1_t1_skeleton.mat"
)

# Load the .mat file
data = scipy.io.loadmat(mat_path)
skeleton_poses = data["d_skel"]  # Assuming the variable name is skeleton_poses

# Create a figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Set up the plot
(line,) = ax.plot([], [], [], "bo-")

# Set plot limits and labels
ax.set_xlim3d([-1, 1])  # Adjust these limits based on your data
ax.set_ylim3d([-1, 1])
ax.set_zlim3d([-1, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


# Function to initialize the plot
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return (line,)


# Function to animate the plot
def animate(frame):
    pose = skeleton_poses[:, :, frame]
    line.set_data(pose[:, 0], pose[:, 1])
    line.set_3d_properties(pose[:, 2])
    ax.set_title("Frame {}".format(frame))
    return (line,)


# Create the animation
num_frames = skeleton_poses.shape[2]
ani = FuncAnimation(
    fig, animate, frames=num_frames, init_func=init, blit=True, interval=100
)

plt.show()
