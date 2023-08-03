import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Replace 'path/to/your/csv/file.csv' with the actual file path
csv_file_path = (
    "~/git/action_classification/utd_mhad_dataset/csvs/a1_s1_t1_skeleton.csv"
)


# Load CSV data into a pandas DataFrame
df = pd.read_csv(csv_file_path, header=None)

# Convert DataFrame to a NumPy array
motion_data = df.values

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Loop through each frame and plot the joints
for frame in range(motion_data.shape[2]):
    joints_data = motion_data[:, :, frame]
    ax.scatter(joints_data[:, 0], joints_data[:, 1], joints_data[:, 2])

# Set axis labels
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# Show the plot
plt.show()
