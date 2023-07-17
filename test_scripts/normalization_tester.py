import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

scriptpath = "../scripts"
sys.path.append(os.path.abspath(scriptpath))
from feature_vector_generator import FeatureVectorEmbedder

# Step 2: Generate sample landmarks
sample_landmarks = np.array(
    [
        [0, 0, 0],  # nose
        [-30, 60, 0],  # left_shoulder
        [30, 60, 0],  # right_shoulder
        [-60, 120, 0],  # left_elbow
        [60, 120, 0],  # right_elbow
        [-80, 180, 0],  # left_wrist
        [80, 180, 0],  # right_wrist
        [-30, 240, 0],  # left_hip
        [30, 240, 0],  # right_hip
        [-30, 360, 0],  # left_knee
        [30, 360, 0],  # right_knee
        [-30, 480, 0],  # left_heel
        [30, 480, 0],  # right_heel
    ]
)

# Step 3: Instantiate FeatureVectorEmbedder
embedder = FeatureVectorEmbedder()

# Step 4: Normalize landmarks
normalized_landmarks = embedder(sample_landmarks, use_orientation_normalization=True)


# Step 5: Visualize the landmarks
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot of non-normalized landmarks
ax.scatter(
    sample_landmarks[:, 0],
    sample_landmarks[:, 1],
    sample_landmarks[:, 2],
    c="blue",
    label="Non-normalized",
)
# Scatter plot of normalized landmarks
ax.scatter(
    normalized_landmarks[:, 0],
    normalized_landmarks[:, 1],
    normalized_landmarks[:, 2],
    c="red",
    label="Normalized",
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
