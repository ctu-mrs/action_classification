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
        [0.0, 0.0, 0.0],  # nose
        [-7.5, 15.0, -7.5],  # left_shoulder
        [7.5, 15.0, -7.5],  # right_shoulder
        [-15.0, 30.0, -15.0],  # left_elbow
        [15.0, 30.0, -7.5],  # right_elbow
        [-20.0, 45.0, -35.0],  # left_wrist
        [20.0, 45.0, -25.5],  # right_wrist
        [-7.5, 60.0, -15.0],  # left_hip
        [7.5, 60.0, -15.0],  # right_hip
        [-7.5, 90.0, -30.5],  # left_knee
        [7.5, 90.0, -30.5],  # right_knee
        [-7.5, 120.0, -22.5],  # left_heel
        [7.5, 120.0, -22.5],  # right_heel
    ]
)

# Step 3: Instantiate FeatureVectorEmbedder
embedder = FeatureVectorEmbedder()

# Step 4: Normalize landmarks
normalized_landmarks = embedder(sample_landmarks, use_orientation_normalization=True)

print(normalized_landmarks)
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

# Line connections for non-normalized landmarks
ax.plot(
    sample_landmarks[[5, 3, 1, 2, 4, 6, 8, 10, 12], 0],
    sample_landmarks[[5, 3, 1, 2, 4, 6, 8, 10, 12], 1],
    sample_landmarks[[5, 3, 1, 2, 4, 6, 8, 10, 12], 2],
    c="blue",
)

# Line connections for normalized landmarks
ax.plot(
    normalized_landmarks[[5, 3, 1, 2, 4, 6, 8, 10, 12], 0],
    normalized_landmarks[[5, 3, 1, 2, 4, 6, 8, 10, 12], 1],
    normalized_landmarks[[5, 3, 1, 2, 4, 6, 8, 10, 12], 2],
    c="red",
)

# Additional line connections
ax.plot(
    sample_landmarks[[1, 2, 0], 0],
    sample_landmarks[[1, 2, 0], 1],
    sample_landmarks[[1, 2, 0], 2],
    c="blue",
)

ax.plot(
    sample_landmarks[[7, 8], 0],
    sample_landmarks[[7, 8], 1],
    sample_landmarks[[7, 8], 2],
    c="blue",
)

# Additional line connections for normalized landmarks
ax.plot(
    normalized_landmarks[[1, 2, 0], 0],
    normalized_landmarks[[1, 2, 0], 1],
    normalized_landmarks[[1, 2, 0], 2],
    c="red",
)

ax.plot(
    normalized_landmarks[[7, 8], 0],
    normalized_landmarks[[7, 8], 1],
    normalized_landmarks[[7, 8], 2],
    c="red",
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
