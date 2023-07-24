import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import datetime

scriptpath = "../scripts"
sys.path.append(os.path.abspath(scriptpath))
from feature_vector_generator import FeatureVectorEmbedder

# Step 2: Generate sample landmarks
sample_landmarks = (
    np.array(
        [
            [0.0, 1.7, 0.0],  # Nose
            [-0.25, 1.5, 0.0],  # Left Shoulder
            [0.75, 1.5, 0.0],  # Right Shoulder
            [-0.9, 1.2, 0.05],  # Left Elbow
            [0.9, 1.2, 0.0],  # Right Elbow
            [-1.1, 0.9, 0.1],  # Left Wrist
            [1.1, 0.9, 0.0],  # Right Wrist
            [-0.25, 0.0, 0.0],  # Left Hip
            [0.25, 0.1, 0.01],  # Right Hip
            [-0.25, -0.5, 0.0],  # Left Knee
            [0.25, -0.5, 0.0],  # Right Knee
            [-0.25, -1.0, 0.0],  # Left Heel
            [0.25, -1.0, 0.0],  # Right Heel
        ]
    )
    * 100
)


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
# Step 3: Instantiate FeatureVectorEmbedder
embedder = FeatureVectorEmbedder()
time = datetime.datetime.now().timestamp()
# Step 4: Normalize landmarks
normalized_landmarks = embedder(
    sample_landmarks,
    time_stamp=time,
    use_orientation_normalization=True,
)

print(normalized_landmarks)

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
