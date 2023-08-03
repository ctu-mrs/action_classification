import os
import numpy as np
import scipy.io as sio
import csv

# Get the mat_path and csv_path
mat_path = os.path.expanduser("~/git/action_classification/utd_mhad_dataset/")
csv_path = os.path.expanduser("~/git/action_classification/utd_mhad_dataset/csvs/")
# Incoming Dataset structure
# 1. head;
# 2. shoulder_center;
# 3. spine;
# 4. hip_center;
# 5. left_shoulder;
# 6. left_elbow;
# 7. left_wrist;
# 8. left_hand;
# 9. right_shoulder;
# 10. right_elbow;
# 11. right_wrist;
# 12. right_hand;
# 13. left_hip;
# 14. left_knee;
# 15. left_ankle;
# 16. left_foot;
# 17. right_hip;
# 18. right_knee;
# 19. right_ankle;
# 20. right_foot;

# Outgoing Dataset structure

# "nose",
# "left_shoulder",
# "right_shoulder",
# "left_elbow",
# "right_elbow",
# "left_wrist",
# "right_wrist",
# "left_hip",
# "right_hip",
# "left_knee",
# "right_knee",
# "left_heel",
# "right_heel",
# Iterate over the directories
for directory in os.listdir(mat_path):
    # Get the full path to the directory
    directory_path = os.path.join(mat_path, directory)

    # Check if the directory is a subdirectory
    if os.path.isdir(directory_path):
        # Iterate over the files in the directory
        for file in os.listdir(directory_path):
            # Check if the file is a .mat file
            if file.endswith(".mat"):
                # Get the full path to the file
                file_path = os.path.join(directory_path, file)

                # Load the .mat file
                data = sio.loadmat(file_path)

                # Get the matrix from the .mat file
                matrix = data["d_skel"]

                matrix_to_be_saved = matrix[
                    [0, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18], :, :
                ]

                # Save the matrix as a CSV file
                csv_file_path = os.path.join(csv_path, file.replace(".mat", ".csv"))
                with open(csv_file_path, "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(matrix_to_be_saved)

                # Print the file path
                print(csv_file_path)
