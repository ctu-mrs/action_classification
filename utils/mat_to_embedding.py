import os
import scipy.io
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../scripts/"))
from feature_vector_generator import FeatureVectorEmbedder

mat_path = os.path.expanduser("~/git/action_classification/utd_mhad_dataset/")
embedding_path = os.path.expanduser(
    "~/git/action_classification/utd_mhad_dataset/embeddings/"
)
embedder = FeatureVectorEmbedder()
for root, directories, files in os.walk(mat_path):
    for file in files:
        if file.endswith(".mat"):
            mat_file_path = os.path.join(root, file)
            data = scipy.io.loadmat(mat_file_path)
            skeleton_poses = data["d_skel"]
            indices_to_select = [0, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]
            skeleton_poses = skeleton_poses[indices_to_select, :, :]
            embedding_poses = []
            for i in range(skeleton_poses.shape[2]):
                embedding_singular_pose = embedder(
                    skeleton_poses[:, :, i], (float(i + 1) / 30.0)
                )  # 30 fps)
                embedding_poses.append(embedding_singular_pose)

            # Get the parent directory of the mat file
            parent_dir = os.path.dirname(mat_file_path)

            # Get the name of the mat file without the extension
            file_name = os.path.basename(file).split(".")[0]

            # Create the output directory if it doesn't exist
            if not os.path.exists(embedding_path):
                os.mkdir(embedding_path)

            # Create the output parent directory
            new_parent_dir = os.path.join(embedding_path, parent_dir)
            if not os.path.exists(new_parent_dir):
                os.mkdir(new_parent_dir)

            # Save the mat file to the output directory
            new_mat_file_path = os.path.join(new_parent_dir, file_name + ".mat")
            scipy.io.savemat(new_mat_file_path, {"embedding": embedding_poses})
