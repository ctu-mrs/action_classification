import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentdir, "../scripts/"))
from knn_ball_tree import main


num_of_clusters_range = [*range(3, 27, 3)]
number_of_candidates_range = [1, 2, 3]
number_of_neighbors_range = [*range(3, 20, 3)]
leaf_size_range = [20, 40, 60]

# This will store the best parameter combination and its accuracy
best_accuracy = 0
best_parameters = {}
n_iter = 1

# Iterate over all combinations of parameters
for (
    num_of_clusters,
    number_of_candidates,
    number_of_neighbors,
    leaf_size,
) in itertools.product(
    num_of_clusters_range,
    number_of_candidates_range,
    number_of_neighbors_range,
    leaf_size_range,
):
    print(f"Iteration: {n_iter}")
    n_iter += 1
    try:
        print(
            f"Testing combination: clusters={num_of_clusters}, candidates={number_of_candidates}, neighbors={number_of_neighbors}, leaf_size={leaf_size}"
        )
        accuracy = main(
            num_of_clusters=num_of_clusters,
            number_of_candidates=number_of_candidates,
            number_of_neighbors=number_of_neighbors,
            leaf_size=leaf_size,
        )
        print(f"Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters = {
                "num_of_clusters": num_of_clusters,
                "number_of_candidates": number_of_candidates,
                "number_of_neighbors": number_of_neighbors,
                "leaf_size": leaf_size,
            }
    except Exception as e:
        print(f"Error with combination: {e}")

print(f"Best accuracy: {best_accuracy}")
print(f"Best parameters: {best_parameters}")
