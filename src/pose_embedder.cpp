#include <pose_embedder.hpp>


namespace knn_action_classifier
{
    FullBodyPoseEmbedder::FullBodyPoseEmbedder(float torso_size_multiplier)
    {
        torso_size_multiplier_var = torso_size_multiplier;
        landmark_names =  {"nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky_1", "right_pinky_1",
        "left_index_1", "right_index_1",
        "left_thumb_2", "right_thumb_2",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    };
    }

    Eigen::Matrix<double, 23, 3> 
            FullBodyPoseEmbedder::call(Eigen::Matrix<double, 33, 3> landmarks)
    {
        Eigen::Matrix<double, 23, 3> embedding;
        Eigen::Matrix<double, 33, 3> landmark_fvar;

        landmark_fvar = landmarks;
        landmark_fvar = normalize_pose_landmarks(landmark_fvar);
        embedding = get_pose_distance_embedding(landmark_fvar);
        return embedding;

    }            

    Eigen::Matrix<double, 33, 3> 
                    FullBodyPoseEmbedder::normalize_pose_landmarks
                                        (Eigen::Matrix<double, 33, 3> landmarks)
                    {
                        
                        Eigen::Matrix<double, 33, 3> landmark_fvar;
                        Eigen::Matrix<double, 1, 3> pose_center;
                        landmark_fvar = landmarks;
                        pose_center = get_pose_center(landmark_fvar);
                        double pose_size;

                        // Landmark[row] - pose_center
                        for (int i=0; i<33; i++)
                        {
                            for(int j=0; j<3; j++)
                            {
                                landmark_fvar(i,j) -= pose_center(1,j);
                            }
                        }

                        pose_size = get_pose_size(landmark_fvar);
                        landmark_fvar /= pose_size;
                        landmark_fvar *= landmark_fvar;



                    } 
    

}

    int main()
    {
        knn_action_classifier::FullBodyPoseEmbedder embedderobj(2.5);
        return 0;
    }