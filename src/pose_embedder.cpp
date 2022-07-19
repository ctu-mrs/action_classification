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
        Eigen::Matrix<double, 33, 3> landmark_fvar_call;

        landmark_fvar_call = landmarks;
        landmark_fvar_call = normalize_pose_landmarks(landmark_fvar_call);
        embedding = get_pose_distance_embedding(landmark_fvar_call);
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

        pose_size = get_pose_size(landmark_fvar, torso_size_multiplier_var);
        landmark_fvar /= pose_size;
        landmark_fvar *= 100;
        return landmark_fvar;



    }

    Eigen::Matrix<double, 1, 3> FullBodyPoseEmbedder::get_pose_center
                                    (Eigen::Matrix<double, 33, 3> landmarks)
    {
        Eigen::Matrix<double, 1, 3> left_hip, right_hip, hip_center;
        left_hip = landmarks.row(getIndex(landmark_names, "left_hip"));

        right_hip = landmarks.row(getIndex(landmark_names, "right_hip"));

        hip_center = (left_hip + right_hip) * 0.5;

        return hip_center;

    }

    double FullBodyPoseEmbedder::get_pose_size(Eigen::Matrix<double, 33, 3> landmarks,
                                         double torso_size_multiplier)
    {
        Eigen::Matrix<double, 1, 2> landmarks2D;
        landmarks2D << landmarks(0,0), landmarks(0,1);
        double torso_size, max_distance;
        Eigen::Matrix<double, 1, 2> pose_center_for_size, hip_center_size, 
                                            shoulder_center_size;
        Eigen::Matrix<double, 1, 3> pose_center, left_hip, right_hip, 
                                        left_shoulder, right_shoulder;
        hip_center_size(0,0) = get_pose_center(landmarks)(0,0);
        hip_center_size(0,1) = get_pose_center(landmarks)(0,1);

        left_shoulder = landmarks.row(
                                getIndex(landmark_names, "left_shoulder"));
        right_shoulder = landmarks.row(
                                getIndex(landmark_names, "right_shoulder"));

        shoulder_center_size << ((left_shoulder + right_shoulder) * 0.5)(0,0), 
                                ((left_shoulder + right_shoulder) * 0.5)(0,1);
        torso_size = (hip_center_size-shoulder_center_size).norm();
        max_distance = ((landmarks2D.rowwise() - hip_center_size)
                                                .rowwise().norm()).maxCoeff();
        return std::max(torso_size* torso_size_multiplier_var, max_distance);
    }


    Eigen::Matrix<double, 23, 3> FullBodyPoseEmbedder::
                        get_pose_distance_embedding(Eigen::Matrix<double, 33, 3>
                        landmarks)
    {
        Eigen::Matrix<double, 23, 3> embedding;
        embedding.row(0) << get_distance(
            get_average_by_names(landmarks, "left_hip", "right_hip"), 
            get_average_by_names(landmarks, "left_shoulder", "right_shoulder")
                                        );

        embedding.row(1) << 
            get_dist_by_names(landmarks, "left_shoulder", "left_elbow");
        embedding.row(2) << 
            get_dist_by_names(landmarks, "right_shoulder", "right_elbow");

        embedding.row(3) << 
            get_dist_by_names(landmarks, "left_elbow", "left_wrist");
        embedding.row(4) << 
            get_dist_by_names(landmarks, "right_elbow", "right_wrist");

        embedding.row(5) << 
            get_dist_by_names(landmarks, "left_hip", "left_knee");
        embedding.row(6) << 
            get_dist_by_names(landmarks, "right_hip", "right_knee");

        embedding.row(7) << 
            get_dist_by_names(landmarks, "left_knee", "left_ankle");
        embedding.row(8) << 
            get_dist_by_names(landmarks, "right_knee", "right_ankle");



        
        embedding.row(9) << 
            get_dist_by_names(landmarks, "left_shoulder", "left_wrist");
        embedding.row(10) << 
            get_dist_by_names(landmarks, "right_shoulder", "right_wrist");

        embedding.row(11) << 
            get_dist_by_names(landmarks, "left_hip", "left_ankle");
        embedding.row(12) << 
            get_dist_by_names(landmarks, "right_hip", "right_ankle");



            
        embedding.row(13) << 
            get_dist_by_names(landmarks, "left_hip", "left_wrist");
        embedding.row(14) << 
            get_dist_by_names(landmarks, "right_hip", "right_wrist");




        embedding.row(15) << 
            get_dist_by_names(landmarks, "left_shoulder", "left_ankle");
        embedding.row(16) << 
            get_dist_by_names(landmarks, "right_shoulder", "right_ankle");

        embedding.row(17) << 
            get_dist_by_names(landmarks, "left_hip", "left_wrist");
        embedding.row(18) << 
            get_dist_by_names(landmarks, "right_hip", "right_wrist");




        embedding.row(19) << 
            get_dist_by_names(landmarks, "left_elbow", "right_elbow");
        embedding.row(20) << 
            get_dist_by_names(landmarks, "left_knee", "right_knee");


        embedding.row(21) << 
            get_dist_by_names(landmarks, "left_wrist", "right_wrist");
        embedding.row(22) << 
            get_dist_by_names(landmarks, "left_ankle", "right_ankle");
        

        return embedding;
    }

    
    Eigen::Matrix<double, 1, 3> FullBodyPoseEmbedder::
                        get_average_by_names(Eigen::Matrix<double, 33, 3>
                        landmarks, std::string name_from, std::string name_to)
    {

        Eigen::Matrix<double, 1, 3> vector_from, vector_to;
        vector_from = landmarks.row(getIndex(landmark_names, name_from));
        vector_to = landmarks.row(getIndex(landmark_names, name_to));
        return (vector_from + vector_to) * 0.5;
    }

    Eigen::Matrix<double, 1, 3> FullBodyPoseEmbedder::
                        get_dist_by_names(Eigen::Matrix<double, 33, 3>
                        landmarks, std::string name_from, std::string name_to)
    {
        Eigen::Matrix<double, 1, 3> vector_from, vector_to;
        vector_from = landmarks.row(getIndex(landmark_names, name_from));
        vector_to = landmarks.row(getIndex(landmark_names, name_to));
        return (vector_to - vector_from);
    }

    Eigen::Matrix<double, 1, 3> FullBodyPoseEmbedder::
                        get_distance(Eigen::Matrix<double, 1, 3>
                        vector_from, Eigen::Matrix<double, 1, 3> vector_to)
    {
        return (vector_to - vector_from);
    }

    int FullBodyPoseEmbedder::getIndex(std::vector<std::string> v, 
                                                        std::string K)
    {
        auto it = find(v.begin(), v.end(), K);
        int index = -1;
        // If element was found
        if (it != v.end())
        {
        
            // calculating the index
            // of K
            index = it - v.begin();
            
        }
        else {
            // If the element is not
            // present in the vector
            index = -1;
        }
        return index;
    }
    int FullBodyPoseEmbedder::getIndex(std::vector<int> v, int K)
    {
        auto it = find(v.begin(), v.end(), K);
        int index = -1;
        // If element was found
        if (it != v.end())
        {
        
            // calculating the index
            // of K
            index = it - v.begin();
            
        }
        else {
            // If the element is not
            // present in the vector
            index = -1;
        }
        return index;
    }
    // void FullBodyPoseEmbedder::test_func(Eigen::Matrix<double, 33, 3>
    //                     landmarks)
    // {
    //     std::cout << get_pose_center(landmarks);
    // }

}

    int main()
    {
        knn_action_classifier::FullBodyPoseEmbedder embedderobj(2.5);
        
        Eigen::Matrix<double, 33, 3> test_mat;
        test_mat << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33, 
                    34,	35,	36,	37	,38, 39,40,	41,	42,	43,	44,	45,	46,	47,	48,	49,50,	51,	52,	53,	54,	55,	56,
                    57, 58,	59,60,	61,	62,	63,	64,	65,	66,	67,	68,	69,70,	71,	72,	73,	74,	75,	76,	77,	78,	79,
                    80, 81,	82,	83,	84,	85,	86,	87,	88,	89,90,	91,	92,	93,	94,	95,	96,	97,	98,	99;
        knn_action_classifier::FullBodyPoseEmbedder posembedobj(2.5);
        std::cout<<posembedobj.call(test_mat);
        return 0;
    }