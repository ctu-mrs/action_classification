#pragma once
#ifndef PoseEmbed_H
#define PoseEmbed_H

#include <iostream>
#include <algorithm>
#include <bits/stdc++.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <array>
// #include <opencv2/core/eigen.hpp>

#include<mrs_lib/param_loader.h>



namespace knn_action_classifier
{
    class FullBodyPoseEmbedder
    {
        public:
            FullBodyPoseEmbedder(float torso_size_multiplier = 2.5);
            float torso_size_multiplier_var;
            std::vector<std::string> landmark_names;

            Eigen::Matrix<double, 23, 3> 
                                call(Eigen::Matrix<double, 33, 3> landmarks);
            int getIndex(std::vector<std::string> v, std::string K);
            int getIndex(std::vector<int> v, int K);
            Eigen::Matrix<double, 1, 3> 
                        get_pose_center(Eigen::Matrix<double, 33, 3>
                        landmarks);
            // void test_func(Eigen::Matrix<double, 33, 3>
            //             landmarks);


        private: 
            
            Eigen::Matrix<double, 33, 3> landmarks;

            Eigen::Matrix<double, 33, 3> 
                        normalize_pose_landmarks(Eigen::Matrix<double, 33, 3> 
                        landmarks); 
            Eigen::Matrix<double, 23, 3> 
                        get_pose_distance_embedding(Eigen::Matrix<double, 33, 3>
                        landmarks);            
            double get_pose_size(Eigen::Matrix<double, 33, 3>
                        landmarks, double torso_size_multiplier);
            

            Eigen::Matrix<double, 1, 3> 
                        get_average_by_names(Eigen::Matrix<double, 33, 3>
                        landmarks, std::string name_from, std::string name_to);
            Eigen::Matrix<double, 1, 3> 
                        get_dist_by_names(Eigen::Matrix<double, 33, 3>
                        landmarks, std::string name_from, std::string name_to);
            Eigen::Matrix<double, 1, 3> 
                        get_distance(Eigen::Matrix<double, 1, 3>
                        vector_from, Eigen::Matrix<double, 1, 3> vector_to);
            
    };
}


#endif