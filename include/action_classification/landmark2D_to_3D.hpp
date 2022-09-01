#pragma once
#ifndef Landmark2Dto3D_H
#define Landmark2Dto3D_H

#include<ros/ros.h>
#include<ros/package.h>
#include<nodelet/nodelet.h>

#include<atomic>
#include<mrs_lib/param_loader.h>
#include<sensor_msgs/Image.h>
#include<eigen3/Eigen/Eigen>
#include<std_srvs/Trigger.h>

#include<action_classification/landmark.h>
#include<action_classification/landmark3D.h>

using namespace std;
namespace e = Eigen;

namespace landmark2Dto3D
{
    class LandmarkConverter: public nodelet::Nodelet {
        public:
        virtual void onInit();

        private:
        atomic<bool> is_initialized_ = false;
        shared_ptr<action_classification::landmark3D> landmark_to_publish_; 

        string _uav_name_;
        string _depth_camera_topic_;

        void callbackDepthImage(const sensor_msgs::ImageConstPtr& depthImage);
        ros::Subscriber sub_depth_;
        void callbackLandmarks(const action_classification::landmark& 
                                                                landmark_data);
        ros::Subscriber sub_landmark_;

        ros::Publisher landmark3D_;
        ros::ServiceServer server_run_ladnmark_converter_;
        bool callbackLandmarkConverterTrigger([[maybe_unused]] 
            std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);
        
    };
}


#endif
