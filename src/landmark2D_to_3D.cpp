#include<landmark2D_to_3D.hpp>

namespace landmark2Dto3D
{
    void LandmarkConverter::onInit()
    {
        is_initialized_ = false;

        ros::NodeHandle nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();
        ros::Time::waitForValid();


        mrs_lib::ParamLoader param_loader(nh_, "Landmark3DConverter");
        param_loader.loadParam("uav_name", _uav_name_);
        param_loader.loadParam("depth_camera_topic", _depth_camera_topic_);


    }
}