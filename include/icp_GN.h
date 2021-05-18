// icp的高斯牛顿解法

#pragma once
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

#include <pcl/kdtree/kdtree_flann.h>
#include <glog/logging.h>
#include <sophus/so3.h>
#include <my_icp_base.h>

// namespace REG{
namespace my_icp_base_namespace{

class icp_GN : public my_icp_base
{
    public:
        icp_GN();
        ~icp_GN();
        void align(const std::vector<Eigen::Vector3d>& target,
                        const std::vector<Eigen::Vector3d>& source,
                        double n_iter,
                        double epsilon,
                        double min_err,
                        Eigen::Matrix3d& R, Eigen::Vector3d& t);
};      // class icp_GN
}       // namespace my_icp_base_namespace