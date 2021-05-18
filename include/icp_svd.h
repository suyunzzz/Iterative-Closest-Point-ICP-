// icp的svd解法

#pragma once
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

#include <pcl/kdtree/kdtree_flann.h>
// #include <glog/logging.h>
#include <my_icp_base.h>

namespace my_icp_base_namespace{

class icp_svd:public my_icp_base
{
    public:
        icp_svd();
        ~icp_svd();
    
        void align(const std::vector<Eigen::Vector3d>& target,
                        const std::vector<Eigen::Vector3d>& source,
                        double n_iter,
                        double epsilon,
                        double min_err,
                        Eigen::Matrix3d& R, Eigen::Vector3d& t);
};      // class icp_svd
}       // namespace my_icp_base_namespace