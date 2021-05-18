#ifndef _MY_ICP_BASE_
#define _MY_ICP_BASE_

#pragma once
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

#include <pcl/kdtree/kdtree_flann.h>
#include <glog/logging.h>
#include <sophus/so3.h>

#include <ctime>


namespace my_icp_base_namespace
{

class my_icp_base{
public:
    my_icp_base();

    virtual ~my_icp_base();

    virtual void align(const std::vector<Eigen::Vector3d>& target,
                        const std::vector<Eigen::Vector3d>& source,
                        double n_iter,
                        double epsilon,
                        double min_err,
                        Eigen::Matrix3d& R, Eigen::Vector3d& t) = 0;

};      // class my_icp_base

}   // namespace my_icp_namespace

#endif