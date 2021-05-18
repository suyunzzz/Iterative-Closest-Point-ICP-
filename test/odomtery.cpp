// Copyright (C) 2019 Dongsheng Yang <ydsf16@buaa.edu.cn>
//(Biologically Inspired Mobile Robot Laboratory, Robotics Institute, Beihang University)

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <pcl/registration/icp.h>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include <icp_svd.h>
#include <icp_GN.h>         // GN解法
// #include <my_icp_base.h>


using namespace std;
using namespace my_icp_base_namespace;

void LoadImages ( const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                  vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps );

int main ( int argc, char **argv )
{
    if ( argc != 4 ) {
        std::cerr << endl << "Usage: ./icp_odometry path_to_sequence path_to_association traj_name" << endl;
        return -1;
    }
    std::cerr<<"---start---"<<std::endl;

    // camera params
    const double depth_factor = 1.0 / 5000.0;
    const double fx = 517.306408;
    const double fy = 516.469215;
    const double cx = 318.643040;
    const double cy =  255.313989;
    const int rows = 480;
    const int cols = 640;

    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string ( argv[2] );
    LoadImages ( strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps );
    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if ( vstrImageFilenamesRGB.empty() ) {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    } else if ( vstrImageFilenamesD.size() !=vstrImageFilenamesRGB.size() ) {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // LOG(INFO)<<"---------"<<std::endl;
    // Main loop
    cv::Mat imRGB, imD;
    cv::Mat imD_last, imRGB_last;

    // Two points clouds.
    std::vector<Eigen::Vector3d> pts_model;
    std::vector<Eigen::Vector3d> pts_cloud;
    pts_model.reserve ( rows*cols );
    pts_cloud.reserve ( rows*cols );

    // pcl中icp
    pcl::IterativeClosestPoint<pcl::PointXYZ,pcl::PointXYZ> icp;
    icp.setMaximumIterations(30);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(0.05);
    // icp.setMaxCorrespondenceDistance()               // 设置对应点对之间的距离，小于此距离才能视为对应点

    // 设置voxel
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setLeafSize(0.1,0.1,0.1);


    // 创建自己实现的icp对象
    my_icp_base* icpPtr = new icp_GN();
    icp_svd icp_svd_obj;
    my_icp_base& icp_ref = icp_svd_obj;

    pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr res(new pcl::PointCloud<pcl::PointXYZ>);

    // Global pose of the camera.
    Eigen::Matrix3d Rwc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d twc ( 0.0, 0.0, 0.0 );

    // Traj file.
    ofstream f;
    std::string traj_name = string(argv[3]);
    f.open ( traj_name );
    f << fixed;

    

    for ( int ni=0; ni<nImages; ni++ ) {
        // Read image and depthmap from file
        imRGB = cv::imread ( string ( argv[1] ) +"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED );
        imD = cv::imread ( string ( argv[1] ) +"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED );
        double tframe = vTimestamps[ni];

        if ( imRGB.empty() ) {
            cerr << endl << "Failed to load image at: "
                 << string ( argv[1] ) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        // using ICP.
        if ( !imD.empty() && !imD_last.empty() ) {
            
            // LOG(INFO)<<"using ICP.\n";
            pts_model.clear();
            pts_cloud.clear();

            // 点云清空
            model->clear();
            cloud->clear();
            res->clear();



            // Get two Clouds
            for ( int v = 0; v < imD.rows; v++ ) {
                for ( int u = 0; u < imD.cols; u ++ ) {
                    unsigned int d = imD.ptr<unsigned short> ( v ) [u];
                    if ( d == 0 ) {
                        continue;
                    }

                    double z = d * depth_factor;
                    double x = ( u-cx ) * z / fx;
                    double y = ( v-cy ) * z / fy;

                    pts_cloud.push_back ( Eigen::Vector3d ( x,y,z ) );
                    cloud->push_back(pcl::PointXYZ(x,y,z));
                } // for all u
            } // for all v

            for ( int v = 0; v < imD_last.rows; v++ ) {
                for ( int u = 0; u < imD_last.cols; u ++ ) {
                    unsigned int d = imD_last.ptr<unsigned short> ( v ) [u];
                    if ( d == 0 ) {
                        continue;
                    }

                    double z = d * depth_factor;
                    double x = ( u-cx ) * z / fx;
                    double y = ( v-cy ) * z / fy;

                    pts_model.push_back ( Eigen::Vector3d ( x,y,z ) );
                    model->push_back(pcl::PointXYZ(x,y,z));
                } // for all u
            } // for all v

            // save
            // pcl::io::savePCDFile("model.pcd",*model);
            // pcl::io::savePCDFile("cloud.pcd",*cloud);

            // find transformation by ICP
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
            Eigen::Vector3d t ( 0.0,0.0,0.0 );
            // LOG(INFO)<<"---icp---"<<std::endl;
            // 使用自己实现的icp
            // icp_svd::align ( pts_model, pts_cloud, 100, 1.0e-10, 1.0e-10, R, t );
            // icpPtr->align( pts_model, pts_cloud, 100, 1.0e-10, 1.0e-10, R, t );
            icp_ref.align( pts_model, pts_cloud, 100, 1.0e-10, 1.0e-10, R, t );


            // // 使用pcl中的icp
            // filter.setInputCloud(model);
            // filter.filter(*model);
            // filter.setInputCloud(cloud);
            // filter.filter(*cloud);
            // LOG(INFO)<<"model num: "<<model->size()<<std::endl;
            // LOG(INFO)<<"cloud num: "<<cloud->size()<<std::endl;
            // icp.setInputTarget(model);
            // icp.setInputSource(cloud);
            // Eigen::Matrix4f guess_mat = Eigen::Matrix4f::Identity();
            // icp.align(*res,guess_mat);
            // LOG(INFO)<<"res num: "<<res->size()<<std::endl;
            // pcl::io::savePCDFile("res.pcd",*res);
            
            // Eigen::Matrix4f T =  icp.getFinalTransformation();
            // LOG(INFO)<<"T:\n";
            // std::cout<<T<<std::endl;
            // R = T.block<3,3>(0,0).cast<double>();
            // t = T.block<3,1>(0,3).cast<double>();
            


            // conver local R t to global pose of the camera Rwc, twc
            Rwc = Rwc * R;
            twc = Rwc * t + twc;

            // convert R to Quaternion.
            Eigen::Quaterniond q ( R );

            // save poses to file.
            f << setprecision ( 6 ) << tframe << " " <<  setprecision ( 9 ) << twc[0] << " " << twc[1] << " " << twc[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

            // print poses.
            std::cerr << std::fixed << setprecision ( 6 ) << tframe << " " <<  setprecision ( 9 ) << twc[0] << " " << twc[1] << " " << twc[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;

            // show two images.
            cv::imshow ( "Cur RGB", imRGB );
            cv::imshow ( "Last RGB", imRGB_last );
            cv::waitKey ( 1 );
        } // using ICP.

        imD_last = imD.clone();
        imRGB_last = imRGB.clone();
    } // for all images.

    delete icpPtr;      // 删除指针
    std::cerr<<"traj_name: "<<traj_name<<std::endl;


    return 0;
}



void LoadImages ( const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                  vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps )
{
    ifstream fAssociation;
    fAssociation.open ( strAssociationFilename.c_str() );
    while ( !fAssociation.eof() ) {
        string s;
        getline ( fAssociation,s );
        if ( !s.empty() ) {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back ( t );
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back ( sRGB );
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back ( sD );
        }
    }
}