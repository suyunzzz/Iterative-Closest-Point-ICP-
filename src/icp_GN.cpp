#include "icp_GN.h"


namespace my_icp_base_namespace{

icp_GN::icp_GN()
{
    std::cout<<"---icp_GN constructor---"<<std::endl;

}

icp_GN::~icp_GN()
{
    std::cout<<"---icp_GN deconstructor---"<<std::endl;
}

void icp_GN::align(const std::vector<Eigen::Vector3d>& target,
                        const std::vector<Eigen::Vector3d>& source,
                        double n_iter,
                        double epsilon,
                        double min_err,
                        Eigen::Matrix3d& R, Eigen::Vector3d& t)
{
    clock_t start = clock();

    // std::cerr<<"\n********icp_GN***********"<<std::endl;

    // 1找对应点
    // 2计算误差函数f(x),雅克比矩阵J-->H,b
    // 3解增量方程，若delta_x小于某个值，或者变换后的mes小于某个值，则收敛

    const double min_err2_th = min_err*min_err;                                     // 距离平方和的平均值小于这个值，收敛
    const double factor = 9.0;
    const int n_selected_points = 500;
    const int step = source.size() / n_selected_points;

    double sqr_distance_th = std::numeric_limits<double>::max();                // 筛选最近邻匹配点的阈值，大于此阈值不考虑
    double cur_sqr_distance = 0.0;
    double last_sqr_distance = std::numeric_limits<double>::max();

    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // target构建点云
    for(const auto& p_target:target)
    {
        pcl::PointXYZ tmp(p_target.x(), p_target.y(), p_target.z());
        target_cloud->push_back(tmp);
    }
    // kd-tree
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr KDTREE(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    KDTREE->setInputCloud(target_cloud);
    std::vector<float> k_sqr_distances;
    std::vector<int> k_indices;

    for(size_t iter = 0;iter<n_iter;iter++)
    {
        
        // 对source中的每一个点，寻找最近点
        size_t count = 0;           // 计数，此次迭代有多少个点对
        double err = 0;             // 统计对应点对的误差和

        // 创建总的J,b
        Eigen::Matrix<double,6,6> H_sum(Eigen::Matrix<double,6,6>::Zero());
        Eigen::Matrix<double,6,1> b_sum(Eigen::Matrix<double,6,1>::Zero());
        for(size_t j = 0;j<source.size();j+=step)
        {

            const Eigen::Vector3d p_src = source.at(j);
            // source点云变换R,t
            Eigen::Vector3d p_src_T = R*p_src+t;      
            // Eigen::Vector3d delta_p = p_src_T-p_src;            // f(x)
            // kdtree search
            if(!KDTREE->nearestKSearch(pcl::PointXYZ(p_src_T.x(),p_src_T.y(),p_src_T.z()),
                                        1,k_indices,k_sqr_distances)) 
                                        {
                                            continue;
                                        }
            if(k_sqr_distances[0]>sqr_distance_th)  continue;

            // 匹配上了
            count++;                                // 匹配点+1
            err+=k_sqr_distances[0];                // 匹配点之间的距离平方累加

            Eigen::Matrix<double,3,6> J(Eigen::Matrix<double,3,6>::Zero());
            Eigen::Vector3d q( target_cloud->at(k_indices[0]).x, target_cloud->at(k_indices[0]).y,target_cloud->at(k_indices[0]).z);    // target点云中的点
            Eigen::Matrix<double,3,1> f_x = p_src_T - q;                // f(x)误差函数就是两个向量的差
            J.block<3,3>(0,0) = -R*Sophus::SO3::hat(p_src);
            J.block<3,3>(0,3) = Eigen::Matrix3d::Identity();            // 右侧为评平移
            Eigen::Matrix<double,6,6> H = J.transpose()*J;
            Eigen::Matrix<double,6,1> b = -J.transpose()*f_x;

            H_sum+=H;
            b_sum+=b;

        }   // for loop source

       
        cur_sqr_distance = err/double(count);                // 得到上一轮的误差
        sqr_distance_th = cur_sqr_distance*factor;                  // 筛选最近邻匹配点的阈值，大于此阈值不考虑
        double err_change = last_sqr_distance-cur_sqr_distance;     // 得到err的变化
        last_sqr_distance  = cur_sqr_distance;
        std::cerr<<"Iter: "<<iter<<", count: "<<count<<", cur_sqr_distance: "<<cur_sqr_distance<<", err_change: "<<err_change<<std::endl;
        if(cur_sqr_distance<min_err2_th || err_change<epsilon)    {std::cerr<<"converge"<<std::endl;return;}      // 误差小，直接返回

        // 没有收敛，计算新的R,t
        if(H_sum.determinant()==0)
        {
            continue;
        }

        Eigen::Matrix<double,6,1> delta_x = H_sum.inverse()*b_sum;
        t+=delta_x.tail<3>();
        auto delta_R = Sophus::SO3::exp(delta_x.head<3>()); 
        R = R*delta_R.matrix();          // 右扰动

        


    }// for loop (iter_n)

    clock_t end = clock();
    double time = double(end-start)/CLOCKS_PER_SEC;
    std::cerr<<"spend time: "<<time*1000<<std::endl;

}
}           // namespace my_icp_base_namespace