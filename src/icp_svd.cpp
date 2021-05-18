// icp_svd的实现

#include "icp_svd.h"


namespace my_icp_base_namespace{
    icp_svd::icp_svd(){
    std::cout<<"---icp_svd constructor---"<<std::endl;
        
    }
    
    icp_svd::~icp_svd()
    {
        std::cout<<"---icp_svd deconstructor---"<<std::endl;

    }

    void icp_svd::align(const std::vector<Eigen::Vector3d>& target,
                        const std::vector<Eigen::Vector3d>& source,
                        double n_iter,
                        double epsilon,
                        double min_err,
                        Eigen::Matrix3d& R, Eigen::Vector3d& t)
    {
        clock_t start = clock();
        // std::cerr<<"\n********icp_svd***********"<<std::endl;
        // 1、找到对应点对
        // 2、求解R,t
        // 3、判断是否收敛

        // 给初始值
        // R = Eigen::Matrix3d::Identity();
        // t.x() = 0;
        // t.y() = 0;
        // t.z() = 0;

        const int min_pointsize =/* target.size()<source.size()?target.size(): */source.size();          // 最少点数量
        const double min_err2 = min_err*min_err;
        const double factor = 9.0;
        const int n_selected_points = 100;
        const int step = min_pointsize / n_selected_points;

        // 创建两个vector来存放匹配点对
        std::vector<Eigen::Vector3d> source_matched;
        std::vector<Eigen::Vector3d> target_matched;
        source_matched.reserve(n_selected_points);
        target_matched.reserve(n_selected_points);

        // 将target构建kd-tree
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for(const auto& p:target)
        {   
            pcl::PointXYZ tmp_p;
            tmp_p.x = p.x();
            tmp_p.y = p.y();
            tmp_p.z = p.z();

            target_cloud->push_back(tmp_p);
        }
        pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>());
        kd_tree->setInputCloud(target_cloud);
        std::vector<float> k_sqr_distances(1);
        std::vector<int> indices(1);

        // 设置每一次迭代后的err
        double sqr_distance_th = std::numeric_limits<double>::max();                // 筛选匹配点的阈值
        double cur_sqr_distance = 0.0;
        double last_sqr_distance = std::numeric_limits<double>::max();

        // 对每一次迭代
        for(size_t iter = 0;iter<n_iter;++iter)
        {

            // 清空匹配点对
            source_matched.clear();
            target_matched.clear();

            // 统计这次的点距离平方和
            double sum_sqr_distance = 0;

            // 1 找到对应点
            int count = 0;      // 匹配点对数
            for(size_t i_source = 0;i_source<source.size();i_source+=step)
            {
                // 将src点云变换到target坐标系下
                Eigen::Vector3d p_srcT = R*source.at(i_source)+t;          // 变换后的点
                // LOG(INFO)<<source.at(i_source).x()<<", "<<source.at(i_source).y()<<", "<<source.at(i_source).z()<<std::endl;
                // LOG(INFO)<<p_srcT.x()<<", "<<p_srcT.y()<<", "<<p_srcT.z()<<std::endl;

                pcl::PointXYZ point_src(p_srcT.x(),p_srcT.y(),p_srcT.z());
                // LOG(INFO)<<point_src.x<<", "<<point_src.y<<", "<<point_src.z<<std::endl;

                // if(!pcl::isFinite<pcl::PointXYZ>(point_src)) continue;                     // 点无效，直接跳过
                // LOG(INFO)<<"kd-tree"<<std::endl;
                // kd-tree查找
                if(!kd_tree->nearestKSearch(point_src,1,indices,k_sqr_distances))        // 不成功对下一个点进行查找
                {
                    std::cerr << "ERROR: no points found.\n";
                    return;
                }
                // LOG(INFO)<<"i_source: "<<i_source<<", k_sqr_distances[0]: "<<k_sqr_distances[0]<<", sqr_distance_th: "<<sqr_distance_th<<std::endl;
                if(k_sqr_distances[0]<sqr_distance_th)
                {
                    sum_sqr_distance+=k_sqr_distances[0];
                    source_matched.push_back(source.at(i_source));
                    target_matched.push_back(target.at(indices[0]));
                    count++;
                }
            }   // for source.size()
            
            // 获取当前的误差以及误差变化量
            cur_sqr_distance = sum_sqr_distance/double(count);
            double sqr_distance_change = last_sqr_distance - cur_sqr_distance;

            // 更新sqr_distance_th，筛选匹配点对
            sqr_distance_th = factor*cur_sqr_distance;

            // 更新上一次的误差
            last_sqr_distance = cur_sqr_distance;

            LOG(INFO)<<"iter "<<iter<<", count: "<<count<<", cur_sqr_distance: "<<cur_sqr_distance<<", sqr_distance_change: "<<sqr_distance_change<<std::endl;
            // 若当前误差小，或者误差变化小，退出迭代
            if(cur_sqr_distance<min_err2 || sqr_distance_change<epsilon)  
            {
                std::cerr<<"["<<iter<<"] coverage"<<std::endl;
                std::cerr<<"final cost: "<<cur_sqr_distance<<", cost change: "<<sqr_distance_change<<std::endl;
                break;
            }
            

            // std::cout<<"[iter: "<<iter<<"]  match_num: "<<source_matched.size()<<", "<<target_matched.size()<<std::endl;
            // std::cout<<"得到对应点对了"<<std::endl;

            // 2 计算R，t
            // 2.1 计算中心
            Eigen::Vector3d center_src(0.0,0.0,0.0);
            Eigen::Vector3d center_tgt(0.0,0.0,0.0);
            for(size_t i = 0;i<source_matched.size();i++)        // 遍历匹配src中的每一个点
            {
                center_src+=source_matched.at(i);
                center_tgt+=target_matched.at(i);
            }
            center_src/=double(source_matched.size());
            center_tgt/=double(target_matched.size());

            // 2.2 求W
            Eigen::Matrix3d W(Eigen::Matrix3d::Identity());         // 记得矩阵W变量给初值，否则会报错！！！
            for(size_t i = 0;i<source_matched.size();i++)
            {
                W+=(source_matched.at(i)-center_src)*( (target_matched.at(i)-center_tgt).transpose() );         // 3*3的矩阵
            }

            // 2.3 求R,t
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU | Eigen::ComputeFullV);


            R = svd.matrixU()*(svd.matrixV().transpose());
            t = center_tgt - R*center_src;

            LOG(INFO)<<"R: "<<std::endl;
            std::cerr<<R<<std::endl;
            LOG(INFO)<<"t: "<<std::endl;
            std::cerr<<t<<std::endl;

            // 检查


        }       // 迭代

        // delete kd_tree;      // share指针不需要删除

        // std::cerr<<"------------------\n"<<std::endl;

        clock_t end = clock();
        double time = double(end-start)/CLOCKS_PER_SEC;
        std::cerr<<"spend time: "<<time*1000<<"ms"<<std::endl;

    }       // align函数结束
                        





//  void icp_svd::align(const std::vector<Eigen::Vector3d>& target,
//                         const std::vector<Eigen::Vector3d>& source,
//                         double n_iter,
//                         double epsilon,
//                         double min_err,
//                         Eigen::Matrix3d& R, Eigen::Vector3d& t)
// {
// 	// default settings.
// 	const double min_err2 = min_err * min_err;
// 	const double factor = 9.0;
// 	const int n_selected_pts = 100;
// 	const int step = source.size() / n_selected_pts;	// step for select points.
	
// 	// two vectors for matched points.
//     std::vector<Eigen::Vector3d> source_matched;
// 	source_matched.reserve ( n_selected_pts );
//     std::vector<Eigen::Vector3d> target_matched;
// 	target_matched.reserve ( n_selected_pts );

//     // construct kd-tree for model cloud.   model代表target点云
//     pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud ( new pcl::PointCloud<pcl::PointXYZ> );
//     for ( size_t i = 0; i < target.size(); ++i ) {
//         const Eigen::Vector3d& ptm = target.at ( i );
//         pcl::PointXYZ pt ( ptm[0], ptm[1], ptm[2] );
//         model_cloud->push_back ( pt );
//     }
//     pcl::KdTreeFLANN<pcl::PointXYZ>* kd_tree = new pcl::KdTreeFLANN <pcl::PointXYZ>();
//     kd_tree->setInputCloud ( model_cloud );

// 	// used for search.
// 	std::vector <int> index (1);
// 	std::vector <float> squared_distance (1);
	
// 	// Dth
// 	double squared_distance_th = std::numeric_limits <double>::max ();
// 	double cur_squared_dist = 0.0;
// 	double last_squared_dist = std::numeric_limits<double>::max();
	
//     //for n_iter
//     for ( int n = 0; n < n_iter; n ++ ) {

// 		// clear two point clouds.
// 		source_matched.clear();
// 		target_matched.clear();
		
//         // step 1. construct matched point clouds.
// 		double sum_squared_dist = 0.0;
		
// 		// 遍历source点云
// 		for ( size_t i = 0; i < source.size(); i += step ) {
			
//             // transformed by T.
//             Eigen::Vector3d pt = R *  source.at ( i ) + t;

//             // find the nearest pints by knn
// 			pcl::PointXYZ pt_d(pt[0], pt[1], pt[2]);
// 			if (!kd_tree->nearestKSearch (pt_d, 1, index, squared_distance))
// 			{
// 				std::cerr << "ERROR: no points found.\n";
// 				return;
// 			}
			
// 			if(squared_distance[0] < squared_distance_th)
// 			{
// 				// add squared distance.
// 				sum_squared_dist += squared_distance[0];
// 				// add the pt in cloud.   source
// 				source_matched.push_back(source.at(i));
// 				// add the pt in model.  target
// 				target_matched.push_back(target.at(index[0]));
// 			}
//         } // for all source


//         //std::cout << "iter:" << n << " mathced size: " << target_matched.size() << " " << source_matched.size() << std::endl;
        
//         // step 2. Get R and t.
//         // step 2.1 find centor of model(X)  and cloud(P)
//         Eigen::Vector3d mu_x(0.0, 0.0, 0.0);
// 		Eigen::Vector3d mu_p(0.0, 0.0, 0.0);
//         for(size_t i = 0; i < source_matched.size(); i ++)
// 		{
// 			mu_x += target_matched.at(i);
// 			mu_p += source_matched.at(i);
// 		}
//         mu_x = mu_x / double(target_matched.size());
// 		mu_p = mu_p / double(source_matched.size());
		
// 		// step 2.2 Get W.
		
// 		Eigen::Matrix3d W;
// 		for(size_t i = 0; i < source_matched.size(); i ++)
// 		{
// 			W += (target_matched.at(i)-mu_x) * ( (source_matched.at(i)-mu_p).transpose() );
// 		}
		
// 		// step 2.3 Get R
// 		Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
		
// 		R = svd.matrixU() * (svd.matrixV().transpose());
		
// 		// step 2.4 Get t
// 		t  = mu_x - R * mu_p;

//         LOG(INFO)<<"R: "<<std::endl;
//         std::cerr<<R<<std::endl;
//         LOG(INFO)<<"t: "<<std::endl;
//         std::cerr<<t<<std::endl;

		
// 		// step 3. Check if convergenced.
// 		cur_squared_dist = sum_squared_dist / (double)source_matched.size();
// 		double squared_dist_change = last_squared_dist -cur_squared_dist;
		
// 		//std::cout << "iter:" << n << " squared_dist_change: " << squared_dist_change << " cur distance " << cur_squared_dist  << std::endl;
		
// 		if(squared_dist_change < epsilon || cur_squared_dist < min_err2)
// 		{
// 			std::cout<<"["<<n<<"] coverage"<<std::endl;
// 			std::cout<<"final cost: "<<cur_squared_dist<<", cost change: "<<squared_dist_change<<std::endl;
// 			break;
// 		}
// 		last_squared_dist = cur_squared_dist;
// 		squared_distance_th = factor * cur_squared_dist;
		
//     } // for n_iter
    
//     delete kd_tree;
// } // findTransformation



}       // namespace my_icp_base_namespace