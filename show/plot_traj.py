import numpy as np
import matplotlib.pyplot as plt


# my_svd_pose = np.loadtxt("test/my_svd_traj.txt")
# zhihu_svd_pose = np.loadtxt("test/zhihu_svd_traj.txt")

gt = np.loadtxt("test/gt.txt")
icp_svd_500 = np.loadtxt("show/traj_svd_500points.txt")
icp_svd_100 = np.loadtxt("show/traj_svd_100points.txt")


icp_GN500 = np.loadtxt("show/traj_GN_500points.txt")
icp_GN100 = np.loadtxt("show/traj_GN_100points.txt")

icp_pcl = np.loadtxt("test/pcl_icp_traj.txt")



fig = plt.figure()

ax = fig.gca(projection =  '3d')
# ax.plot(my_svd_pose[:,1],my_svd_pose[:,2],my_svd_pose[:,3],label = "my_svd_icp")
# ax.plot(zhihu_svd_pose[:,1],zhihu_svd_pose[:,2],zhihu_svd_pose[:,3],label = "zhihu_svd_icp")
# ax.plot(pcl_icp[:,1],pcl_icp[:,2],pcl_icp[:,3],label = "pcl_icp")
ax.plot(gt[:,1],gt[:,2],gt[:,3],label = 'gt')
ax.plot(icp_GN500[:,1],icp_GN500[:,2],icp_GN500[:,3],label = "icp_GN500")
ax.plot(icp_GN100[:,1],icp_GN100[:,2],icp_GN100[:,3],label = 'icp_GN100')
ax.plot(icp_svd_100[:,1],icp_svd_100[:,2],icp_svd_100[:,3],label = 'icp_svd_100')
ax.plot(icp_svd_500[:,1],icp_svd_500[:,2],icp_svd_500[:,3],label = 'icp_svd_500')
ax.plot(icp_pcl[:,1],icp_pcl[:,2],icp_pcl[:,3],label = "icp_pcl")

ax.legend()
plt.show()


