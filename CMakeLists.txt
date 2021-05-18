cmake_minimum_required(VERSION 2.6)
project(my_icp)

add_compile_options(-std=c++11)
SET(CMAKE_BUILD_TYPE Release)

list( APPEND CMAKE_MODULE_PATH "/home/s/3rdparty/ceres-solver/cmake" )

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Glog REQUIRED)
find_package(Sophus REQUIRED)


include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}

    ${Glog_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib SHARED  src/my_icp_base.cpp src/icp_svd.cpp src/icp_GN.cpp)

target_link_libraries(${PROJECT_NAME}_lib 
                        ${PCL_LIBRARIES}
                        ${OpenCV_LIBS}
                        ${CERES_LIBRARIES}
                        ${Sophus_LIBRARIES}
                        )


                    
add_executable(odom test/odomtery.cpp)
target_link_libraries(odom
                    ${PROJECT_NAME}_lib
                    )

