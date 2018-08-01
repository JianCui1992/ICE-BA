cmake_minimum_required(VERSION 2.8.11)
project(L1_test)

set(VTRACKER_DIR /home/jianc/uos_stable)
#opencv
include_directories(SYSTEM ${VTRACKER_DIR}/src/uos_3rdparty/opencv33/16.04/include)
link_directories(${VTRACKER_DIR}/src/uos_3rdparty/opencv33/16.04/lib)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_calib3d opencv_features2d opencv_imgproc opencv_imgcodecs opencv_video)
# =========================
# Example
# =========================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)

add_executable(ice_ba
    ../App/L1_test.cpp
)

target_link_libraries(ice_ba
    IBA
    OF
    ${OpenCV_LIBS}
    ${BOOST_LIBS}
)
