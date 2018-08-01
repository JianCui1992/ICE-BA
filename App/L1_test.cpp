/******************************************************************************
 * Copyright 2017 Baidu Robotic Vision Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include "IBA/IBA.h"
#include "basic_datatype.h"
#include "feature_utils.h"
#include "iba_helper.h"
#include "image_utils.h"
#include "param.h" // calib
#include "pose_viewer.h"
#include "xp_quaternion.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

Eigen::Matrix4d eigen_T;
std::string image_path;
std::string imu_path;
int image_width;
int image_height;
cv::Mat cv_K;
cv::Mat cv_D;
int start_id;
int end_id;

int grid_row_num;
int grid_col_num;
int max_num_per_grid;

double uniform_radius;
int ft_len;
double ft_droprate;
double max_feature_distance_over_baseline_ratio;
double min_feature_distance_over_baseline_ratio;
int fast_thresh;
int pyra_level;
int not_use_fast;
double TIME_SHIFT;

void read_parameters(const std::string &filename)
{
    // std::cout << filename << std::endl;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "ERROR: Wrong path to setting" << std::endl;
    }
    image_path = (std::string)fs["SLAM.image_path"];
    imu_path = (std::string)fs["SLAM.imu_path"];
    // std::cout<< image_path << std::endl;
    // std::cout<< imu_path << std::endl;


    image_height = fs["CAMERA.image_height"];
    image_width = fs["CAMERA.image_width"];

    // extrinsic
    cv::Mat cv_R, cv_t, cv_T;
    cv_T = cv::Mat::eye(4, 4, CV_64F);
    fs["CAMERA.extrinsicRotation"] >> cv_R;
    fs["CAMERA.extrinsicTranslation"] >> cv_t;
    printf("%2f, %2f, %2f, %2f\n", cv_R.at<double>(0, 0), cv_R.at<double>(0, 1),
           cv_R.at<double>(0, 2), cv_t.at<double>(0, 0));
    printf("%2f, %2f, %2f, %2f\n", cv_R.at<double>(1, 0), cv_R.at<double>(1, 1),
           cv_R.at<double>(1, 2), cv_t.at<double>(0, 1));
    printf("%2f, %2f, %2f, %2f\n", cv_R.at<double>(2, 0), cv_R.at<double>(2, 1),
           cv_R.at<double>(2, 2), cv_t.at<double>(0, 2));
    printf("0.0, 0.0, 0.0, 1.0\n");
    cv_R.copyTo(cv_T.rowRange(0, 3).colRange(0, 3));
    
    cv_t.copyTo(cv_T.rowRange(0, 3).col(3));
    
    printf("%2f, %2f, %2f, %2f\n", cv_T.at<double>(0, 0), cv_T.at<double>(0, 1),
           cv_T.at<double>(0, 2), cv_T.at<double>(0, 3));
    printf("%2f, %2f, %2f, %2f\n", cv_T.at<double>(1, 0), cv_T.at<double>(1, 1),
           cv_T.at<double>(1, 2), cv_T.at<double>(1, 3));
    printf("%2f, %2f, %2f, %2f\n", cv_T.at<double>(2, 0), cv_T.at<double>(2, 1),
           cv_T.at<double>(2, 2), cv_T.at<double>(2, 3));
    printf("%2f, %2f, %2f, %2f\n", cv_T.at<double>(3, 0), cv_T.at<double>(3, 1),
           cv_T.at<double>(3, 2), cv_T.at<double>(3, 3));
    std::cout << cv_T << std::endl;

    cv::cv2eigen(cv_T, eigen_T);
    fs["CAMERA.intrinsics.K"] >> cv_K;
    fs["CAMERA.intrinsics.D"] >> cv_D;

    start_id = fs["SLAM.start_id"];
    end_id = fs["SLAM.end_id"];
    grid_row_num = fs["FEATURE.grid_row_num"];
    grid_col_num = fs["FEATURE.grid_col_num"];
    max_num_per_grid = fs["FEATURE.max_num_per_grid"];

    uniform_radius = fs["FEATURE.uniform_radius"];
    ft_len = fs["FEATURE.ft_len"];
    ft_droprate = fs["FEATURE.ft_droprate"];
    max_feature_distance_over_baseline_ratio = fs["FEATURE.max_feature_distance_over_baseline_ratio"];
    min_feature_distance_over_baseline_ratio = fs["FEATURE.min_feature_distance_over_baseline_ratio"];
    fast_thresh = fs["FEATURE.fast_thresh"];
    pyra_level = fs["FEATURE.pyra_level"];
    not_use_fast = fs["FEATURE.not_use_fast"];
    TIME_SHIFT = fs["time_shift"];
}

size_t load_img_from_file(const std::string &image_path,
                          std::vector<std::string> &v_image_name,
                          std::vector<double> &v_time_stamp)
{

    std::ifstream fimage;
    std::string strPathImageFile = image_path + "image_name.txt";

    fimage.open(strPathImageFile.c_str());
    std::string strPrefixLeft = image_path;

    while (!fimage.eof())
    {
        std::string s;
        getline(fimage, s);
        if (!s.empty())
        {
            v_image_name.push_back(strPrefixLeft + s);

            std::stringstream stime(s.substr(0, 17));
            double time;
            stime >> time;
            v_time_stamp.push_back(time - TIME_SHIFT);
        }
    }
    printf("The image load over, total %ld .\n", v_image_name.size());
    return v_time_stamp.size();
}
size_t load_imu_from_file(const std::string &imu_path,
                                     std::list<XP::ImuData> *imu_samples_ptr)
{
    std::list<XP::ImuData> &imu_samples = *imu_samples_ptr;
    std::ifstream fimu(imu_path);
    if (fimu.is_open())
    {
        double var0, var1, var2, var3, var4, var5, var6;
        imu_samples.clear();
        while (fimu >> var0 >> var1 >> var2 >> var3 >> var4 >> var5 >> var6)
        {

            XP::ImuData imu_sample;
            imu_sample.time_stamp = var6 - TIME_SHIFT;
            imu_sample.ang_v(0) = var0;
            imu_sample.ang_v(1) = var1;
            imu_sample.ang_v(2) = var2;
            imu_sample.accel(0) = var3;
            imu_sample.accel(1) = var4;
            imu_sample.accel(2) = var5;
            // VLOG(3) << "accel " << imu_sample.accel.transpose() << " gyro "
            //         << imu_sample.ang_v.transpose();
            imu_samples.push_back(imu_sample);
        }
    }
    // LOG(INFO) << "loaded " << imu_samples.size() << " imu samples";
    return imu_samples.size();
}

void load_asl_calib(const cv::Mat cv_K, const cv::Mat cv_D,
                    XP::DuoCalibParam &calib_param)
{

    // intrinsics
    calib_param.Camera.cv_camK_lr[0] << 
        cv_K.at<float>(0, 0),
        cv_K.at<float>(0, 1), cv_K.at<float>(0, 2), cv_K.at<float>(1, 0),
        cv_K.at<float>(1, 1), cv_K.at<float>(1, 2), cv_K.at<float>(2, 0),
        cv_K.at<float>(2, 1), cv_K.at<float>(2, 2);

    calib_param.Camera.cameraK_lr[0] << cv_K.at<float>(0, 0),
        cv_K.at<float>(0, 1), cv_K.at<float>(0, 2), cv_K.at<float>(1, 0),
        cv_K.at<float>(1, 1), cv_K.at<float>(1, 2), cv_K.at<float>(2, 0),
        cv_K.at<float>(2, 1), cv_K.at<float>(2, 2);

    // distortion_coefficients
    calib_param.Camera.cv_dist_coeff_lr[0] =
        (cv::Mat_<float>(8, 1) << cv_D.at<float>(0), cv_D.at<float>(1),
         cv_D.at<float>(2), cv_D.at<float>(3), 0.0, 0.0, 0.0, 0.0);


    calib_param.Camera.D_T_C_lr[0] = Eigen::Matrix4f::Identity();
    // Image size
    calib_param.Camera.img_size = cv::Size(image_width, image_height);
    // IMU
    calib_param.Imu.accel_TK = Eigen::Matrix3f::Identity();
    calib_param.Imu.accel_bias = Eigen::Vector3f::Zero();
    calib_param.Imu.gyro_TK = Eigen::Matrix3f::Identity();
    calib_param.Imu.gyro_bias = Eigen::Vector3f::Zero();
    calib_param.Imu.accel_noise_var = Eigen::Vector3f{0.0016, 0.0016, 0.0016};
    calib_param.Imu.angv_noise_var = Eigen::Vector3f{0.0001, 0.0001, 0.0001};
    calib_param.Imu.D_T_I = eigen_T.cast<float>();
    calib_param.device_id = "ASL";
    calib_param.sensor_type = XP::DuoCalibParam::SensorType::UNKNOWN;

    calib_param.initUndistortMap(calib_param.Camera.img_size);
}


int main(int argc, char **argv)
{
    
    read_parameters(argv[1]);

    std::vector<std::string> v_image_name;
    std::vector<double> v_image_timestamp;
    load_img_from_file(image_path, v_image_name, v_image_timestamp);

    if (v_image_name.size() == 0)
    {
        printf("No image !!!");
        return -1;
    }
    XP::DuoCalibParam duo_calib_param;
    try
    {
        load_asl_calib(cv_K, cv_D, duo_calib_param);
    } catch (...)
    {
        std::cout<<"Load calibration file error"<<std::endl;
        return -1;
    }
    // Create masks based on FOVs computed from intrinsics
    cv::Mat_<uchar> masks;

    float fov;
    if (XP::generate_cam_mask(duo_calib_param.Camera.cv_camK_lr[0],
                              duo_calib_param.Camera.cv_dist_coeff_lr[0],
                              duo_calib_param.Camera.img_size, &masks,
                              &fov))
    {
        std::cout << " fov: " << fov << " deg\n";
    }


    // Load IMU samples to predict OF point locations
    std::list<XP::ImuData> imu_samples;
    if (load_imu_from_file(imu_path, &imu_samples) > 0)
    {
        std::cout << "Load imu data. Enable OF prediciton with gyro\n";
    } else
    {
        std::cout << "Cannot load imu data.\n";
        return -1;
    }
    // Adjust end image index for detection
    if (end_id < 0 || end_id > v_image_name.size())
    {
        end_id = v_image_name.size();
    }

    start_id = std::max(0, start_id);
    // remove all frames before the first IMU data
    while (start_id < end_id && v_image_timestamp[start_id] <= imu_samples.front().time_stamp)
        start_id++;

    XP::FeatureTrackDetector feat_track_detector(
        ft_len, ft_droprate, !not_use_fast,
        uniform_radius, duo_calib_param.Camera.img_size);

    XP::PoseViewer pose_viewer;
    pose_viewer.set_clear_canvas_before_draw(true);

    float prev_time_stamp = 0.0f;
    // load previous image
    std::vector<cv::KeyPoint> pre_image_key_points;
    cv::Mat pre_image_features;
    for (int it_img = start_id; it_img < end_id; ++it_img)
    {

        auto read_img_start = std::chrono::high_resolution_clock::now();
        cv::Mat img_in_raw;
        img_in_raw =
            cv::imread(v_image_name[it_img], CV_LOAD_IMAGE_GRAYSCALE);
        // cv::imshow("test",img_in_raw);
        // cv::waitKey(0);
        // getchar();

        CHECK_EQ(img_in_raw.rows, duo_calib_param.Camera.img_size.height);
        CHECK_EQ(img_in_raw.cols, duo_calib_param.Camera.img_size.width);
        cv::Mat img_in_smooth;
        cv::blur(img_in_raw, img_in_smooth, cv::Size(3, 3));
        if (img_in_smooth.rows == 0)
        {
            std::cout << "Cannot load " << v_image_name[it_img] << std::endl;
            return -1;
        }
        // get timestamp from image file name (s)
        const float time_stamp = v_image_timestamp[it_img];

        std::vector<cv::KeyPoint> key_pnts;
        cv::Mat orb_feat;
        cv::Mat pre_img_in_smooth;

        std::vector<XP::ImuData> imu_meas;

        // Get the imu measurements within prev_time_stamp and time_stamp to
        // compute old_R_new
        imu_meas.reserve(10);
        for (auto it_imu = imu_samples.begin(); it_imu != imu_samples.end();)
        {
            if (it_imu->time_stamp < time_stamp)
            {
                imu_meas.push_back(*it_imu);
                it_imu++;
                imu_samples.pop_front();
            } else
            {
                break;
            }
        }
        std::cout << "imu_meas size = " << imu_meas.size() << std::endl;
        std::cout << "img ts prev -> curr " << prev_time_stamp << " -> "
                << time_stamp << std::endl;
        if (imu_meas.size() > 0)
        {
            std::cout << "imu ts prev -> curr " << imu_meas.front().time_stamp
                    << " -> " << imu_meas.back().time_stamp << std::endl;
        }

        // use optical flow  from the 1st frame
        if (it_img != start_id)
        {
            // CHECK(it_img >= 1);
            std::cout << "pre_image_key_points.size(): "
                    << pre_image_key_points.size() << std::endl;
            const int request_feat_num = max_num_per_grid *
                                         grid_row_num *
                                         grid_col_num;
            feat_track_detector.build_img_pyramids(
                img_in_smooth, XP::FeatureTrackDetector::BUILD_TO_CURR);
            if (imu_meas.size() > 1 || 0)
            {
                // Here we simply the transformation chain to rotation only and
                // assume zero translation
                cv::Matx33f old_R_new;
                XP::XpQuaternion I_new_q_I_old; // The rotation between the new
                                                // {I} and old {I}
                for (size_t i = 1; i < imu_meas.size(); ++i)
                {
                    XP::XpQuaternion q_end;
                    XP::IntegrateQuaternion(
                        imu_meas[i - 1].ang_v, imu_meas[i].ang_v, I_new_q_I_old,
                        imu_meas[i].time_stamp - imu_meas[i - 1].time_stamp,
                        &q_end);
                    I_new_q_I_old = q_end;
                }
                Eigen::Matrix3f I_new_R_I_old =
                    I_new_q_I_old.ToRotationMatrix();
                Eigen::Matrix4f I_T_C = duo_calib_param.Imu.D_T_I.inverse() *
                                        duo_calib_param.Camera.D_T_C_lr[0];
                Eigen::Matrix3f I_R_C = I_T_C.topLeftCorner<3, 3>();
                Eigen::Matrix3f C_new_R_C_old =
                    I_R_C.transpose() * I_new_R_I_old * I_R_C;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        old_R_new(j, i) = C_new_R_C_old(i, j);
                    }
                }


                feat_track_detector.optical_flow_and_detect(
                    masks, pre_image_features, pre_image_key_points,
                    request_feat_num, pyra_level, fast_thresh,
                    &key_pnts, &orb_feat, cv::Vec2f(0, 0), // shift init pixels
                    &duo_calib_param.Camera.cv_camK_lr[0],
                    &duo_calib_param.Camera.cv_dist_coeff_lr[0], &old_R_new);
            } else
            {
                feat_track_detector.optical_flow_and_detect(
                    masks, pre_image_features, pre_image_key_points,
                    request_feat_num, pyra_level, fast_thresh,
                    &key_pnts, &orb_feat);
            }
            feat_track_detector.update_img_pyramids();
            std::cout << "after OF key_pnts.size(): " << key_pnts.size()
                    << " requested # "
                    << max_num_per_grid * grid_row_num *
                           grid_col_num;
        } else
        {
            // first frame
            feat_track_detector.detect(
                img_in_smooth, masks,
                max_num_per_grid * grid_row_num *
                    grid_col_num,
                pyra_level, fast_thresh, &key_pnts, &orb_feat);
            feat_track_detector.build_img_pyramids(
                img_in_smooth, XP::FeatureTrackDetector::BUILD_TO_PREV);
        }

        pre_image_key_points = key_pnts;
        pre_image_features = orb_feat.clone();
        // // show pose
        pose_viewer.displayTo("trajectory");
        cv::waitKey(1);

        prev_time_stamp = time_stamp;
    }
    return 0;
}
