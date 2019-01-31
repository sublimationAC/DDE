#pragma once
#include"load_data_test.hpp"
double ceres_post_processing(
	DataPoint &data, Target_type &last_2, Target_type &last_1, Target_type &now, Eigen::MatrixXf &exp_r_t_point_matrix);
void lk_post_processing(
	cv::Mat frame_last, cv::Mat frame_now, DataPoint &data_last, DataPoint &data_now);