#pragma once

#ifndef DDE_X_H_
#define DDE_X_H_

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "regressor_dde.hpp"

extern cv::Mat G_debug_up_image;

class DDEX
{
public:
	// Construct the object and load model from file.
	//
	// filename: The file name of the model file.
	//
	// Throw runtime_error if the model file cannot be opened.
	DDEX(const std::string &filename);


	void dde(
		cv::Mat debug_init_img, DataPoint &data, Eigen::MatrixXf &bldshps,
		Eigen::MatrixX3i &tri_idx, std::vector<DataPoint> &train_data, Eigen::VectorXi &jaw_land_corr,
		std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,Eigen::MatrixXf &exp_r_t_all_matrix) const;

	void dde_onlyexpdis(
		cv::Mat debug_init_img, DataPoint &data, Eigen::MatrixXf &bldshps,
		Eigen::MatrixX3i &tri_idx, std::vector<DataPoint> &train_data, Eigen::VectorXi &jaw_land_corr,
		std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::MatrixXf &exp_r_t_all_matrix)const;


	void visualize_feature_cddt(cv::Mat rbg_image, Eigen::MatrixX3i &tri_idx, std::vector<cv::Point2d> &landmarks)const;


	// Return how many landmarks the model provides for a face.
	//int landmarks_count() const
	//{
	//	return mean_shape_.size();
	//}
	std::vector<cv::Point2d> get_ref_shape() {
		return ref_shape_;
	}
private:
	std::vector<cv::Point2d> ref_shape_;
	//std::vector<std::vector<cv::Point2d>> test_init_shapes_;
	std::vector<regressor_dde> stage_regressors_dde_;
};

#endif