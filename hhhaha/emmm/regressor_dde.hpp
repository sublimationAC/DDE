#pragma once
#ifndef FACE_X_REGRESSOR_DDE_H_
#define FACE_X_REGRESSOR_DDE_H_

#include<vector>
#include<utility>
#include<string>

#include<opencv2/opencv.hpp>

#include "fern_dde.h"


class regressor_dde
{
public:
	Target_type Apply(//const Transform &t,
		const Target_type &data, Eigen::MatrixX3i &tri_idx, DataPoint &ini_data, Eigen::MatrixXf &bldshps,
		Eigen::MatrixXf &exp_r_t_all_matrix) const;
	void visualize_feature_cddt(//const Transform &t,
		cv::Mat rgb_images, Eigen::MatrixX3i &tri_idx, std::vector<cv::Point2d> &landmarks) const;


	void read(const cv::FileNode &fn);

private:

	std::vector<std::pair<int, cv::Point2d>> pixels_dde_;
	std::vector<Fern_dde> ferns_dde_;
	cv::Mat base_exp_, base_dis_;
};

void read(const cv::FileNode& node, regressor_dde& r, const regressor_dde&);

#endif