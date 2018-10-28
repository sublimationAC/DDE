#pragma once
#ifndef FACE_X_REGRESSOR_H_
#define FACE_X_REGRESSOR_H_

#include<vector>
#include<utility>
#include<string>

#include<opencv2/opencv.hpp>

#include "fern_dde.h"


class regressor_dde
{
public:
	Target_type Apply(//const Transform &t,
		const Target_type &data, Eigen::MatrixX3i &tri_idx, DataPoint &ini_data, Eigen::MatrixXf &bldshps) const;

	void read(const cv::FileNode &fn);

private:

	std::vector<std::pair<int, cv::Point2d>> pixels_dde_;
	std::vector<Fern_dde> ferns_dde_;
	cv::Mat base_;
};

void read(const cv::FileNode& node, regressor_dde& r, const regressor_dde&);

#endif