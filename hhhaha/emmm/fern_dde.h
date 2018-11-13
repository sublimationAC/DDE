
#ifndef FACE_X_FERN_DDE_H_
#define FACE_X_FERN_DDE_H_

#include<vector>
#include<utility>
#include "utils_dde_test.hpp"

#include<opencv2/opencv.hpp>

struct Fern_dde
{
	void ApplyMini(cv::Mat features, cv::Mat coeffs)const;

	void read(const cv::FileNode &fn);

	std::vector<double> thresholds;
	std::vector<std::pair<int, int>> features_index;
	std::vector<std::vector<std::pair<int, double>>> outputs_mini;
};

void read(const cv::FileNode& node, Fern_dde& f, const Fern_dde&);

#endif
