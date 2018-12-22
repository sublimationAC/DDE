
#ifndef FACE_X_FERN_DDE_H_
#define FACE_X_FERN_DDE_H_

#include<vector>
#include<utility>
#include "utils_dde_test.hpp"

#include<opencv2/opencv.hpp>

struct Fern_dde
{
	void ApplyMini(cv::Mat features, cv::Mat coeffs_exp, cv::Mat coeffs_dis)const;
	void apply_tslt_angle(cv::Mat features, cv::Mat tslt, cv::Mat angle)const;
	void visualize_feature_cddt(//const Transform &t,
		cv::Mat rgb_images, Eigen::MatrixX3i &tri_idx, std::vector<cv::Point> &pixel_positions) const;

	void read(const cv::FileNode &fn);

	std::vector<double> thresholds_exp, thresholds_dis, thresholds_tslt, thresholds_angle;
	std::vector<std::pair<int, int>> features_index_exp, features_index_dis, features_index_tslt, features_index_angle;
	std::vector<std::vector<std::pair<int, double>>> outputs_mini_exp, outputs_mini_dis;
	std::vector<std::vector<double> > outputs_tslt, outputs_angle;
};

void read(const cv::FileNode& node, Fern_dde& f, const Fern_dde&);

#endif
