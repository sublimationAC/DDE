

#ifndef FACE_X_FERN_TRAIN_H_
#define FACE_X_FERN_TRAIN_H_

#include<vector>
#include<utility>

#include<opencv2/opencv.hpp>

#include "load_data.hpp"

struct FernTrain
{
	FernTrain(const TrainingParameters &tp);
	void Regress(std::vector<Target_type> *targets,
		cv::Mat pixels_val, cv::Mat pixels_cov);
	Target_type Apply(cv::Mat features)const;
	void ApplyMini(cv::Mat features, std::vector<double> &coeffs_exp, std::vector<double> &coeffs_dis)const;
	void apply_tslt_angle(cv::Mat features, cv::Mat tslt, cv::Mat angle)const;

	void write(cv::FileStorage &fs)const;

	std::vector<double> thresholds_exp, thresholds_dis, thresholds_tslt, thresholds_angle;
	std::vector<std::pair<int, int>> features_index_exp, features_index_dis, features_index_tslt, features_index_angle;
	std::vector<std::vector<double> > outputs_exp, outputs_dis, outputs_tslt, outputs_angle;
	std::vector<std::vector<std::pair<int, double>>> outputs_mini_exp, outputs_mini_dis;
private:
	const TrainingParameters &training_parameters;
};

void write(cv::FileStorage& fs, const std::string&, const FernTrain& f);

#endif