

#ifndef FACE_X_REGRESSOR_TRAIN_H_
#define FACE_X_REGRESSOR_TRAIN_H_

#include<vector>
#include<utility>
#include<string>

#include<opencv2/opencv.hpp>

//#include "utils_train.h"
#include "fern_train.h"

class RegressorTrain
{
public:
	RegressorTrain(const TrainingParameters &tp);
	void Regress(std::vector<cv::Vec6f> &triangleList, cv::Rect &rect,
		Eigen::MatrixX3i &tri_idx, std::vector<cv::Point2d> &ref_shape, std::vector<Target_type> *targets,
		const std::vector<DataPoint> &training_data, Eigen::MatrixXf &bldshps,std:: vector<Eigen::MatrixXf> &arg_exp_land_matrix) ;

	Target_type Apply(//const std::vector<cv::Point2d> &mean_shape,
		const DataPoint &data, Eigen::MatrixXf &bldshps, Eigen::MatrixX3i &tri_idx, std::vector<Eigen::MatrixXf> &arg_exp_land_matrix) const;

	void write(cv::FileStorage &fs)const;

private:
	std::vector<std::pair<int, cv::Point2d>> pixels_;
	
	std::vector<FernTrain> ferns_;
	cv::Mat base_exp_,base_dis_;
	const TrainingParameters &training_parameters_;

	void CompressFerns();
};

void write(cv::FileStorage& fs, const std::string&, const RegressorTrain& r);

#endif