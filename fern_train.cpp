#include "fern_train.h"

#include<iostream>
#include<cstdlib>
#include<memory>
#include<algorithm>

using namespace std;

FernTrain::FernTrain(const TrainingParameters &tp) : training_parameters(tp)
{
}

void cal_Y_pixels_cov_proj_cov(cv::Mat Y,cv::Mat pixels_val, vector<double> &Y_pixels_cov, double &Y_proj_cov) {
	cv::Mat projection(Y.cols, 1, CV_64FC1);
	cv::theRNG().fill(projection, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(1));
	unique_ptr<double[]> Y_proj_data(new double[Y.rows + 3]);
	cv::Mat Y_proj(Y.rows, 1, CV_64FC1, cv::alignPtr(Y_proj_data.get(), 32));
	static_cast<cv::Mat>(Y * projection).copyTo(Y_proj);
	Y_proj_cov = Covariance(Y_proj.ptr<double>(0),
		Y_proj.ptr<double>(0), Y_proj.total());	// Use AVXCovariance if you want.

	for (int j = 0; j < pixels_val.rows; ++j)
	{
		Y_pixels_cov[j] = Covariance(Y_proj.ptr<double>(0),
			pixels_val.ptr<double>(j), Y_proj.total());	// Use AVXCovariance if you want.
	}
}




void get_feature_idx_thresthod(
	cv::Mat pixels_val, vector<double> &Y_pixels_cov, double Y_proj_cov, cv::Mat pixels_cov,
	std::vector<std::pair<int, int>> &features_index, std::vector<double> &thresholds,int i) {

	double max_corr = -1;
	for (int j = 0; j < pixels_val.rows; ++j)
	{
		for (int k = 0; k < pixels_val.rows; ++k)
		{
			double corr = (Y_pixels_cov[j] - Y_pixels_cov[k]) / sqrt(
				Y_proj_cov * (pixels_cov.at<double>(j, j)
					+ pixels_cov.at<double>(k, k)
					- 2 * pixels_cov.at<double>(j, k)));
			bool fl = 0;
			for (int p = 0; p < i; p++)
				if ((features_index[p].first == j && features_index[p].second == k) ||
					(features_index[p].first == k && features_index[p].second == j)) {
					fl = 1;
					break;
				}
			if (fl) continue;
			if (corr > max_corr)
			{
				max_corr = corr;
				features_index[i].first = j;
				features_index[i].second = k;
			}
		}
	}

	double threshold_max = -1000000;
	double threshold_min = 1000000;
	for (int j = 0; j < pixels_val.cols; ++j)
	{
		double val = pixels_val.at<double>(features_index[i].first, j)
			- pixels_val.at<double>(features_index[i].second, j);
		threshold_max = max(threshold_max, val);
		threshold_min = min(threshold_min, val);
	}
	thresholds[i] = (threshold_max + threshold_min) / 2
		+ cv::theRNG().uniform(-(threshold_max - threshold_min) * 0.1,
		(threshold_max - threshold_min) * 0.1);
}

void cal_outputs(
	std::vector<std::vector<double> > &outputs,vector<Target_type> *targets,int F,int Beta, cv::Mat pixels_val,
	std::vector<std::pair<int, int>> &features_index, std::vector<double> &thresholds,char which) {

	int outputs_count = 1 << F;
	vector<int> each_output_count(outputs_count);

	for (int i = 0; i < targets->size(); ++i)
	{
		int mask = 0;
		for (int j = 0; j < F; ++j)
		{
			double p1 = pixels_val.at<double>(features_index[j].first, i);
			double p2 = pixels_val.at<double>(features_index[j].second, i);
			mask |= (p1 - p2 > thresholds[j]) << j;
		}
		outputs[mask] = shape_adjustment(outputs[mask], (*targets)[i], which);
		++each_output_count[mask];
	}

	for (int i = 0; i < outputs_count; ++i)
	{
		/*for (cv::Point2d &p : outputs[i])
			p *= 1.0 / (each_output_count[i] + training_parameters.Beta);*/
		for (int j = 0; j < outputs[i].size(); j++)
			outputs[i][j] *= 1.0 / (each_output_count[i] + Beta);
	}
}



void FernTrain::Regress(vector<Target_type> *targets,
	cv::Mat pixels_val, cv::Mat pixels_cov)
{


	cv::Mat Y_exp(targets->size(), G_nShape-1, CV_64FC1);
	cv::Mat Y_dis(targets->size(), 2*G_land_num, CV_64FC1);
	cv::Mat Y_tslt(targets->size(), G_tslt_num, CV_64FC1);
	cv::Mat Y_angle(targets->size(), G_angle_num, CV_64FC1);
	for (int i = 0; i < targets->size(); ++i)
	{
		/*for (int j = 0; j < Y.cols; j += 2)
		{
			Y.at<double>(i, j) = (*targets)[i][j / 2].x;
			Y.at<double>(i, j + 1) = (*targets)[i][j / 2].y;
		}*/

		for (int j = 1; j < G_nShape; j++)
			Y_exp.at<double>(i, j - 1) = (*targets)[i].exp(j);
		
		for (int j = 0; j < G_land_num; j++) for (int k = 0; k < 2; k++)
			Y_dis.at<double>(i, j * 2 + k) = (*targets)[i].dis(j, k);

		for (int j = 0; j < G_tslt_num; j++)
			Y_tslt.at<double>(i, j) = (*targets)[i].tslt(j);

		for (int j=0;j< G_angle_num;j++)
			Y_angle.at<double>(i, j) = (*targets)[i].angle(j);
					

		//Eigen::VectorXf temp_v;
		//target2vector((*targets)[i], temp_v);
		//for (int j = 0; j < G_target_type_size; j++)
		//	Y.at<double>(i, j) = temp_v(j);

	}


	features_index_exp.assign(training_parameters.F, pair<int, int>());
	features_index_dis.assign(training_parameters.F, pair<int, int>());
	features_index_tslt.assign(training_parameters.F, pair<int, int>());
	features_index_angle.assign(training_parameters.F, pair<int, int>());
	thresholds_exp.assign(training_parameters.F, 0);
	thresholds_dis.assign(training_parameters.F, 0);
	thresholds_tslt.assign(training_parameters.F, 0);
	thresholds_angle.assign(training_parameters.F, 0);

	//for (int i = 0; i <= 20; i++) {
	//	printf("line %d :", i);
	//	for (int j = 0; j < 100; j++)
	//		std::cout << Y.at<double>(i, j);
	//}

	for (int i = 0; i < training_parameters.F; ++i)
	{
		//cv::Mat projection_exp(Y_exp.cols, 1, CV_64FC1);
		//cv::Mat projection_dis(Y_dis.cols, 1, CV_64FC1);
		//cv::Mat projection_tslt(Y_tslt.cols, 1, CV_64FC1);
		//cv::Mat projection_angle(Y_angle.cols, 1, CV_64FC1);

		double Y_exp_proj_cov = 0, Y_dis_proj_cov = 0, Y_tslt_proj_cov = 0, Y_angle_proj_cov = 0;
		std::vector<double> 
			Y_exp_pixels_cov(pixels_val.rows), Y_dis_pixels_cov(pixels_val.rows), 
			Y_tslt_pixels_cov(pixels_val.rows), Y_angle_pixels_cov(pixels_val.rows);

		cal_Y_pixels_cov_proj_cov(Y_exp, pixels_val, Y_exp_pixels_cov, Y_exp_proj_cov);
		cal_Y_pixels_cov_proj_cov(Y_dis, pixels_val, Y_dis_pixels_cov, Y_dis_proj_cov);
		cal_Y_pixels_cov_proj_cov(Y_tslt, pixels_val, Y_tslt_pixels_cov, Y_tslt_proj_cov);
		cal_Y_pixels_cov_proj_cov(Y_angle, pixels_val, Y_angle_pixels_cov, Y_angle_proj_cov);

		get_feature_idx_thresthod(pixels_val, Y_exp_pixels_cov, Y_exp_proj_cov, pixels_cov, features_index_exp, thresholds_exp, i);
		get_feature_idx_thresthod(pixels_val, Y_dis_pixels_cov, Y_dis_proj_cov, pixels_cov, features_index_dis, thresholds_dis, i);
		get_feature_idx_thresthod(pixels_val, Y_tslt_pixels_cov, Y_tslt_proj_cov, pixels_cov, features_index_tslt, thresholds_tslt, i);
		get_feature_idx_thresthod(pixels_val, Y_angle_pixels_cov, Y_angle_proj_cov, pixels_cov, features_index_angle, thresholds_angle, i);

	}

	int outputs_count = 1 << training_parameters.F;

	outputs_exp.assign(outputs_count, vector<double>(G_nShape-1));
	outputs_dis.assign(outputs_count, vector<double>(2*G_land_num));
	outputs_tslt.assign(outputs_count, vector<double>(3));
	outputs_angle.assign(outputs_count, vector<double>(G_angle_num));

	//vector<int> each_output_count_exp(outputs_count), each_output_count_dis(outputs_count), each_output_count_tslt(outputs_count), each_output_count_angle(outputs_count);

	cal_outputs(outputs_exp, targets, training_parameters.F, training_parameters.Beta, pixels_val, features_index_exp, thresholds_exp,'e');
	cal_outputs(outputs_dis, targets, training_parameters.F, training_parameters.Beta, pixels_val, features_index_dis, thresholds_dis,'d');
	cal_outputs(outputs_tslt, targets, training_parameters.F, training_parameters.Beta, pixels_val, features_index_tslt, thresholds_tslt,'t');
	cal_outputs(outputs_angle, targets, training_parameters.F, training_parameters.Beta, pixels_val, features_index_angle, thresholds_angle,'a');

	
}

int get_feature_index(
	int F, cv::Mat features,const std::vector<std::pair<int, int>> &features_index, const std::vector<double> &thresholds) {

	int outputs_index = 0;
	for (int i = 0; i < F; ++i)
	{
		pair<int, int> feature = features_index[i];
		double p1 = features.at<double>(feature.first);
		double p2 = features.at<double>(feature.second);
		outputs_index |= (p1 - p2 > thresholds[i]) << i;
	}
	return outputs_index;
}

Target_type FernTrain::Apply(cv::Mat features)const
{
	Target_type result;

	int feature_index_exp = 0, feature_index_dis = 0, feature_index_tslt = 0, feature_index_angle = 0;

	feature_index_exp=get_feature_index(training_parameters.F, features, features_index_exp, thresholds_exp);
	feature_index_dis=get_feature_index(training_parameters.F, features, features_index_dis, thresholds_dis);
	feature_index_tslt=get_feature_index(training_parameters.F, features, features_index_tslt, thresholds_tslt);
	feature_index_angle=get_feature_index(training_parameters.F, features, features_index_angle, thresholds_angle);

	result.exp.resize(G_nShape);
	result.exp(0) = 0;
	for (int i = 1; i < G_nShape; i++)
		result.exp(i) = outputs_exp[feature_index_exp][i - 1];

	result.dis.resize(G_land_num, 2);
	for (int j = 0; j < G_land_num; j++) for (int k = 0; k < 2; k++)
		result.dis(j, k) = outputs_dis[feature_index_dis][j * 2 + k];

	result.tslt.setZero();
	for (int i = 0; i < G_tslt_num; i++)
		result.tslt(i) = outputs_tslt[feature_index_tslt][i];

	for (int i = 0; i < G_angle_num; i++)
		result.angle(i) = outputs_angle[feature_index_angle][i];

	return result;
}

void FernTrain::ApplyMini(cv::Mat features, std::vector<double> &coeffs_exp, std::vector<double> &coeffs_dis)const
{
	//int outputs_index = 0;
	//for (int i = 0; i < training_parameters.F; ++i)
	//{
	//	pair<int, int> feature = features_index[i];
	//	double p1 = features.at<double>(feature.first);
	//	double p2 = features.at<double>(feature.second);
	//	outputs_index |= (p1 - p2 > thresholds[i]) << i;
	//}

	int outputs_index_exp = 0, outputs_index_dis = 0;

	outputs_index_exp = get_feature_index(training_parameters.F, features, features_index_exp, thresholds_exp);
	outputs_index_dis = get_feature_index(training_parameters.F, features, features_index_dis, thresholds_dis);


	//const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	//for (int i = 0; i < training_parameters.Q; ++i)
	//	coeffs[output[i].first] += output[i].second;

	const vector<pair<int, double>> &output_exp = outputs_mini_exp[outputs_index_exp];
	const vector<pair<int, double>> &output_dis = outputs_mini_dis[outputs_index_dis];

	for (int i = 0; i < training_parameters.Q; ++i) {
		coeffs_exp[output_exp[i].first] += output_exp[i].second;
		coeffs_dis[output_dis[i].first] += output_dis[i].second;
	}
}

void FernTrain::apply_tslt_angle(cv::Mat features, cv::Mat tslt, cv::Mat angle)const {
	int outputs_index_tslt = 0, outputs_index_angle = 0;

	outputs_index_tslt = get_feature_index(training_parameters.F, features, features_index_tslt, thresholds_tslt);
	outputs_index_angle = get_feature_index(training_parameters.F, features, features_index_angle, thresholds_angle);


	//const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	//for (int i = 0; i < training_parameters.Q; ++i)
	//	coeffs[output[i].first] += output[i].second;

	for (int i = 0; i < G_tslt_num; i++) tslt.at<double>(i) += outputs_tslt[outputs_index_tslt][i];
	for (int i = 0; i < G_angle_num; i++) angle.at<double>(i) += outputs_angle[outputs_index_angle][i];

}



void FernTrain::write(cv::FileStorage &fs)const
{
	fs << "{";
	//fs << "thresholds" << thresholds;
	fs << "thresholds_exp" << thresholds_exp;
	fs << "thresholds_dis" << thresholds_dis;
	fs << "thresholds_tslt" << thresholds_tslt;
	fs << "thresholds_angle" << thresholds_angle;
	//fs << "features_index";
	//fs << "[";
	//for (auto it = features_index.begin(); it != features_index.end(); ++it)
	//	fs << "{" << "first" << it->first << "second" << it->second << "}";
	//fs << "]";

	fs << "features_index_exp";
	fs << "[";
	for (auto it = features_index_exp.begin(); it != features_index_exp.end(); ++it)
		fs << "{" << "first" << it->first << "second" << it->second << "}";
	fs << "]";
	fs << "features_index_dis";
	fs << "[";
	for (auto it = features_index_dis.begin(); it != features_index_dis.end(); ++it)
		fs << "{" << "first" << it->first << "second" << it->second << "}";
	fs << "]";
	fs << "features_index_tslt";
	fs << "[";
	for (auto it = features_index_tslt.begin(); it != features_index_tslt.end(); ++it)
		fs << "{" << "first" << it->first << "second" << it->second << "}";
	fs << "]";
	fs << "features_index_angle";
	fs << "[";
	for (auto it = features_index_angle.begin(); it != features_index_angle.end(); ++it)
		fs << "{" << "first" << it->first << "second" << it->second << "}";
	fs << "]";


	/*fs << "outputs_mini";
	fs << "[";
	for (const auto &output: outputs_mini)
	{
		fs << "[";
		for (int i = 0; i < training_parameters.Q; ++i)
		{
			fs << "{" << "index" << output[i].first <<
				"coeff" << output[i].second << "}";
		}
		fs << "]";
	}
	fs << "]";*/

	fs << "outputs_mini_exp";
	fs << "[";
	for (const auto &output : outputs_mini_exp)
	{
		fs << "[";
		for (int i = 0; i < training_parameters.Q; ++i)
		{
			fs << "{" << "index" << output[i].first <<
				"coeff" << output[i].second << "}";
		}
		fs << "]";
	}
	fs << "]";
	fs << "outputs_mini_dis";
	fs << "[";
	for (const auto &output : outputs_mini_dis)
	{
		fs << "[";
		for (int i = 0; i < training_parameters.Q; ++i)
		{
			fs << "{" << "index" << output[i].first <<
				"coeff" << output[i].second << "}";
		}
		fs << "]";
	}
	fs << "]";
	fs << "outputs_tslt";
	fs << "[";
	for (const auto &output : outputs_tslt)
	{
//		fs << "[";
		fs << output;
//		fs << "]";
	}
	fs << "]";
	fs << "outputs_angle";
	fs << "[";
	for (const auto &output : outputs_angle)
	{
//		fs << "[";
		fs << output;
//		fs << "]";
	}
	fs << "]";



	fs << "}";
}

void write(cv::FileStorage& fs, const string&, const FernTrain &f)
{
	f.write(fs);
}