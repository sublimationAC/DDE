/*
The MIT License(MIT)

Copyright(c) 2015 Yang Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "fern_dde.h"

#include<iostream>
#include<cstdlib>
#include<memory>
#include<algorithm>


using namespace std;
int get_feature_index(
	int F, cv::Mat features, const std::vector<std::pair<int, int>> &features_index, const std::vector<double> &thresholds) {

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

void Fern_dde::ApplyMini(cv::Mat features, cv::Mat coeffs_exp, cv::Mat coeffs_dis)const
{
	//int outputs_index = 0;
	//for (int i = 0; i < features_index.size(); ++i)
	//{
	//	pair<int, int> feature = features_index[i];
	//	double p1 = features.at<double>(feature.first);
	//	double p2 = features.at<double>(feature.second);
	//	outputs_index |= (p1 - p2 > thresholds[i]) << i;
	//}

	//const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	//for (int i = 0; i < output.size(); ++i)
	//	coeffs.at<double>(output[i].first) += output[i].second;

	int outputs_index_exp = 0, outputs_index_dis = 0;

	outputs_index_exp = get_feature_index(features_index_exp.size(), features, features_index_exp, thresholds_exp);
	outputs_index_dis = get_feature_index(features_index_dis.size(), features, features_index_dis, thresholds_dis);


	//const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	//for (int i = 0; i < training_parameters.Q; ++i)
	//	coeffs[output[i].first] += output[i].second;

	const vector<pair<int, double>> &output_exp = outputs_mini_exp[outputs_index_exp];
	const vector<pair<int, double>> &output_dis = outputs_mini_dis[outputs_index_dis];

	for (int i = 0; i < output_exp.size(); ++i) coeffs_exp.at<double>(output_exp[i].first) += output_exp[i].second;
	for (int i = 0; i < output_dis.size(); ++i) coeffs_dis.at<double>(output_dis[i].first) += output_dis[i].second;
	
}

void Fern_dde::apply_tslt_angle(cv::Mat features, cv::Mat tslt, cv::Mat angle)const {
	int outputs_index_tslt = 0, outputs_index_angle = 0;

	outputs_index_tslt = get_feature_index(features_index_tslt.size(), features, features_index_tslt, thresholds_tslt);
	outputs_index_angle = get_feature_index(features_index_angle.size(), features, features_index_angle, thresholds_angle);


	//const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	//for (int i = 0; i < training_parameters.Q; ++i)
	//	coeffs[output[i].first] += output[i].second;

	//printf("tslt %d %d angle %d %d\n", outputs_index_tslt, outputs_tslt.size(), outputs_index_angle, outputs_angle.size());
	for (int i = 0; i < G_tslt_num; i++) tslt.at<double>(i) += outputs_tslt[outputs_index_tslt][i];//printf("tslt %d %d\n",i, outputs_tslt[outputs_index_tslt].size()),
	for (int i = 0; i < G_angle_num; i++) angle.at<double>(i) += outputs_angle[outputs_index_angle][i];//printf("angle %d %d\n", i, outputs_angle[outputs_index_tslt].size()),

}

void Fern_dde::read(const cv::FileNode &fn)
{
	//thresholds.clear();
	thresholds_exp.clear();
	thresholds_dis.clear();
	thresholds_tslt.clear();
	thresholds_angle.clear();
	//features_index.clear();
	features_index_exp.clear();
	features_index_dis.clear();
	features_index_tslt.clear();
	features_index_angle.clear();
	//outputs_mini.clear();
	outputs_mini_exp.clear();
	outputs_mini_dis.clear();
	outputs_tslt.clear();
	outputs_angle.clear();


	//fn["thresholds"] >> thresholds;
	fn["thresholds_exp"] >> thresholds_exp;
	fn["thresholds_dis"] >> thresholds_dis;
	fn["thresholds_tslt"] >> thresholds_tslt;
	fn["thresholds_angle"] >> thresholds_angle;


	//cv::FileNode features_index_node = fn["features_index"];
	//for (auto it = features_index_node.begin(); it != features_index_node.end(); ++it)
	//{
	//	pair<int, int> feature_index;
	//	(*it)["first"] >> feature_index.first;
	//	(*it)["second"] >> feature_index.second;
	//	features_index.push_back(feature_index);
	//}
	cv::FileNode features_index_node = fn["features_index_exp"];
	for (auto it = features_index_node.begin(); it != features_index_node.end(); ++it)
	{
		pair<int, int> feature_index;
		(*it)["first"] >> feature_index.first;
		(*it)["second"] >> feature_index.second;
		features_index_exp.push_back(feature_index);
	}
	features_index_node = fn["features_index_dis"];
	for (auto it = features_index_node.begin(); it != features_index_node.end(); ++it)
	{
		pair<int, int> feature_index;
		(*it)["first"] >> feature_index.first;
		(*it)["second"] >> feature_index.second;
		features_index_dis.push_back(feature_index);
	}
	features_index_node = fn["features_index_tslt"];
	for (auto it = features_index_node.begin(); it != features_index_node.end(); ++it)
	{
		pair<int, int> feature_index;
		(*it)["first"] >> feature_index.first;
		(*it)["second"] >> feature_index.second;
		features_index_tslt.push_back(feature_index);
	}
	features_index_node = fn["features_index_angle"];
	for (auto it = features_index_node.begin(); it != features_index_node.end(); ++it)
	{
		pair<int, int> feature_index;
		(*it)["first"] >> feature_index.first;
		(*it)["second"] >> feature_index.second;
		features_index_angle.push_back(feature_index);
	}


	//cv::FileNode outputs_mini_node = fn["outputs_mini"];
	//for (auto it = outputs_mini_node.begin(); it != outputs_mini_node.end(); ++it)
	//{
	//	vector<std::pair<int, double>> output;
	//	cv::FileNode output_node = *it;
	//	for (auto it2 = output_node.begin(); it2 != output_node.end(); ++it2)
	//		output.push_back(make_pair((*it2)["index"], (*it2)["coeff"]));
	//	outputs_mini.push_back(output);
	//}

	cv::FileNode outputs_mini_node = fn["outputs_mini_exp"];
	for (auto it = outputs_mini_node.begin(); it != outputs_mini_node.end(); ++it)
	{
		vector<std::pair<int, double>> output;
		cv::FileNode output_node = *it;
		for (auto it2 = output_node.begin(); it2 != output_node.end(); ++it2)
			output.push_back(make_pair((*it2)["index"], (*it2)["coeff"]));
		outputs_mini_exp.push_back(output);
	}
	outputs_mini_node = fn["outputs_mini_dis"];
	for (auto it = outputs_mini_node.begin(); it != outputs_mini_node.end(); ++it)
	{
		vector<std::pair<int, double>> output;
		cv::FileNode output_node = *it;
		for (auto it2 = output_node.begin(); it2 != output_node.end(); ++it2)
			output.push_back(make_pair((*it2)["index"], (*it2)["coeff"]));
		outputs_mini_dis.push_back(output);
	}
	cv::FileNode outputs_node = fn["outputs_tslt"];
	for (auto it = outputs_node.begin(); it != outputs_node.end(); ++it)
	{
		vector<double> output;
		cv::FileNode output_node = *it;
		//output_node = *(output_node.begin());
		for (auto it2 = output_node.begin(); it2 != output_node.end(); ++it2)
			output.push_back((*it2));
		outputs_tslt.push_back(output);
	}
	outputs_node = fn["outputs_angle"];
	for (auto it = outputs_node.begin(); it != outputs_node.end(); ++it)
	{
		vector<double> output;
		cv::FileNode output_node = *it;
		//output_node = *(output_node.begin());
		for (auto it2 = output_node.begin(); it2 != output_node.end(); ++it2)
			output.push_back((*it2));
		outputs_angle.push_back(output);
	}


}

void read(const cv::FileNode& node, Fern_dde &f, const Fern_dde&)
{
	if (node.empty())
		throw runtime_error("Model file is corrupt!");
	else
		f.read(node);
}

void Fern_dde::visualize_feature_cddt(//const Transform &t,
	cv::Mat rgb_images, Eigen::MatrixX3i &tri_idx, std::vector<cv::Point> &pixel_positions) const {

	cv::Mat images = rgb_images.clone();
	for (int i = 0; i < features_index_exp.size(); ++i)
	{
		pair<int, int> feature = features_index_exp[i];
		cv::circle(images, pixel_positions[feature.first], 0.1, cv::Scalar(0, 0, 255), 2);
		cv::circle(images, pixel_positions[feature.second], 0.1, cv::Scalar(255, 0, 0), 2);
	}
	cv::imshow("feature point candidate", images);
	cv::waitKey();
}