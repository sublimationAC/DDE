/*
FaceX-Train is a tool to train model file for FaceX, which is an open
source face alignment library.

Copyright(C) 2015  Yang Cao

This program is free software : you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.If not, see <http://www.gnu.org/licenses/>.
*/

#include "regressor_train.h"

#include <utility>
#include <iostream>
#include <memory>
#include <algorithm>

#include "utils_dde.hpp"

using namespace std;

RegressorTrain::RegressorTrain(const TrainingParameters &tp)
	: training_parameters_(tp)
{
	ferns_ = vector<FernTrain>(training_parameters_.K, FernTrain(tp));
	pixels_ = std::vector<std::pair<int, cv::Point2d>>(training_parameters_.P);
}

void get_rti_center(
	std::vector<cv::Vec6f> &triangleList, std::vector<cv::Point2f> &tri_center, Eigen::MatrixX3i &tri_idx,
	const vector<cv::Point2d> &ref_shape, cv::Rect rect) {
	std::vector<cv::Point> pt(3);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		cv::Vec6f t = triangleList[i];
		pt[0] = cv::Point(t[0], t[1]);
		pt[1] = cv::Point(t[2],t[3]);
		pt[2] = cv::Point(t[4],t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			tri_center[i] = (pt[0] + pt[1] + pt[2]) / 3;			
			for (int k = 0; k < ref_shape.size(); k++)
				for (int j = 0; j < 3; j++)
					if (dis_cv_pt(ref_shape[k], pt[j]) < EPSILON) tri_idx(i, j) = k;
		}
	}
}

int find_neat_center(cv::Point2f x, std::vector<cv::Point2f> &tri_center) {
	float mi = 1000000;
	int ans = 0;
	for (int i = 0; i < tri_center.size(); i++)
		if (dis_cv_pt(x, tri_center[i]) < mi) mi = dis_cv_pt(x, tri_center[i]), ans = i;
	return ans;
}





void RegressorTrain::Regress(const vector<cv::Point2d> &ref_shape,
	vector<Target_type> *targets,
	const vector<DataPoint> & training_data, Eigen::MatrixXf &bldshps)
{
	double left = 10000, right = -10000, top = 10000, bottom = -10000;
	for (cv::Point2d landmark : ref_shape) {
		left = std::min(left, landmark.x);
		right = std::max(right, landmark.x);
		top = std::min(top, landmark.y);
		bottom = std::max(bottom, landmark.y);
	}
	std::vector<cv::Vec6f> triangleList;
	cal_del_tri(ref_shape, cv::Rect(left - 10, top - 10, right - left + 21, bottom - top + 21), triangleList);

	std::vector<cv::Point2f> tri_center(triangleList.size());
	tri_idx.resize(triangleList.size(), 3);
	get_rti_center(triangleList,tri_center, tri_idx, ref_shape,
		cv::Rect(left - 10, top - 10, right - left + 21, bottom - top + 21));

	for (int i = 0; i < training_parameters_.P; ++i)
	{
		float s[4];
		int idx;
		do {
			cv::Point2f temp;
			temp.x = cv::theRNG().uniform(left, right);
			temp.y = cv::theRNG().uniform(top, bottom);
			idx = find_neat_center(temp, tri_center);			
			s[0] = cal_cv_area(temp, ref_shape[tri_idx(idx, 2)], ref_shape[tri_idx(idx, 1)]);
			s[1] = cal_cv_area(temp, ref_shape[tri_idx(idx, 0)], ref_shape[tri_idx(idx, 2)]);
			s[2] = cal_cv_area(temp, ref_shape[tri_idx(idx, 0)], ref_shape[tri_idx(idx, 1)]);
			s[3] = cal_cv_area(ref_shape[tri_idx(idx, 2)], ref_shape[tri_idx(idx, 0)], ref_shape[tri_idx(idx, 1)]);
		} while (s[0] + s[1] + s[2] - s[3] > EPSILON);

		for (int j = 0; j < 3; j++) s[j] /= s[3];

		pixels_[i].first = idx;
		pixels_[i].second.x = s[0];
		pixels_[i].second.y = s[1];
	}

	// If you want to use AVX2, you must pay attention to memory alignment.
	// AVX2 is not used by default. You can change Covariance in fern_train.cpp
	// to AVXCovariance to enable it.
	unique_ptr<double[]> pixels_val_data(new double[
		training_parameters_.P * training_data.size() + 3]);//////////////////////?????????????/////////

	cv::Mat pixels_val(training_parameters_.P, training_data.size(), CV_64FC1,
	cv::alignPtr(pixels_val_data.get(), 32));
	for (int i = 0; i < pixels_val.cols; ++i)
	{
		/*Transform t = Procrustes(training_data[i].init_shape, mean_shape);
		vector<cv::Point2d> offsets(training_parameters_.P);
		for (int j = 0; j < training_parameters_.P; ++j)
			offsets[j] = pixels_[j].second;
		t.Apply(&offsets, false);*/
		vector<cv::Point2d> temp(G_land_num);
		cal_init_2d_land_i(temp, training_data[i], bldshps);

		for (int j = 0; j < training_parameters_.P; ++j)
		{
			cv::Point pixel_pos = 
				temp[tri_idx(pixels_[j].first,0)]* pixels_[j].second.x+
				temp[tri_idx(pixels_[j].first, 1)] * pixels_[j].second.y+
				temp[tri_idx(pixels_[j].first, 2)]*(1- pixels_[j].second.x- pixels_[j].second.y);

			if (pixel_pos.inside(cv::Rect(0, 0,
				training_data[i].image.cols, training_data[i].image.rows)))
			{
				pixels_val.at<double>(j, i) =
					training_data[i].image.at<uchar>(pixel_pos);
			}
			else
				pixels_val.at<double>(j, i) = 0;
		}
	}

	cv::Mat pixels_cov, means;
	cv::calcCovarMatrix(pixels_val, pixels_cov, means,
		cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_COLS);

	for (int i = 0; i < training_parameters_.K; ++i)
	{
		ferns_[i].Regress(targets, pixels_val, pixels_cov);
		for (int j = 0; j < targets->size(); ++j)
		{
			(*targets)[j] = shape_difference((*targets)[j], ferns_[i].Apply(
				pixels_val(cv::Range::all(), cv::Range(j, j + 1))));
		}
	}

	CompressFerns();
}

void RegressorTrain::CompressFerns()
{
	base_.create(ferns_[0].outputs[0].size, training_parameters_.Base, CV_64FC1);
	vector<int> rand_index;
	for (int i = 0; i < training_parameters_.K * (1 << training_parameters_.F); ++i)
		rand_index.push_back(i);
	random_shuffle(rand_index.begin(), rand_index.end());
	for (int i = 0; i < training_parameters_.Base; ++i)
	{
		const Target_type &output = ferns_[rand_index[i] >> training_parameters_.F]
			.outputs[rand_index[i] & ((1 << training_parameters_.F) - 1)];
		/*for (int j = 0; j < output.size; ++j)
		{
			base_.at<double>(j * 2, i) = output[j].x;
			base_.at<double>(j * 2 + 1, i) = output[j].y;
		}*/
		for (int j = 0; j < G_nShape; j++)
			base_.at<double>(j, i) = output.exp(j);

		for (int j = 0; j < 3; j++)
			base_.at<double>(G_nShape + j, i) = output.tslt(j);

		for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++)
			base_.at<double>(G_nShape + 3 + j * 3 + k, i) = output.rot(j, k);

		for (int j = 0; j < G_land_num; j++) for (int k = 0; k < 2; k++)
			base_.at<double>(G_nShape + 3 + 3 * 3 + j * 2 + k, i) = output.dis(j, k);
		cv::normalize(base_.col(i), base_.col(i));
	}

	for (int i = 0; i < training_parameters_.K; ++i)
	{
		for (int j = 0; j < (1 << training_parameters_.F); ++j)
		{
			const Target_type &output = ferns_[i].outputs[j];
			cv::Mat output_mat(base_.rows, 1, CV_64FC1);

			/*for (int k = 0; k < output.size; ++k)
			{
				output_mat.at<double>(k * 2) = output[k].x;
				output_mat.at<double>(k * 2 + 1) = output[k].y;
			}*/

			for (int p = 0; p < G_nShape; p++)
				output_mat.at<double>(p) = output.exp(p);

			for (int p = 0; p < 3; p++)
				output_mat.at<double>(G_nShape + p) = output.tslt(p);

			for (int p = 0; p < 3; p++) for (int q = 0; q < 3; q++)
				output_mat.at<double>(G_nShape + 3 + p * 3 + q) = output.rot(p, q);

			for (int p = 0; p < G_land_num; p++) for (int q = 0; q < 2; q++)
				output_mat.at<double>(G_nShape + 3 + 3 * 3 + p * 2 + q) = output.dis(p, q);
			ferns_[i].outputs_mini.push_back(OMP(output_mat, base_, training_parameters_.Q));
		}
	}
}

Target_type RegressorTrain::Apply(//const vector<cv::Point2d> &mean_shape, 
	const DataPoint &data, Eigen::MatrixXf &bldshps) const
{
	cv::Mat pixels_val(1, training_parameters_.P, CV_64FC1);
	//Transform t = Procrustes(data.init_shape, mean_shape);
	//vector<cv::Point2d> offsets(training_parameters_.P);
	//for (int j = 0; j < training_parameters_.P; ++j)
	//	offsets[j] = pixels_[j].second;
	//t.Apply(&offsets, false);

	double *p = pixels_val.ptr<double>(0);
	for (int j = 0; j < training_parameters_.P; ++j)
	{
		//cv::Point pixel_pos = data.init_shape[pixels_[j].first] + offsets[j];
		vector<cv::Point2d> temp(G_land_num);
		cal_init_2d_land_i(temp, data, bldshps);
		cv::Point pixel_pos =
			temp[tri_idx(pixels_[j].first, 0)] * pixels_[j].second.x +
			temp[tri_idx(pixels_[j].first, 1)] * pixels_[j].second.y +
			temp[tri_idx(pixels_[j].first, 2)] * (1-pixels_[j].second.x- pixels_[j].second.y);
		
		if (pixel_pos.inside(cv::Rect(0, 0, data.image.cols, data.image.rows)))
			p[j] = data.image.at<uchar>(pixel_pos);
		else
			p[j] = 0;
	}

	vector<double> coeffs(training_parameters_.Base);
	for (int i = 0; i < training_parameters_.K; ++i)
		ferns_[i].ApplyMini(pixels_val, coeffs);

	cv::Mat result_mat = cv::Mat::zeros(data.shape.size, 1, CV_64FC1);
	for (int i = 0; i < training_parameters_.Base; ++i)
		result_mat += coeffs[i] * base_.col(i);
		
	//vector<cv::Point2d> result(mean_shape.size());
	//for (int i = 0; i < result.size(); ++i)
	//{
	//	result[i].x = result_mat.at<double>(i * 2);
	//	result[i].y = result_mat.at<double>(i * 2 + 1);
	//}

	Target_type result;
	result.dis.resize(G_land_num, 2);
	result.exp.resize(G_nShape);

	return result;
}


void RegressorTrain::write(cv::FileStorage &fs)const
{
	fs << "{";
	fs << "pixels";
	fs << "[";
	for (auto it = pixels_.begin(); it != pixels_.end(); ++it)
		fs << "{" << "first" << it->first << "second" << it->second << "}";
	fs << "]";
	fs << "ferns" << "[";
	for (auto it = ferns_.begin(); it != ferns_.end(); ++it)
		fs << *it;
	fs << "]";
	fs << "base" << base_;
	fs << "}";
}

void write(cv::FileStorage& fs, const string&, const RegressorTrain& r)
{
	r.write(fs);
}
