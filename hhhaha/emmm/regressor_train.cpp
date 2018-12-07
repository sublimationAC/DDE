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

cv::Point2f div_3(std::vector<cv::Point2f> &pt) {
	return cv::Point2f((pt[0].x + pt[1].x + pt[2].x) / 3.0, (pt[0].y + pt[1].y + pt[2].y) / 3.0);
}


void get_tri_center(
	std::vector<cv::Vec6f> &triangleList, std::vector<cv::Point2f> &tri_center, cv::Rect &rect) {
	tri_center.resize(triangleList.size());
	std::vector<cv::Point2f> pt(3);
	for (size_t i = 0; i < triangleList.size(); i++)
	{
		cv::Vec6f t = triangleList[i];
		pt[0] = cv::Point2f(t[0], t[1]);
		pt[1] = cv::Point2f(t[2], t[3]);
		pt[2] = cv::Point2f(t[4], t[5]);

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
			tri_center[i] = div_3(pt);
	}
}


int find_nearest_center(cv::Point2f x, std::vector<cv::Point2f> &tri_center) {
	double mi = 1000000;
	int ans = 0;
	for (int i = 0; i < tri_center.size(); i++)
		if (dis_cv_pt(x, tri_center[i]) < mi) mi = dis_cv_pt(x, tri_center[i]), ans = i;
	return ans;
}



void RegressorTrain::Regress(std::vector<cv::Vec6f> &triangleList, cv::Rect &rect, 
	Eigen::MatrixX3i &tri_idx, std::vector<cv::Point2d> &ref_shape, std::vector<Target_type> *targets,
	const std::vector <DataPoint> &training_data, Eigen::MatrixXf &bldshps, std::vector<Eigen::MatrixXf> &arg_exp_land_matrix)
{
	puts("regressing");
	std::vector<cv::Point2f> tri_center;
	get_tri_center(triangleList, tri_center, rect);
	for (int i = 0; i < triangleList.size(); i++) {
		printf("%d %.2f %.2f %.2f %.2f %.2f %.2f\n", i, triangleList[i][0], triangleList[i][1], triangleList[i][2], triangleList[i][3], triangleList[i][4], triangleList[i][5]);
		printf("%d %.2f %.2f\n", i, tri_center[i].x, tri_center[i].y);
		printf("%d %d %d %d\n", i, tri_idx(i, 0), tri_idx(i, 1), tri_idx(i, 2));
		
	}

	cv::Point2f center;
	center.x = rect.x + rect.width / 2; center.y = rect.y + rect.height / 2;
	for (int i = 0; i < training_parameters_.P; ++i)
	{
		printf("+ +%d\n", i);

		double s[4];
		int idx;
		do {
			cv::Point2f temp;
			//temp.x = cv::theRNG().uniform(rect.x, rect.x+rect.width);
			//temp.y = cv::theRNG().uniform(rect.y, rect.y+rect.height);
			temp.x = cv::theRNG().gaussian(rect.width/6);
			temp.y = cv::theRNG().gaussian(rect.height/6);
			temp += center;
			std::cout << rect.x << " " << rect.x+rect.width << " " << rect.y << " " << rect.y+rect.height
				<< " " << center << " " << temp << "\n";

			idx = find_nearest_center(temp, tri_center);
			printf("i %d idx %d\n %d %d %d\n",i, idx, tri_idx(idx, 0), tri_idx(idx, 1), tri_idx(idx, 2));
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

		//std::vector<cv::Point2d> temp(G_land_num);
		//cal_init_2d_land_ang_i(temp, training_data[i], bldshps);

		std::vector<cv::Point2d> temp(G_land_num);
		//cal_init_2d_land_ang_0ide_i(temp, training_data[i],arg_exp_land_matrix[training_data[i].ide_idx]);

		get_init_land_ang_0ide_i(temp, training_data[i], bldshps, arg_exp_land_matrix);

		/*for (int p = 0; p < G_land_num; p++)
			printf("check exp matrix %.10f %.10f \n", temp[p].x - temp___[p].x, temp[p].y - temp___[p].y);*/

		for (int j = 0; j < training_parameters_.P; ++j)
		{
			cv::Point pixel_pos =
				temp[tri_idx(pixels_[j].first, 0)] * pixels_[j].second.x +
				temp[tri_idx(pixels_[j].first, 1)] * pixels_[j].second.y +
				temp[tri_idx(pixels_[j].first, 2)] * (1 - pixels_[j].second.x - pixels_[j].second.y);

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
		printf("inner regressing no:%d\n",i);
		ferns_[i].Regress(targets, pixels_val, pixels_cov);
		for (int j = 0; j < targets->size(); ++j)
		{
			(*targets)[j] = shape_difference((*targets)[j], ferns_[i].Apply(
				pixels_val(cv::Range::all(), cv::Range(j, j + 1))));
		}
		int idx= cv::theRNG().uniform(0, targets->size());
		cout << "-----------\nidx:" << idx << "\n";
		print_target((*targets)[idx]);
	}

	CompressFerns();
}




void RegressorTrain::CompressFerns()
{
	base_exp_.create(G_nShape -1, training_parameters_.Base, CV_64FC1);
	base_dis_.create(2*G_land_num, training_parameters_.Base, CV_64FC1);
	vector<int> rand_index;
	for (int i = 0; i < training_parameters_.K * (1 << training_parameters_.F); ++i)
		rand_index.push_back(i);
	random_shuffle(rand_index.begin(), rand_index.end());
	for (int i = 0; i < training_parameters_.Base; ++i)
	{
		//Target_type output = ferns_[rand_index[i] >> training_parameters_.F]
		//	.outputs[rand_index[i] & ((1 << training_parameters_.F) - 1)];
		std::vector<double> output_exp = ferns_[rand_index[i] >> training_parameters_.F]
			.outputs_exp[rand_index[i] & ((1 << training_parameters_.F) - 1)];

		std::vector<double> output_dis = ferns_[rand_index[i] >> training_parameters_.F]
			.outputs_dis[rand_index[i] & ((1 << training_parameters_.F) - 1)];
		/*for (int j = 0; j < output.size(); ++j)
		{
			base_.at<double>(j * 2, i) = output[j].x;
			base_.at<double>(j * 2 + 1, i) = output[j].y;
		}*/
		
		//Eigen::VectorXf temp_v;
		//target2vector(output, temp_v);
		//for (int j = 0; j < G_target_type_size; j++)
		//	base_.at<double>(j,i) = temp_v(j);
		for (int j = 0; j < output_exp.size(); ++j) base_exp_.at<double>(j, i) = output_exp[j];
		for (int j = 0; j < output_dis.size(); ++j) base_dis_.at<double>(j, i) = output_dis[j];

		//cv::normalize(base_.col(i), base_.col(i));
		cv::normalize(base_exp_.col(i), base_exp_.col(i));
		cv::normalize(base_dis_.col(i), base_dis_.col(i));
	}

	for (int i = 0; i < training_parameters_.K; ++i)
	{
		for (int j = 0; j < (1 << training_parameters_.F); ++j)
		{
			//const Target_type &output = ferns_[i].outputs[j];
			//Eigen::VectorXf temp_v;
			//target2vector(ferns_[i].outputs[j], temp_v);
			const std::vector<double> &output_exp = ferns_[i].outputs_exp[j];
			const std::vector<double> &output_dis = ferns_[i].outputs_dis[j];

			//cv::Mat output_mat(base_.rows, 1, CV_64FC1);
			cv::Mat output_mat_exp(base_exp_.rows, 1, CV_64FC1);
			cv::Mat output_mat_dis(base_dis_.rows, 1, CV_64FC1);

			/*for (int k = 0; k < output.size(); ++k)
			{
				output_mat.at<double>(k * 2) = output[k].x;
				output_mat.at<double>(k * 2 + 1) = output[k].y;
			}*/

			//for (int p = 0; p < G_nShape; p++)
			//	output_mat.at<double>(p) = output.exp(p);

			//for (int p = 0; p < 3; p++)
			//	output_mat.at<double>(G_nShape + p) = output.tslt(p);

			//for (int p = 0; p < 3; p++) for (int q = 0; q < 3; q++)
			//	output_mat.at<double>(G_nShape + 3 + p * 3 + q) = output.rot(p, q);

			//for (int p = 0; p < G_land_num; p++) for (int q = 0; q < 2; q++)
			//	output_mat.at<double>(G_nShape + 3 + 3 * 3 + p * 2 + q) = output.dis(p, q);

			//for (int p = 0; p < G_target_type_size; p++)
			//	output_mat.at<double>(p) = temp_v(p);

			for (int p = 0; p < output_exp.size(); ++p) output_mat_exp.at<double>(p) = output_exp[p];
			for (int p = 0; p < output_dis.size(); ++p) output_mat_dis.at<double>(p) = output_dis[p];

//			ferns_[i].outputs_mini.push_back(OMP(output_mat, base_, training_parameters_.Q));
			ferns_[i].outputs_mini_exp.push_back(OMP(output_mat_exp, base_exp_, training_parameters_.Q));
			ferns_[i].outputs_mini_dis.push_back(OMP(output_mat_dis, base_dis_, training_parameters_.Q));

		}
	}
}

Target_type RegressorTrain::Apply(//const vector<cv::Point2d> &mean_shape, 
	const DataPoint &data, Eigen::MatrixXf &bldshps,Eigen::MatrixX3i &tri_idx, std::vector<Eigen::MatrixXf> &arg_exp_land_matrix) const
{
	cv::Mat pixels_val(1, training_parameters_.P, CV_64FC1);
	//Transform t = Procrustes(data.init_shape, mean_shape);
	//vector<cv::Point2d> offsets(training_parameters_.P);
	//for (int j = 0; j < training_parameters_.P; ++j)
	//	offsets[j] = pixels_[j].second;
	//t.Apply(&offsets, false);

	double *p = pixels_val.ptr<double>(0);
	vector<cv::Point2d> temp(G_land_num);
	//cal_init_2d_land_i(temp, data, bldshps);
	//cal_init_2d_land_ang_i(temp, data, bldshps);
	//std::vector<cv::Point2d> temp___(G_land_num);
	//cal_init_2d_land_ang_0ide_i(temp___, data, exp_matrix);
	
	get_init_land_ang_0ide_i(temp, data, bldshps, arg_exp_land_matrix);

	/*for (int p = 0; p < G_land_num; p++)
		printf("check exp matrix %.10f %.10f \n", temp[p].x - temp___[p].x, temp[p].y - temp___[p].y);*/
	for (int j = 0; j < training_parameters_.P; ++j)
	{
		//cv::Point pixel_pos = data.init_shape[pixels_[j].first] + offsets[j];
		
		cv::Point pixel_pos =
			temp[tri_idx(pixels_[j].first, 0)] * pixels_[j].second.x +
			temp[tri_idx(pixels_[j].first, 1)] * pixels_[j].second.y +
			temp[tri_idx(pixels_[j].first, 2)] * (1 - pixels_[j].second.x - pixels_[j].second.y);
		
		if (pixel_pos.inside(cv::Rect(0, 0, data.image.cols, data.image.rows)))
			p[j] = data.image.at<uchar>(pixel_pos);
		else
			p[j] = 0;
	}

	//vector<double> coeffs(training_parameters_.Base);//initial 0
	vector<double> coeffs_exp(training_parameters_.Base);
	vector<double> coeffs_dis(training_parameters_.Base);

	cv::Mat result_mat_tslt = cv::Mat::zeros(G_tslt_num, 1, CV_64FC1);
	cv::Mat result_mat_angle = cv::Mat::zeros(G_angle_num, 1, CV_64FC1);


	for (int i = 0; i < training_parameters_.K; ++i) {
		ferns_[i].ApplyMini(pixels_val, coeffs_exp, coeffs_dis);
		ferns_[i].apply_tslt_angle(pixels_val, result_mat_tslt, result_mat_angle);
	}

	//cv::Mat result_mat = cv::Mat::zeros(G_target_type_size, 1, CV_64FC1);
	//for (int i = 0; i < training_parameters_.Base; ++i)
	//	result_mat += coeffs[i] * base_.col(i);

	cv::Mat result_mat_exp = cv::Mat::zeros(G_nShape-1, 1, CV_64FC1);
	cv::Mat result_mat_dis = cv::Mat::zeros(2*G_land_num, 1, CV_64FC1);
	for (int i = 0; i < training_parameters_.Base; ++i) {
		result_mat_exp += coeffs_exp[i] * base_exp_.col(i);
		result_mat_dis += coeffs_dis[i] * base_dis_.col(i);
	}



	//vector<cv::Point2d> result(mean_shape.size());
	//for (int i = 0; i < result.size(); ++i)
	//{
	//	result[i].x = result_mat.at<double>(i * 2);
	//	result[i].y = result_mat.at<double>(i * 2 + 1);
	//}

	Target_type result;
	result.dis.resize(G_land_num, 2);
	result.exp.resize(G_nShape);

	result.exp(0) = 0;
	for (int j = 1; j < G_nShape; j++)
		result.exp(j) = result_mat_exp.at<double>(j-1);
		
	for (int j = 0; j < G_land_num; j++) for (int k = 0; k < 2; k++)
		result.dis(j, k) = result_mat_dis.at<double>(j * 2 + k);

	result.tslt.setZero();
	for (int j = 0; j < G_tslt_num; j++)
		result.tslt(j) = result_mat_tslt.at<double>(j);

	for (int j = 0; j < G_angle_num; j++)
		result.angle(j) = result_mat_angle.at<double>(j);
		
	//Eigen::VectorXf temp_v(G_target_type_size);
	//for (int i = 0; i < G_target_type_size; i++)
	//	temp_v(i) = result_mat.at<double>(i);
	//vector2target(temp_v, result);
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
	//fs << "base" << base_;
	fs << "base_exp_" << base_exp_;
	fs << "base_dis_" << base_dis_;
	fs << "}";
}

void write(cv::FileStorage& fs, const string&, const RegressorTrain& r)
{
	r.write(fs);
}
