
#include "regressor_dde.hpp"

#include <utility>
#include <algorithm>
#include <stdexcept>


using namespace std;

//void cal_2d_land_i_0ide(
//	std::vector<cv::Point2d> &ans, Eigen::MatrixXf &exp_r_t_all_matrix,const Target_type &data,DataPoint &ini_data) {
//	ans.resize(G_land_num);
//	Eigen::RowVector2f T = data.tslt.block(0, 0, 1, 2);
//	Eigen::VectorXf exp = data.exp;
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		Eigen::Vector3f v;
//		for (int axis = 0; axis < 3; axis++)
//			v(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, exp, ini_data.land_cor(i_v), axis);
//		Eigen::RowVector2f temp = ((ini_data.s) * ((data.rot) * v)).transpose() + T + data.dis.row(i_v);
//		ans[i_v].x = temp(0); ans[i_v].y = ini_data.image.rows - temp(1);
//	}
//}

void cal_2d_land_i_ang_0ide(
	std::vector<cv::Point2d> &ans, Eigen::MatrixXf &exp_r_t_all_matrix, const Target_type &data, DataPoint &ini_data) {
	ans.resize(G_land_num);
	Eigen::RowVector2f T = data.tslt.block(0, 0, 1, 2);
	Eigen::Matrix3f rot = get_r_from_angle(data.angle);
	Eigen::VectorXf exp = data.exp;
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, exp, ini_data.land_cor(i_v), axis);
		Eigen::RowVector2f temp = ((ini_data.s) * (rot * v)).transpose() + T + data.dis.row(i_v);
		ans[i_v].x = temp(0); ans[i_v].y = ini_data.image.rows - temp(1);
	}
}

Target_type regressor_dde::Apply(//const Transform &t,
	const Target_type &data, Eigen::MatrixX3i &tri_idx,DataPoint &ini_data,Eigen::MatrixXf &bldshps,
	Eigen::MatrixXf &exp_r_t_all_matrix) const
{
	//for (int i = 0; i < 50; i++) {
	//	printf("%d %d %d %d\n", i, tri_idx(i, 0), tri_idx(i, 1), tri_idx(i, 2));

	//}


	cv::Mat pixels_val(1, pixels_dde_.size(), CV_64FC1);
	/*vector<cv::Point2d> offsets(pixels_dde_.size());
	for (int j = 0; j < pixels_.size(); ++j)
		offsets[j] = pixels_[j].second;
	t.Apply(&offsets, false);*/
	std ::vector<cv::Point2d> temp(G_land_num);
	//cal_2d_land_i(temp, data, bldshps,ini_data);
	cal_2d_land_i_ang_0ide(temp, exp_r_t_all_matrix, data,ini_data);
	
	double *p = pixels_val.ptr<double>(0);
	for (int j = 0; j < pixels_dde_.size(); ++j)
	{
		//cv::Point pixel_pos = init_shape[pixels_[j].first] + offsets[j];

		
		cv::Point pixel_pos =
			temp[tri_idx(pixels_dde_[j].first, 0)] * pixels_dde_[j].second.x +
			temp[tri_idx(pixels_dde_[j].first, 1)] * pixels_dde_[j].second.y +
			temp[tri_idx(pixels_dde_[j].first, 2)] * (1 - pixels_dde_[j].second.x - pixels_dde_[j].second.y);
		if (pixel_pos.inside(cv::Rect(0, 0, ini_data.image.cols, ini_data.image.rows)))
			p[j] = ini_data.image.at<uchar>(pixel_pos);
		else
			p[j] = 0;
	}

	cv::Mat coeffs = cv::Mat::zeros(base_.cols, 1, CV_64FC1);
	for (int i = 0; i < ferns_dde_.size(); ++i) {
		printf("inner regressor %d:\n", i);
		ferns_dde_[i].ApplyMini(pixels_val, coeffs);
	}

	cv::Mat result_mat = base_ * coeffs;

	/*vector<cv::Point2d> result(init_shape.size());
	for (int i = 0; i < result.size(); ++i)
	{
		result[i].x = result_mat.at<double>(i * 2);
		result[i].y = result_mat.at<double>(i * 2 + 1);
	}*/

	Target_type result;
	//result.dis.resize(G_land_num, 2);
	//result.exp.resize(G_nShape);

	//for (int j = 0; j < G_nShape; j++)
	//	result.exp(j) = result_mat.at<double>(j);

	//for (int j = 0; j < 3; j++)
	//	result.tslt(j) = result_mat.at<double>(j + G_nShape);

	//for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++)
	//	result.rot(j, k) = result_mat.at<double>(G_nShape + 3 + j * 3 + k);

	//for (int j = 0; j < G_land_num; j++) for (int k = 0; k < 2; k++)
	//	result.dis(j, k) = result_mat.at<double>(G_nShape + 3 + 3 * 3 + j * 2 + k);
	Eigen::VectorXf temp_v;
	for (int i = 0; i < G_target_type_size; i++) temp_v(i) = result_mat.at<double>(i);
	vector2target(temp_v, result);
	return result;
}

void regressor_dde::read(const cv::FileNode &fn)
{
	pixels_dde_.clear();
	ferns_dde_.clear();
	cv::FileNode pixels_node = fn["pixels"];
	for (auto it = pixels_node.begin(); it != pixels_node.end(); ++it)
	{
		pair<int, cv::Point2d> pixel;
		(*it)["first"] >> pixel.first;
		(*it)["second"] >> pixel.second;
		pixels_dde_.push_back(pixel);
	}
	cv::FileNode ferns_node = fn["ferns"];
	for (auto it = ferns_node.begin(); it != ferns_node.end(); ++it)
	{
		Fern_dde f;
		*it >> f;
		ferns_dde_.push_back(f);
	}
	fn["base"] >> base_;
}

void read(const cv::FileNode& node, regressor_dde& r, const regressor_dde&)
{
	if (node.empty())
		throw runtime_error("Model file is corrupt!");
	else
		r.read(node);
}