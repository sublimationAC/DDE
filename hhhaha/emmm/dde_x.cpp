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

#include "dde_x.h"

#include <algorithm>
#include <stdexcept>


using namespace std;

DDEX::DDEX(const string & filename)
{
	cv::FileStorage model_file;
	model_file.open(filename, cv::FileStorage::READ);
	if (!model_file.isOpened())
		throw runtime_error("Cannot open model file \"" + filename + "\".");

	model_file["ref_shape"] >> ref_shape_;
	cv::FileNode fn;/*= model_file["test_init_shapes"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		vector<cv::Point2d> shape;
		*it >> shape;
		test_init_shapes_.push_back(shape);
	}*/
	fn = model_file["stage_regressors"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		regressor_dde r;
		*it >> r;
		stage_regressors_dde_.push_back(r);
	}
}

vector<cv::Point2d> FaceX::Alignment(cv::Mat image, cv::Rect face_rect) const
{
	vector<vector<double>> all_results(test_init_shapes_[0].size() * 2);
	for (int i = 0; i < test_init_shapes_.size(); ++i)
	{
		vector<cv::Point2d> init_shape = MapShape(cv::Rect(0, 0, 1, 1),
			test_init_shapes_[i], face_rect);
		for (int j = 0; j < stage_regressors_.size(); ++j)
		{
			Transform t = Procrustes(init_shape, mean_shape_);
			vector<cv::Point2d> offset =
				stage_regressors_[j].Apply(t, image, init_shape);
			t.Apply(&offset, false);
			init_shape = ShapeAdjustment(init_shape, offset);
		}

		for (int i = 0; i < init_shape.size(); ++i)
		{
			all_results[i * 2].push_back(init_shape[i].x);
			all_results[i * 2 + 1].push_back(init_shape[i].y);
		}
	}

	vector<cv::Point2d> result(test_init_shapes_[0].size());
	for (int i = 0; i < result.size(); ++i)
	{
		nth_element(all_results[i * 2].begin(),
			all_results[i * 2].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2].end());
		result[i].x = all_results[i * 2][test_init_shapes_.size() / 2];
		nth_element(all_results[i * 2 + 1].begin(),
			all_results[i * 2 + 1].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2 + 1].end());
		result[i].y = all_results[i * 2 + 1][test_init_shapes_.size() / 2];
	}
	return result;
}


void change_nearest(DataPoint &data, Eigen::MatrixXf &bldshps, std::vector<DataPoint> &train_data) {
	float mi = 100000, mi_land = 100000000;
	int idx = 0;
	for (int i = 0; i < train_data.size(); i++) {
		float distance = (data.center - train_data[i].center).norm();
		float distance_land = (data.land_2d - train_data[i].land_2d).norm();
		if (distance < mi || ((fabs(distance - mi) < EPSILON)&&(distance_land < mi_land))) {
			idx = i;
			mi = distance;
			mi_land = distance_land;
		}
	}
	data.shape.rot = train_data[idx].shape.rot;
}
void get_init_shape(std::vector<Target_type> &ans, DataPoint data, std::vector<DataPoint>&train_data) {
	float mi[G_dde_K], mi_land[G_dde_K]; int idx[G_dde_K];
	for (int i = 0; i < G_dde_K; i++) mi[i] = 1000000, mi_land[i] = 100000000, idx[i] = 0;
	for (int i = 0; i < train_data.size(); i++) {
		float distance = (data.center - train_data[i].center).norm();
		float distance_land = (data.land_2d - train_data[i].land_2d).norm();
		for (int j=0;j<G_dde_K;j++)
			if (distance < mi[j] ||
				((fabs(distance - mi[j]) < EPSILON) && (distance_land < mi_land[j]))) {
				for (int k = j + 1; k < G_dde_K; k++) {
					idx[k] = idx[k - 1];
					mi[k] = mi[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi[j] = distance;
				mi_land[j] = distance_land;
				break;
			}
	}
	for (int i = 0; i < G_dde_K; i++)
		ans[i] = train_data[idx[i]].shape;
}

void DDEX::dde(
	DataPoint &data, Eigen::MatrixXf &bldshps,
	Eigen::MatrixX3i &tri_idx, std::vector<DataPoint> &train_data){

	Target_type result;
	result.dis.resize(G_land_num, 2);
	result.dis.setZero();
	result.exp.resize(G_nShape);
	result.exp.setZero();
	result.tslt.setZero();
	result.rot.setZero();

	

	change_nearest(data,bldshps,train_data);
	update_2d_land(data,bldshps);
	//update_slt();

	//find init
	std::vector<Target_type> init_shape(G_dde_K);
	get_init_shape(init_shape, data, train_data);


	for (int i = 0; i < init_shape.size(); ++i)
	{
		//Transform t = Procrustes(initial_landmarks, test_init_shapes_[i]);
		//t.Apply(&init_shape);

		Target_type result_shape = init_shape[i];
		
		for (int j = 0; j < stage_regressors_dde_.size(); ++j)
		{
			//Transform t = Procrustes(init_shape, mean_shape_);
			Target_type offset =
				stage_regressors_dde_[j].Apply(result_shape,tri_idx,data,bldshps);
			//t.Apply(&offset, false);
			result_shape = shape_adjustment(result_shape, offset);
		}


		result.dis.array() += result_shape.dis.array();
		result.exp.array() += result_shape.exp.array();
		result.rot.array() += result_shape.rot.array();
		result.tslt.array() += result_shape.tslt.array();

	}
	result.dis.array() /= G_dde_K;
	result.exp.array() /= G_dde_K;
	result.rot.array() /= G_dde_K;
	result.tslt.array() /= G_dde_K;
	data.shape = result;
}