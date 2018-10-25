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

#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "regressor_train.h"
#include "load_data.hpp"

//#define win64
#define linux
#define changed

#ifdef win64

std::string fwhs_path = "D:/sydney/first/data_me/FaceWarehouse";
std::string lfw_path = "D:/sydney/first/data_me/lfw_image";
std::string gtav_path = "D:/sydney/first/data_me/GTAV_image";
std::string test_path = "D:/sydney/first/data_me/test";
std::string test_path_one = "D:/sydney/first/data_me/test_only_one";
std::string coef_path = "D:/sydney/first/data_me/fitting_coef/ide_fw_p1.lv";


#endif // win64
#ifdef linux
std::string fwhs_path = "/home/weiliu/DDE/cal_coeff/data_me/FaceWarehouse";
std::string lfw_path = "/home/weiliu/DDE/cal_coeff/data_me/lfw_image";
std::string gtav_path = "/home/weiliu/DDE/cal_coeff/data_me/GTAV_image";
std::string test_path = "./test";
std::string test_path_one = "/home/weiliu/DDE/cal_coeff/data_me/test_only_one";
std::string coef_path = "../fitting_coef/ide_fw_p1.lv";
#endif // linux

using namespace std;

TrainingParameters ReadParameters(const string &filename)
{
	/*std::cout << filename << "\n";
	system("pause");*/
	ifstream fin(filename);
	TrainingParameters result;
	if (fin)
	{
		map<string, string> items;
		string line;
		int line_no = 0;
		while (getline(fin, line))
		{
			++line_no;
			line = TrimStr(line);
			if (line.empty() || line[0] == '#')
				continue;

			int colon_pos = line.find(':');
			if (colon_pos == string::npos)
			{
				throw runtime_error("Illegal line " + to_string(line_no) +
					" in config file " + filename);
			}
			
			items[TrimStr(line.substr(0, colon_pos))] = TrimStr(
				line.substr(colon_pos + 1));
		}

		result.training_data_root = items.at("training_data_root");
		result.landmark_count = stoi(items.at("landmark_count"));
		if (result.landmark_count <= 0)
			throw invalid_argument("landmark_count must be positive.");
		result.left_eye_index = stoi(items.at("left_eye_index"));
		if (result.left_eye_index < 0 || result.left_eye_index >= result.landmark_count)
			throw out_of_range("left_eye_index not in range.");
		result.right_eye_index = stoi(items.at("right_eye_index"));
		if (result.right_eye_index < 0 || result.right_eye_index >= result.landmark_count)
			throw out_of_range("right_eye_index not in range.");
		result.output_model_pathname = items.at("output_model_pathname");
		result.T = stoi(items.at("T"));
		if (result.T <= 0)
			throw invalid_argument("T must be positive.");
		result.K = stoi(items.at("K"));
		if (result.K <= 0)
			throw invalid_argument("K must be positive.");
		result.P = stoi(items.at("P"));
		if (result.P <= 0)
			throw invalid_argument("P must be positive.");
		result.Kappa = stod(items.at("Kappa"));
		if (result.Kappa < 0.01 || result.Kappa > 1)
			throw out_of_range("Kappa must be in [0.01, 1].");
		result.F = stoi(items.at("F"));
		if (result.F <= 0)
			throw invalid_argument("F must be positive.");
		result.Beta = stoi(items.at("Beta"));
		if (result.Beta <= 0)
			throw invalid_argument("Beta must be positive.");
		result.TestInitShapeCount = stoi(items.at("TestInitShapeCount"));
		if (result.TestInitShapeCount <= 0)
			throw invalid_argument("TestInitShapeCount must be positive.");
		result.ArgumentDataFactor = stoi(items.at("ArgumentDataFactor"));
		if (result.ArgumentDataFactor <= 0)
			throw invalid_argument("ArgumentDataFactor must be positive.");
		result.Base = stoi(items.at("Base"));
		if (result.Base <= 0)
			throw invalid_argument("Base must be positive.");
		result.Q = stoi(items.at("Q"));
		if (result.Q <= 0)
			throw invalid_argument("Q must be positive.");
	}
	else
		throw runtime_error("Cannot open config file: " + filename);

	return result;
}

vector<DataPoint> GetTrainingData(const TrainingParameters &tp)
{

	vector<DataPoint> result;
#ifdef changed
	/*load_img_land(fwhs_path,".jpg",result);
	load_img_land(lfw_path, ".jpg", result);
	load_img_land(gtav_path, ".bmp", result);*/
	load_img_land_coef(fwhs_path, ".jpg", result);

#else

	const string label_pathname = tp.training_data_root + "/labels.txt";
	ifstream fin(label_pathname);
	/*cout << label_pathname << "\n";
	system("pause");*/
	if (!fin)
		throw runtime_error("Cannot open label file " + label_pathname + " (Pay attention to path separator!)");


	string current_image_pathname;
	int count = 0;
	while (fin >> current_image_pathname)
	{
		cout << current_image_pathname << "\n";

		DataPoint current_data_point;
		current_data_point.image = cv::imread(tp.training_data_root + "/" +
			current_image_pathname);
		//cv::imshow("result", current_data_point.image);
		//cv::waitKey();
		//system("pause");
		if (current_data_point.image.data == nullptr)
			throw runtime_error("Cannot open image file " + current_image_pathname + " (Pay attention to path separator!)");
		int left, right, top, bottom;
		fin >> left >> right >> top >> bottom;
		current_data_point.face_rect =
			cv::Rect(left, top, right - left + 1, bottom - top + 1);

		for (int i = 0; i < tp.landmark_count; ++i)
		{
			cv::Point2d p;
			fin >> p.x >> p.y;
			current_data_point.landmarks.push_back(p);
		}
		std::cout << left << ' ' << right << "\n";
		result.push_back(current_data_point);
		test_data_2dland(current_data_point);
		++count;
	}
#endif // changed

	return result;
}

vector<DataPoint> CreateTestInitShapes(
	const vector<DataPoint> &training_data, const TrainingParameters &tp)
{
	if (tp.TestInitShapeCount > training_data.size())
	{
		throw invalid_argument("TestInitShapeCount is larger than training image count"
			", which is not allowed.");
	}
#ifdef changed
	vector<DataPoint> result;
	set<int> shape_indices;
	while (shape_indices.size() < tp.TestInitShapeCount){
		int rand_index = cv::theRNG().uniform(0, training_data.size());
		shape_indices.insert(rand_index);
	}

	for (auto it = shape_indices.cbegin(); it!=shape_indices.cend();  ++it)
	{
		//vector<cv::Point2d> landmarks = MapShape(training_data[*it].face_rect,
		//	training_data[*it].landmarks, cv::Rect(0, 0, 1, 1));
		result.push_back(training_data[*it]);
	}

#else

	const int kLandmarksSize = training_data[0].landmarks.size();
	cv::Mat all_landmarks(training_data.size(), kLandmarksSize * 2, CV_32FC1);
	for (int i = 0; i < training_data.size(); ++i)
	{
		vector<cv::Point2d> landmarks = MapShape(training_data[i].face_rect,
			training_data[i].landmarks, cv::Rect(0, 0, 1, 1));
		for (int j = 0; j < kLandmarksSize; ++j)
		{
			all_landmarks.at<float>(i, j * 2) = static_cast<float>(landmarks[j].x);
			all_landmarks.at<float>(i, j * 2 + 1) = static_cast<float>(landmarks[j].y);
		}
	}
	cv::Mat labels, centers;
	cv::kmeans(all_landmarks, tp.TestInitShapeCount, labels, 
		cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0), 
		10, cv::KMEANS_RANDOM_CENTERS | cv::KMEANS_PP_CENTERS, centers);

	vector<vector<cv::Point2d>> result;
	for (int i = 0; i < tp.TestInitShapeCount; ++i)
	{
		vector<cv::Point2d> landmarks;
		for (int j = 0; j < kLandmarksSize; ++j)
		{
			landmarks.push_back(cv::Point2d(
				centers.at<float>(i, j * 2), centers.at<float>(i, j * 2 + 1)));
		}
		result.push_back(landmarks);
	}
#endif
	return result;
}

vector<DataPoint> ArgumentData(const vector<DataPoint> &training_data, int factor , Eigen::MatrixXf &bldshps)
{
	if (training_data.size() < 2 * factor)
	{
		throw invalid_argument("You should provide training data with at least "
			"2*ArgumentDataFactor images.");
	}
	vector<DataPoint> result(training_data.size() * G_trn_factor);
	int idx = 0;
	for (int i = 0; i < training_data.size(); ++i)
	{
		aug_rand_rot(training_data,result,idx,i);
		aug_rand_tslt(training_data, result, idx, i);
		aug_rand_exp(training_data, result, idx, i);
		aug_rand_user(training_data, result, idx, i, bldshps);
		aug_rand_f(training_data, result, idx, i, bldshps);
	}
	return result;
}

set<int> rand_df_idx(int which,int border, int num) {
	set<int> result;
	result.clear();
	while (result.size() < num)
	{
		int rand_index = cv::theRNG().uniform(0, border);
		if (rand_index != which)
			result.insert(rand_index);
	}
	return result;
}

void aug_rand_rot(const vector<DataPoint> &traindata, vector<DataPoint> &data,int &idx,int train_idx) {

	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_rot);
	auto it = temp.cbegin();
	for (int i = 0; i < G_rnd_rot; i++, it++) {
		data[idx] = traindata[train_idx];
		data[idx].init_shape.dis = traindata[*it].shape.dis;
		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
		
		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
		Eigen::RowVector3f V[2];
		do {
			do {
				V[0] = traindata[train_idx].shape.rot.row(0);
				for (int j = 0; j < 3; j++)
					V[0](j) += cv::theRNG().uniform(-G_rand_rot_border, G_rand_rot_border);
			} while (V[0].norm <= EPSILON);
			V[0].normalize();
			V[1] = traindata[train_idx].shape.rot.row(1);
			V[1] = V[1] - (V[1].dot(V[0]))*V[0];
		} while (V[1].norm <= EPSILON);
		V[1].normalize();
		data[idx].init_shape.rot.row(0) = V[0];
		data[idx].init_shape.rot.row(1) = V[1];
		data[idx].init_shape.rot.row(2) = V[0].cross(V[1]);
		idx++;
	}
}
void aug_rand_tslt(const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx) {

	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_tslt);
	auto it = temp.cbegin();
	for (int i = 0; i < G_rnd_tslt; i++, it++) {
		data[idx] = traindata[train_idx];
		data[idx].init_shape.dis = traindata[*it].shape.dis;
		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
		data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		for (int j=0;j<3;j++)
			data[idx].init_shape.tslt(j) =traindata[train_idx].shape.tslt(j) + cv::theRNG().uniform(-G_rand_tslt_border, G_rand_tslt_border);
		idx++;
	}
}
void aug_rand_exp(const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx) {

	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
	auto it = temp.cbegin();
	set<int> temp_e = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
	auto it_e = temp_e.cbegin();
	for (int i = 0; i < G_rnd_exp; i++, it++,it_e++) {
		data[idx] = traindata[train_idx];
		data[idx].init_shape.dis = traindata[*it].shape.dis;
		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
		data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		data[idx].init_shape.exp = traindata[*it_e].shape.exp;
//		update_slt();
		idx++;
	}
}
void aug_rand_user(const vector<DataPoint> &traindata, vector<DataPoint> &data, 
	int &idx, int train_idx, Eigen::MatrixXf &bldshps) {

	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_user);
	auto it = temp.cbegin();
	set<int> temp_u = rand_df_idx(train_idx, traindata.size(), G_rnd_user);
	auto it_u = temp_u.cbegin();
	for (int i = 0; i < G_rnd_user; i++, it++, it_u++) {
		data[idx] = traindata[train_idx];
		data[idx].init_shape.dis = traindata[*it].shape.dis;
		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
		data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		data[idx].user = traindata[*it_u].user;
		recal_dis(data[idx],bldshps);
		idx++;
	}
}

void aug_rand_f(const vector<DataPoint> &traindata, vector<DataPoint> &data, 
	int &idx, int train_idx, Eigen::MatrixXf &bldshps) {

	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_user);
	auto it = temp.cbegin();
	for (int i = 0; i < G_rnd_user; i++, it++) {
		data[idx] = traindata[train_idx];
		data[idx].init_shape.dis = traindata[*it].shape.dis;
		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
		data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
#ifdef posit
		data[idx].init_f = traindata[train_idx].f + cv::theRNG().uniform(-G_rand_f_border, G_rand_f_border);
#endif // posit
#ifdef normalization
		data[idx].s(0,0)+= cv::theRNG().uniform(-G_rand_s_border, G_rand_s_border);
		data[idx].s(1,1)+= cv::theRNG().uniform(-G_rand_s_border, G_rand_s_border);
#endif
		recal_dis(data[idx], bldshps);
		idx++;
	}
}




vector<Target_type> ComputeTargets(const vector<DataPoint> &data)
{
	vector<Target_type> result;

	for (const DataPoint& dp : data)
	{
		Target_type error= shape_difference(dp.shape,dp.init_shape);
		result.push_back(error);
	}

	return result;
}


void TrainModel(const vector<DataPoint> &training_data, const TrainingParameters &tp, Eigen::MatrixXf &bldshps)
{
	cout << "Training data count: " << training_data.size() << endl;

	vector<vector<cv::Point2d>> shapes;
	for (const DataPoint &dp : training_data)
		shapes.push_back(dp.landmarks);
	vector<cv::Point2d> ref_shape = MeanShape(shapes, tp);


	vector<DataPoint> test_init_shapes =
		CreateTestInitShapes(training_data, tp);

	vector<DataPoint> argumented_training_data = 
		ArgumentData(training_data, tp.ArgumentDataFactor, bldshps);
	vector<RegressorTrain> stage_regressors(tp.T, RegressorTrain(tp));
	for (int i = 0; i < tp.T; ++i)
	{
		long long s = cv::getTickCount();

		vector<Target_type> targets = 
			ComputeTargets(argumented_training_data);


		stage_regressors[i].Regress(ref_shape, &targets,
			argumented_training_data, bldshps);
		for (DataPoint &dp : argumented_training_data)
		{
			Target_type offset = 
				stage_regressors[i].Apply(dp,bldshps);
			Transform t = Procrustes(dp.init_shape, mean_shape);
			t.Apply(&offset, false);
			dp.init_shape = ShapeAdjustment(dp.init_shape, offset);
		}

		cout << "(^_^) Finish training " << i + 1 << " regressor. Using " 
			<< (cv::getTickCount() - s) / cv::getTickFrequency() 
			<< "s. " << tp.T << " in total." << endl;
		cout << "around " << (tp.T - i - 1)*(cv::getTickCount() - s) / cv::getTickFrequency() / 60
			<< "minutes letf!\n";
	}
	puts("B");
	system("pause");
	cv::FileStorage model_file;
	model_file.open(tp.output_model_pathname, cv::FileStorage::WRITE);
	model_file << "mean_shape" << mean_shape;
	model_file << "test_init_shapes" << "[";
	for (auto it = test_init_shapes.begin(); it != test_init_shapes.end(); ++it)
	{
		model_file << *it;
	}
	model_file << "]";
	model_file << "stage_regressors" << "[";
	for (auto it = stage_regressors.begin(); it != stage_regressors.end(); ++it)
		model_file << *it;
	model_file << "]";
	model_file.release();
}
Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		cout << "Usage: FaceX-Train config.txt" << endl;
		return 0;
	}

	try
	{
		TrainingParameters tp = ReadParameters(argv[1]);
		//cout << tp.Base << ' ' << tp.F << "\n";
		//system("pause");
		cout << "Training begin." << endl;
		load_bldshps(bldshps, bldshps_path);
		vector<DataPoint> training_data = GetTrainingData(tp);
		//system("pause");
		TrainModel(training_data, tp, bldshps);
	}
	catch (const exception &e)
	{
		cout << e.what() << endl;
		return -1;
	}
}
