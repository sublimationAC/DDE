
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
//#include "load_data.hpp"

#define win64
//#define linux
#define changed

#ifdef win64

std::string fwhs_path = "D:/sydney/first/data_me/test_lv/fw";
std::string lfw_path = "D:/sydney/first/data_me/test_lv/lfw_image";
std::string gtav_path = "D:/sydney/first/data_me/test_lv/GTAV_image";
std::string test_path = "D:/sydney/first/data_me/test";
std::string test_path_one = "D:/sydney/first/data_me/test_only_one";
std::string coef_path = "D:/sydney/first/data_me/fitting_coef/ide_fw_p1.lv";
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";

#endif // win64
#ifdef linux
std::string fwhs_path = "/home/weiliu/fitting_dde/fitting_coef/fw";
std::string lfw_path = "/home/weiliu/fitting_dde/fitting_coef/lfw_image";
std::string gtav_path = "/home/weiliu/fitting_dde/fitting_coef/GTAV_image";
std::string test_path = "./test";
std::string test_path_one = "/home/weiliu/DDE/cal_coeff/data_me/test_only_one";
//std::string fwhs_path_p1 = "/home/weiliu/fitting_dde/1cal/data_me/fw_p1";
std::string fwhs_path_p1 = "D:/sydney/first/data_me/test_lv";
std::string fwhs_path_p2 = "/home/weiliu/fitting_dde/2cal/data_me/fw_p2";
std::string fwhs_path_p3 = "/home/weiliu/fitting_dde/3cal/data_me/fw_p3";
std::string fwhs_path_p4 = "/home/weiliu/fitting_dde/4cal/data_me/fw_p4";
std::string fwhs_path_p5 = "/home/weiliu/fitting_dde/5cal/data_me/fw_p5";
std::string bldshps_path = "/home/weiliu/fitting_dde/cal/deal_data/blendshape_ide_svd_77.lv";

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

		result.left_eye_index_x = stoi(items.at("left_eye_index_x"));
		result.left_eye_index_y = stoi(items.at("left_eye_index_y"));
		if (result.left_eye_index_x < 0 || result.left_eye_index_x >= result.landmark_count)
			throw out_of_range("left_eye_index_x not in range.");
		if (result.left_eye_index_y < 0 || result.left_eye_index_y >= result.landmark_count)
			throw out_of_range("left_eye_index_y not in range.");
		result.right_eye_index_x = stoi(items.at("right_eye_index_x"));
		result.right_eye_index_y = stoi(items.at("right_eye_index_y"));
		if (result.right_eye_index_x < 0 || result.right_eye_index_x >= result.landmark_count)
			throw out_of_range("right_eye_index_x not in range.");
		if (result.right_eye_index_y < 0 || result.right_eye_index_y >= result.landmark_count)
			throw out_of_range("right_eye_index_y not in range.");

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
	/*load_img_land_coef(fwhs_path,".jpg",result);
	load_img_land_coef(lfw_path, ".jpg", result);
	load_img_land_coef(gtav_path, ".bmp", result);*/
	load_img_land_coef(fwhs_path, ".jpg", result);

	return result;
}

void train_data_normalize(vector<DataPoint> &training_data) {
	puts("normalizing face rect....!");
	for (auto data : training_data) {
//		cv::imshow("ini", data.image);
		//cv::waitKey(0);
		normalize_gauss_face_rect(data.image, data.face_rect);
//		cv::rectangle(data.image, data.face_rect, 0);
		//cv::imshow("after_norm", data.image);
		//cv::waitKey(0);
	}
}

set<int> rand_df_idx(int which, int border, int num) {
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

void aug_rand_rot(const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx) {

	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_rot);
	auto it = temp.cbegin();
	for (int i = 0; i < G_rnd_rot; i++, it++) {
		data[idx] = traindata[train_idx];
		data[idx].init_shape.dis = traindata[*it].shape.dis;
		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
//----------------------------------------------------------------------------------------------
		/*Eigen::RowVector3f V[2];
		do {
			do {
				V[0] = traindata[train_idx].shape.rot.row(0);
				for (int j = 0; j < 3; j++)
					V[0](j) += cv::theRNG().uniform(-G_rand_rot_border, G_rand_rot_border);
			} while (V[0].norm() <= EPSILON);
			V[0].normalize();
			V[1] = traindata[train_idx].shape.rot.row(1);
			V[1] = V[1] - (V[1].dot(V[0]))*V[0];
		} while (V[1].norm() <= EPSILON);
		V[1].normalize();
		data[idx].init_shape.rot.row(0) = V[0];
		data[idx].init_shape.rot.row(1) = V[1];
		data[idx].init_shape.rot.row(2) = V[0].cross(V[1]);*/
//-----------------------------------------------------------------------------------------------------
		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
		data[idx].ide_idx = train_idx;
		for (int j=0;j<3;j++)
			data[idx].init_shape.angle(j)+= cv::theRNG().uniform(-G_rand_angle_border, G_rand_angle_border);
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
		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
		data[idx].ide_idx = train_idx;
		for (int j = 0; j < 2; j++)
			data[idx].init_shape.tslt(j) = traindata[train_idx].shape.tslt(j) + cv::theRNG().uniform(-G_rand_tslt_border, G_rand_tslt_border);
		data[idx].init_shape.tslt(2) = 0;
		idx++;
	}
}
void aug_rand_exp(const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx) {

	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
	auto it = temp.cbegin();
	set<int> temp_e = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
	auto it_e = temp_e.cbegin();
	for (int i = 0; i < G_rnd_exp; i++, it++, it_e++) {
		data[idx] = traindata[train_idx];
		data[idx].init_shape.dis = traindata[*it].shape.dis;
		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
		data[idx].init_shape.exp = traindata[*it_e].shape.exp;

		/*data[idx].init_shape.exp = traindata[train_idx].shape.exp;
		for (int j = 1; j < G_nShape; j++)
			data[idx].init_shape.exp(j) += cv::theRNG().uniform(-G_rand_exp_border, G_rand_exp_border);
*/

		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
		data[idx].ide_idx = train_idx;
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
		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
		data[idx].user = traindata[*it_u].user;
		data[idx].ide_idx = -1;
		recal_dis_ang(data[idx], bldshps);
		//recal_dis_ang_0ide(data[idx],arg_exp_land_matrix[*it_u]);
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
		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
		data[idx].ide_idx = train_idx;
#ifdef posit
		data[idx].init_f = traindata[train_idx].f + cv::theRNG().uniform(-G_rand_f_border, G_rand_f_border);
#endif // posit
#ifdef normalization
		data[idx].s(0, 0) += cv::theRNG().uniform(-G_rand_s_border, G_rand_s_border);
		data[idx].s(1, 1) += cv::theRNG().uniform(-G_rand_s_border, G_rand_s_border);
#endif
		recal_dis_ang(data[idx], bldshps);
		//recal_dis_ang_0ide(data[idx], arg_exp_land_matrix[train_idx]);
		idx++;
	}
}

vector<DataPoint> ArgumentData(
	const vector<DataPoint> &training_data, Eigen::MatrixXf &bldshps)
{

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

void cal_exp_land_matrix(std::vector<Eigen::MatrixXf> &arg_exp_land_matrix, Eigen::MatrixXf &bldshps, const vector<DataPoint> &training_data) {
	for (int idx = 0; idx < training_data.size(); idx++) {
		arg_exp_land_matrix[idx].resize(G_nShape, 3 * G_land_num);
		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
			for (int i_v = 0; i_v < G_land_num; i_v++) {
				Eigen::Vector3f V;
				V.setZero();
				for (int j = 0; j < 3; j++)
					for (int i_id = 0; i_id < G_iden_num; i_id++)
						if (i_shape == 0)
							V(j) += training_data[idx].user(i_id)*bldshps(i_id, training_data[idx].land_cor(i_v) * 3 + j);
						else
							V(j) += training_data[idx].user(i_id)*
							(bldshps(i_id, i_shape*G_nVerts * 3 + training_data[idx].land_cor(i_v) * 3 + j)
								- bldshps(i_id, training_data[idx].land_cor(i_v) * 3 + j));

				for (int j = 0; j < 3; j++)
					arg_exp_land_matrix[idx](i_shape, i_v * 3 + j) = V(j);
			}
	}
}


void TrainModel(const vector<DataPoint> &training_data, const TrainingParameters &tp, Eigen::MatrixXf &bldshps)
{
	cout << "Training data count: " << training_data.size() << endl;

	vector<vector<cv::Point2d>> shapes;
	//puts("A");
	for (const DataPoint &dp : training_data)
		shapes.push_back(dp.landmarks);
	//puts("D");
	vector<cv::Point2d> ref_shape = mean_shape(shapes, tp);
	//puts("A");	

	vector<Eigen::MatrixXf> arg_exp_land_matrix(training_data.size());

	cal_exp_land_matrix(arg_exp_land_matrix,bldshps,training_data);

	vector<DataPoint> argumented_training_data =
		ArgumentData(training_data, bldshps);

	//for (int i = 0; i < 100; i++)
	//	print_datapoint(argumented_training_data[i]);
	puts("C");
	Eigen::MatrixX3i tri_idx;
	std::vector<cv::Vec6f> triangleList;
	cv::Rect rect;
	puts("D");
	cal_del_tri(ref_shape, rect, triangleList, tri_idx);

	cv::Rect left_eye_rect, right_eye_rect, mouse_rect;
	cal_left_eye_rect(ref_shape, left_eye_rect);
	cal_right_eye_rect(ref_shape, right_eye_rect);
	cal_mouse_rect(ref_shape, mouse_rect);

	puts("E");

	puts("B");
	vector<RegressorTrain> stage_regressors(tp.T, RegressorTrain(tp));
	puts("training tr.....");
	for (int i = 0; i < tp.T; ++i)
	{
		long long s = cv::getTickCount();

		vector<Target_type> targets = 
			ComputeTargets(argumented_training_data);
		FILE *fp;
		fopen_s(&fp, "debug_target.txt", "a");
		fprintf(fp, "out number: %d ++++++++++++++++++++++++++++++++++++++++\n", i);
		fclose(fp);


		stage_regressors[i].Regress_ta(
			triangleList,rect, left_eye_rect, right_eye_rect, mouse_rect,tri_idx,ref_shape, &targets,
			argumented_training_data, bldshps, arg_exp_land_matrix);


		for (DataPoint &dp : argumented_training_data)
		{
			Target_type offset = 
				stage_regressors[i].Apply_ta(dp,bldshps,tri_idx, arg_exp_land_matrix);
			/*Transform t = Procrustes(dp.init_shape, mean_shape);
			t.Apply(&offset, false);*/
			dp.init_shape = shape_adjustment(dp.init_shape, offset);
		}

		cout << "(^_^) Finish training  tr..... " << i + 1 << " regressor. Using " 
			<< (cv::getTickCount() - s) / cv::getTickFrequency() 
			<< "s. " << tp.T << " in total." << endl;
		cout << "around " << (tp.T - i - 1)*(cv::getTickCount() - s) / cv::getTickFrequency() / 60
			<< "minutes letf!\n";
	}

	puts("training dis exp.....");
	for (int i = 0; i < tp.T; ++i)
	{
		long long s = cv::getTickCount();

		vector<Target_type> targets =
			ComputeTargets(argumented_training_data);
		FILE *fp;
		fopen_s(&fp, "debug_target.txt", "a");
		fprintf(fp, "out number: %d ++++++++++++++++++++++++++++++++++++++++\n", i);
		fclose(fp);

		stage_regressors[i].Regress_expdis(
			triangleList, rect, left_eye_rect, right_eye_rect, mouse_rect, tri_idx, ref_shape, &targets,
			argumented_training_data, bldshps, arg_exp_land_matrix);


		for (DataPoint &dp : argumented_training_data)
		{
			Target_type offset =
				stage_regressors[i].Apply_expdis(dp, bldshps, tri_idx, arg_exp_land_matrix);
			/*Transform t = Procrustes(dp.init_shape, mean_shape);
			t.Apply(&offset, false);*/
			dp.init_shape = shape_adjustment(dp.init_shape, offset);
		}

		cout << "(^_^) Finish training dis exp....." << i + 1 << " regressor. Using "
			<< (cv::getTickCount() - s) / cv::getTickFrequency()
			<< "s. " << tp.T << " in total." << endl;
		cout << "around " << (tp.T - i - 1)*(cv::getTickCount() - s) / cv::getTickFrequency() / 60
			<< "minutes letf!\n";
	}


	puts("B");
	system("pause");
	cv::FileStorage model_file;
	model_file.open(tp.output_model_pathname, cv::FileStorage::WRITE);
	model_file << "ref_shape" << ref_shape;
	//model_file << "test_init_shapes" << "[";
	//for (auto it = test_init_shapes.begin(); it != test_init_shapes.end(); ++it)
	//{
	//	model_file << *it;
	//}
	//model_file << "]";
	model_file << "stage_regressors" << "[";
	for (auto it = stage_regressors.begin(); it != stage_regressors.end(); ++it)
		model_file << *it;
	model_file << "]";
	model_file.release();
}
Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);

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
		vector<DataPoint> training_data = GetTrainingData(tp);
		train_data_normalize(training_data);
		load_bldshps(bldshps, bldshps_path);
		
		//system("pause");
		TrainModel(training_data, tp, bldshps);
	}
	catch (const exception &e)
	{
		cout << e.what() << endl;
		return -1;
	}
}

/*
1 grep -rl 'fopen_s(&fp,' ./ | xargs sed -i 's/fopen_s(&fp,/fp=fopen(/g'
2
3
4 grep -rl 'fscanf_s' ./ | xargs sed -i 's/fscanf_s/fscanf/g'


grep -rl 'fopen_s(&fpr,' ./ | xargs sed -i 's/fopen_s(&fpr,/fpr=fopen(/g'
grep -rl 'fopen_s(&fpw,' ./ | xargs sed -i 's/fopen_s(&fpw,/fpw=fopen(/g'
*/

