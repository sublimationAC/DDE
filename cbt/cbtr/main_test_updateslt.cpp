//
//#include <iostream>
//#include <fstream>
//#include <string>
//#include <set>
//#include <map>
//#include <algorithm>
//#include <numeric>
//#include <stdexcept>
//
//#include <opencv2/opencv.hpp>
//
//#include "regressor_train.h"
////#include "load_data.hpp"
//
////#define win64
//#define linux
//#define changed
//#define rand_exp_from_set_def
//
//
//#ifdef win64
//
//std::string fwhs_path = "D:/sydney/first/data_me/test_lv/fw";
//std::string lfw_path = "D:/sydney/first/data_me/test_lv/lfw_image";
//std::string gtav_path = "D:/sydney/first/data_me/test_lv/GTAV_image";
//std::string test_path = "D:/sydney/first/data_me/test_fw_psp_f";
//std::string test_path_one = "D:/sydney/first/data_me/test_only_one";
////std::string coef_path = "D:/sydney/first/data_me/fitting_coef/ide_fw_p1.lv";
//std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
//
//#endif // win64
//#ifdef linux
//std::string fwhs_path = "/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/fw";
//std::string lfw_path = "/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/lfw_image";
//std::string gtav_path = "/home/weiliu/fitting_dde/fitting_psp_f_l12_slt/GTAV_image";
//std::string test_path = "./test";
//std::string test_path_one = "/home/weiliu/DDE/cal_coeff/data_me/test_only_one";
////std::string fwhs_path_p1 = "/home/weiliu/fitting_dde/1cal/data_me/fw_p1";
//std::string fwhs_path_p1 = "D:/sydney/first/data_me/test_lv";
//std::string fwhs_path_p2 = "/home/weiliu/fitting_dde/2cal/data_me/fw_p2";
//std::string fwhs_path_p3 = "/home/weiliu/fitting_dde/3cal/data_me/fw_p3";
//std::string fwhs_path_p4 = "/home/weiliu/fitting_dde/4cal/data_me/fw_p4";
//std::string fwhs_path_p5 = "/home/weiliu/fitting_dde/5cal/data_me/fw_p5";
//std::string bldshps_path = "/home/weiliu/fitting_dde/const_file/deal_data/blendshape_ide_svd_77.lv";
//
//#endif // linux
//
//using namespace std;
//
//TrainingParameters ReadParameters(const string &filename)
//{
//	/*std::cout << filename << "\n";
//	system("pause");*/
//	ifstream fin(filename);
//	TrainingParameters result;
//	if (fin)
//	{
//		map<string, string> items;
//		string line;
//		int line_no = 0;
//		while (getline(fin, line))
//		{
//			++line_no;
//			line = TrimStr(line);
//			if (line.empty() || line[0] == '#')
//				continue;
//
//			int colon_pos = line.find(':');
//			if (colon_pos == string::npos)
//			{
//				throw runtime_error("Illegal line " + to_string(line_no) +
//					" in config file " + filename);
//			}
//
//			items[TrimStr(line.substr(0, colon_pos))] = TrimStr(
//				line.substr(colon_pos + 1));
//		}
//
//		result.training_data_root = items.at("training_data_root");
//		result.landmark_count = stoi(items.at("landmark_count"));
//		if (result.landmark_count <= 0)
//			throw invalid_argument("landmark_count must be positive.");
//
//		result.left_eye_index_x = stoi(items.at("left_eye_index_x"));
//		result.left_eye_index_y = stoi(items.at("left_eye_index_y"));
//		if (result.left_eye_index_x < 0 || result.left_eye_index_x >= result.landmark_count)
//			throw out_of_range("left_eye_index_x not in range.");
//		if (result.left_eye_index_y < 0 || result.left_eye_index_y >= result.landmark_count)
//			throw out_of_range("left_eye_index_y not in range.");
//		result.right_eye_index_x = stoi(items.at("right_eye_index_x"));
//		result.right_eye_index_y = stoi(items.at("right_eye_index_y"));
//		if (result.right_eye_index_x < 0 || result.right_eye_index_x >= result.landmark_count)
//			throw out_of_range("right_eye_index_x not in range.");
//		if (result.right_eye_index_y < 0 || result.right_eye_index_y >= result.landmark_count)
//			throw out_of_range("right_eye_index_y not in range.");
//
//		result.output_model_pathname = items.at("output_model_pathname");
//		result.T = stoi(items.at("T"));
//		if (result.T <= 0)
//			throw invalid_argument("T must be positive.");
//		result.K = stoi(items.at("K"));
//		if (result.K <= 0)
//			throw invalid_argument("K must be positive.");
//		result.P = stoi(items.at("P"));
//		if (result.P <= 0)
//			throw invalid_argument("P must be positive.");
//		result.Kappa = stod(items.at("Kappa"));
//		if (result.Kappa < 0.01 || result.Kappa > 1)
//			throw out_of_range("Kappa must be in [0.01, 1].");
//		result.F = stoi(items.at("F"));
//		if (result.F <= 0)
//			throw invalid_argument("F must be positive.");
//		result.Beta = stoi(items.at("Beta"));
//		if (result.Beta <= 0)
//			throw invalid_argument("Beta must be positive.");
//		result.TestInitShapeCount = stoi(items.at("TestInitShapeCount"));
//		if (result.TestInitShapeCount <= 0)
//			throw invalid_argument("TestInitShapeCount must be positive.");
//		result.ArgumentDataFactor = stoi(items.at("ArgumentDataFactor"));
//		if (result.ArgumentDataFactor <= 0)
//			throw invalid_argument("ArgumentDataFactor must be positive.");
//		result.Base = stoi(items.at("Base"));
//		if (result.Base <= 0)
//			throw invalid_argument("Base must be positive.");
//		result.Q = stoi(items.at("Q"));
//		if (result.Q <= 0)
//			throw invalid_argument("Q must be positive.");
//	}
//	else
//		throw runtime_error("Cannot open config file: " + filename);
//
//	return result;
//}
//
//vector<DataPoint> GetTrainingData()
//{
//
//	vector<DataPoint> result;
//	/*load_img_land_coef(fwhs_path,".jpg",result);
//	load_img_land_coef(lfw_path, ".jpg", result);
//	load_img_land_coef(gtav_path, ".bmp", result);*/
//	load_img_land_coef(fwhs_path, ".jpg", result);
//
//	return result;
//}
//
//void train_data_normalize(vector<DataPoint> &training_data) {
//	puts("normalizing face rect....!");
//	for (auto data : training_data) {
//		//		cv::imshow("ini", data.image);
//				//cv::waitKey(0);
//		normalize_gauss_face_rect(data.image, data.face_rect);
//		//		cv::rectangle(data.image, data.face_rect, 0);
//				//cv::imshow("after_norm", data.image);
//				//cv::waitKey(0);
//	}
//}
//
//set<int> rand_df_idx(int which, int border, int num) {
//	set<int> result;
//	result.clear();
//	while (result.size() < num)
//	{
//		int rand_index = cv::theRNG().uniform(0, border);
//		if (rand_index != which)
//			result.insert(rand_index);
//	}
//	return result;
//}
//
//void aug_rand_rot(
//	const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx,
//	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
//	Eigen::MatrixXf &bldshps, vector<Eigen::MatrixXf> &arg_inner_bldshp_matrix) {
//
//	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_rot);
//	auto it = temp.cbegin();
//	for (int i = 0; i < G_rnd_rot; i++, it++) {
//		data[idx] = traindata[train_idx];
//		data[idx].init_shape.dis = traindata[*it].shape.dis;
//		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
//		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
//		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
//		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
//		data[idx].ide_idx = train_idx;
//		//std::cout << "s: " << data[idx].s << "\n";
//		//system("pause");
//		//for (int j=0;j<3;j++)
//		//	data[idx].init_shape.angle(j)+= cv::theRNG().uniform(-G_rand_angle_border, G_rand_angle_border);
//		data[idx].init_shape.angle(0) += cv::theRNG().uniform(-G_rand_angle_border * 15, G_rand_angle_border * 15);
//		data[idx].init_shape.angle(1) += cv::theRNG().uniform(-G_rand_angle_border * 10, G_rand_angle_border * 10);
//		data[idx].init_shape.angle(2) += cv::theRNG().uniform(-G_rand_angle_border * 5, G_rand_angle_border * 5);
//		puts("calculating landmarks");
//		cv::Mat image_save = data[idx].image.clone();
//		draw_image_land_2d(image_save, data[idx].landmarks, cv::Scalar(0, 255, 0));
//
//		std::vector<cv::Point2d> land_temp(G_land_num);
//		puts("calculating init landmarks");
//		cal_2d_land_i_ang_tg(land_temp, data[idx].init_shape, bldshps, data[idx]);
//		draw_image_land_2d(image_save, land_temp, cv::Scalar(255,0, 0));
//
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data[idx]);
//		cal_2d_land_i_ang_tg(land_temp, data[idx].shape, bldshps, data[idx]);
//		puts("calculating new init landmarks");
//		draw_image_land_2d(image_save, land_temp, cv::Scalar(255, 255, 255));
//
//		puts("calculating slt init landmarks");
//		cal_2d_land_i_ang_tg(land_temp, data[idx].init_shape, bldshps, data[idx]);
//		draw_image_land_2d(image_save, land_temp, cv::Scalar(0, 0, 255));
//
//		recal_dis_ang_sqz_optmz(data[idx], bldshps);
//
//		cal_slt_bldshps(data[idx], bldshps);
//		std::vector<cv::Point2d> land_temp_2(G_land_num);
//
//		get_init_land_ang_0ide_i(land_temp_2, data[idx], bldshps, arg_inner_bldshp_matrix);
//		draw_image_land_2d(image_save, land_temp_2, cv::Scalar(150, 150, 150));
//		std::cout << "init_shape:" << land_temp << "\n";
//		std::cout << "init_shape sqz way:" << land_temp_2 << "\n";
//		cv::imshow("check",image_save);
//		
//		cv::waitKey(0);
//		idx++;
//	}
//#ifdef mxini_def
//	set<int> temp_e = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
//	auto it_e = temp_e.cbegin();
//	temp.clear();
//	temp = rand_df_idx(train_idx, traindata.size(), G_rnd_rot);
//	it = temp.cbegin();
//	for (int i = 0; i < G_rnd_rot; i++, it++) {
//		data[idx] = traindata[train_idx];
//		data[idx].init_shape.dis = traindata[*it].shape.dis;
//		data[idx].init_shape.exp = traindata[*it_e].shape.exp;
//		//data[idx].init_shape.exp = traindata[train_idx].shape.exp;
//		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
//		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
//		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
//		data[idx].ide_idx = train_idx;
//		//for (int j=0;j<3;j++)
//		//	data[idx].init_shape.angle(j)+= cv::theRNG().uniform(-G_rand_angle_border, G_rand_angle_border);
//		data[idx].init_shape.angle(0) += cv::theRNG().uniform(-G_rand_angle_border * 15, G_rand_angle_border * 15);
//		data[idx].init_shape.angle(1) += cv::theRNG().uniform(-G_rand_angle_border * 10, G_rand_angle_border * 10);
//		data[idx].init_shape.angle(2) += cv::theRNG().uniform(-G_rand_angle_border * 5, G_rand_angle_border * 5);
//
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data[idx]);
//		recal_dis_ang_sqz_optmz(data[idx], bldshps);
//		cal_slt_bldshps(data[idx], bldshps);
//		idx++;
//	}
//#endif // mxini_def
//
//}
//void aug_rand_tslt(
//	const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx,
//	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
//	Eigen::MatrixXf &bldshps) {
//
//	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_tslt);
//	auto it = temp.cbegin();
//	for (int i = 0; i < G_rnd_tslt; i++, it++) {
//		data[idx] = traindata[train_idx];
//		data[idx].init_shape.dis = traindata[*it].shape.dis;
//		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
//		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
//		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
//		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
//		data[idx].ide_idx = train_idx;
//		for (int j = 0; j < G_tslt_num; j++)
//			data[idx].init_shape.tslt(j) = traindata[train_idx].shape.tslt(j) + cv::theRNG().uniform(-G_rand_tslt_border, G_rand_tslt_border);
//#ifdef normalization
//		data[idx].init_shape.tslt(2) = 0;
//#endif // normalization		
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data[idx]);
//		recal_dis_ang_sqz_optmz(data[idx], bldshps);
//		cal_slt_bldshps(data[idx], bldshps);
//		idx++;
//	}
//#ifdef mxini_def
//	set<int> temp_e = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
//	auto it_e = temp_e.cbegin();
//	temp.clear();
//	temp = rand_df_idx(train_idx, traindata.size(), G_rnd_tslt);
//	it = temp.cbegin();
//	for (int i = 0; i < G_rnd_tslt; i++, it++) {
//		data[idx] = traindata[train_idx];
//		data[idx].init_shape.dis = traindata[*it].shape.dis;
//		//data[idx].init_shape.exp = traindata[train_idx].shape.exp;
//		data[idx].init_shape.exp = traindata[*it_e].shape.exp;
//		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
//		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
//		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
//		data[idx].ide_idx = train_idx;
//		for (int j = 0; j < G_tslt_num; j++)
//			data[idx].init_shape.tslt(j) = traindata[train_idx].shape.tslt(j) + cv::theRNG().uniform(-G_rand_tslt_border, G_rand_tslt_border);
//#ifdef normalization
//		data[idx].init_shape.tslt(2) = 0;
//#endif // normalization	
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data[idx]);
//		recal_dis_ang_sqz_optmz(data[idx], bldshps);
//		cal_slt_bldshps(data[idx], bldshps);
//		idx++;
//
//	}
//#endif // mxini_def
//
//}
//void aug_rand_exp(
//	const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx,
//	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
//	Eigen::MatrixXf &bldshps) {
//
//	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
//	auto it = temp.cbegin();
//	set<int> temp_e = rand_df_idx(train_idx, traindata.size(), G_rnd_exp);
//	auto it_e = temp_e.cbegin();
//	for (int i = 0; i < G_rnd_exp; i++, it++, it_e++) {
//		data[idx] = traindata[train_idx];
//		data[idx].init_shape.dis = traindata[*it].shape.dis;
//		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
//		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
//		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
//#ifdef rand_exp_from_set_def
//		data[idx].init_shape.exp = traindata[*it_e].shape.exp;
//#else
//		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
//		for (int j = 1; j < G_nShape; j++)
//			data[idx].init_shape.exp(j) += cv::theRNG().uniform(-G_rand_exp_border, G_rand_exp_border);
//
//#endif // rand_exp_from_set
//
//		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
//		data[idx].ide_idx = train_idx;
//		//		update_slt();
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data[idx]);
//		recal_dis_ang_sqz_optmz(data[idx], bldshps);
//		cal_slt_bldshps(data[idx], bldshps);
//		idx++;
//	}
//}
//void aug_rand_user(
//	const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx,
//	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
//	Eigen::MatrixXf &bldshps) {
//
//	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_user);
//	auto it = temp.cbegin();
//	set<int> temp_u = rand_df_idx(train_idx, traindata.size(), G_rnd_user);
//	auto it_u = temp_u.cbegin();
//	for (int i = 0; i < G_rnd_user; i++, it++, it_u++) {
//		data[idx] = traindata[train_idx];
//		data[idx].init_shape.dis = traindata[*it].shape.dis;
//		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
//		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
//		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
//		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
//		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
//		data[idx].user = traindata[*it_u].user;
//		data[idx].ide_idx = *it_u;
//		//recal_dis_ang_0ide(data[idx],arg_exp_land_matrix[*it_u]);
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data[idx]);
//		recal_dis_ang_sqz_optmz(data[idx], bldshps);
//		cal_slt_bldshps(data[idx], bldshps);
//		idx++;
//	}
//}
//
//void aug_rand_f(
//	const vector<DataPoint> &traindata, vector<DataPoint> &data, int &idx, int train_idx,
//	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
//	Eigen::MatrixXf &bldshps) {
//
//	set<int> temp = rand_df_idx(train_idx, traindata.size(), G_rnd_user);
//	auto it = temp.cbegin();
//	for (int i = 0; i < G_rnd_user; i++, it++) {
//		data[idx] = traindata[train_idx];
//		data[idx].init_shape.dis = traindata[*it].shape.dis;
//		data[idx].init_shape.tslt = traindata[train_idx].shape.tslt;
//		//data[idx].init_shape.rot = traindata[train_idx].shape.rot;
//		data[idx].init_shape.angle = traindata[train_idx].shape.angle;
//		data[idx].init_shape.exp = traindata[train_idx].shape.exp;
//		data[idx].init_shape.exp(0) = data[idx].shape.exp(0) = 1;
//		data[idx].ide_idx = train_idx;
//#ifdef perspective
//		data[idx].fcs = traindata[train_idx].fcs + cv::theRNG().uniform(-G_rand_f_border, G_rand_f_border);
//#endif // perspective
//#ifdef normalization
//		data[idx].s(0, 0) += cv::theRNG().uniform(-G_rand_s_border, G_rand_s_border);
//		data[idx].s(1, 1) += cv::theRNG().uniform(-G_rand_s_border, G_rand_s_border);
//#endif
//		//recal_dis_ang_0ide(data[idx], arg_exp_land_matrix[train_idx]);
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data[idx]);
//		recal_dis_ang_sqz_optmz(data[idx], bldshps);
//		cal_slt_bldshps(data[idx], bldshps);
//		idx++;
//	}
//}
//
//vector<DataPoint> ArgumentData(
//	const vector<DataPoint> &training_data,
//	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::MatrixXf &bldshps,
//	vector<Eigen::MatrixXf> &arg_inner_bldshp_matrix)
//{
//
//	vector<DataPoint> result(training_data.size() * G_trn_factor);
//	int idx = 0;
//	for (int i = 0; i < training_data.size(); ++i)
//	{
//		printf("ArgumentData i: %d\n", i);
//		aug_rand_rot(training_data, result, idx, i, slt_line, slt_point_rect, bldshps, arg_inner_bldshp_matrix);
//		aug_rand_tslt(training_data, result, idx, i, slt_line, slt_point_rect, bldshps);
//		aug_rand_exp(training_data, result, idx, i, slt_line, slt_point_rect, bldshps);
//		aug_rand_user(training_data, result, idx, i, slt_line, slt_point_rect, bldshps);
//		aug_rand_f(training_data, result, idx, i, slt_line, slt_point_rect, bldshps);
//	}
//	return result;
//}
//
//
//
//
//
//
//vector<Target_type> ComputeTargets(const vector<DataPoint> &data)
//{
//	vector<Target_type> result;
//
//	for (const DataPoint& dp : data)
//	{
//		Target_type error = shape_difference(dp.shape, dp.init_shape);
//		result.push_back(error);
//	}
//
//	return result;
//}
//
//void cal_inner_bldshp_matrix(std::vector<Eigen::MatrixXf> &arg_exp_land_matrix, Eigen::MatrixXf &bldshps, const vector<DataPoint> &training_data) {
//	for (int idx = 0; idx < training_data.size(); idx++) {
//		arg_exp_land_matrix[idx].resize(G_nShape, 3 * G_inner_land_num);
//		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//			for (int i_v = 0; i_v < G_inner_land_num; i_v++) {
//				Eigen::Vector3f V;
//				V.setZero();
//				for (int j = 0; j < 3; j++)
//					for (int i_id = 0; i_id < G_iden_num; i_id++)
//						if (i_shape == 0)
//							V(j) += training_data[idx].user(i_id)*bldshps(i_id, training_data[idx].land_cor(i_v + G_outer_land_num) * 3 + j);
//						else
//							V(j) += training_data[idx].user(i_id)*
//							(bldshps(i_id, i_shape*G_nVerts * 3 + training_data[idx].land_cor(i_v + G_outer_land_num) * 3 + j)
//								- bldshps(i_id, training_data[idx].land_cor(i_v + G_outer_land_num) * 3 + j));
//
//				for (int j = 0; j < 3; j++)
//					arg_exp_land_matrix[idx](i_shape, i_v * 3 + j) = V(j);
//			}
//	}
//}
//#ifdef win64
//std::string slt_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_line_4_10.txt";
//std::string rect_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_rect_4_10.txt";
//#endif // win64
//#ifdef linux
//std::string slt_path = "/home/weiliu/fitting_dde/const_file/3d22d/slt_line_4_10.txt";
//std::string rect_path = "/home/weiliu/fitting_dde/const_file/3d22d/slt_rect_4_10.txt";
//#endif // linux
//
//std::vector<std::pair<int, int> > slt_point_rect[G_nVerts];
//std::vector<int> slt_line[G_line_num];
//void TrainModel(const vector<DataPoint> &training_data, const TrainingParameters &tp, Eigen::MatrixXf &bldshps)
//{
//	cout << "Training data count: " << training_data.size() << endl;
//
//	vector<vector<cv::Point2d>> shapes;
//	//puts("A");
//	for (const DataPoint &dp : training_data)
//		shapes.push_back(dp.landmarks);
//	//puts("D");
//	vector<cv::Point2d> ref_shape = mean_shape(shapes, tp);
//	//puts("A");	
//
//	load_slt(slt_line, slt_point_rect, slt_path, rect_path);
//
//	get_index_slt_point_rect(slt_line, slt_point_rect, training_data[0]);
//
//
//	vector<Eigen::MatrixXf> arg_inner_bldshp_matrix(training_data.size());
//
//	cal_inner_bldshp_matrix(arg_inner_bldshp_matrix, bldshps, training_data);
//
//
//	vector<DataPoint> argumented_training_data =
//		ArgumentData(training_data, slt_line, slt_point_rect, bldshps, arg_inner_bldshp_matrix);
//
//	//for (int i = 0; i < 100; i++)
//	//	print_datapoint(argumented_training_data[i]);
//	puts("C");
//	Eigen::MatrixX3i tri_idx;
//	std::vector<cv::Vec6f> triangleList;
//	cv::Rect rect;
//	puts("D");
//	cal_del_tri(ref_shape, rect, triangleList, tri_idx);
//
//	cv::Rect left_eye_rect, right_eye_rect, mouse_rect;
//	cal_left_eye_rect(ref_shape, left_eye_rect);
//	cal_right_eye_rect(ref_shape, right_eye_rect);
//	cal_mouse_rect(ref_shape, mouse_rect);
//
//	puts("E");
//
//	puts("B");
//	vector<RegressorTrain> stage_regressors(tp.T, RegressorTrain(tp));
//	puts("training tr.....");
//	for (int i = 0; i < tp.T; ++i)
//	{
//		long long s = cv::getTickCount();
//
//		vector<Target_type> targets =
//			ComputeTargets(argumented_training_data);
//		FILE *fp;
//		fp=fopen( "debug_target.txt", "a");
//		fprintf(fp, "out number: %d ++++++++++++++++++++++++++++++++++++++++\n", i);
//		fclose(fp);
//
//
//		stage_regressors[i].Regress_ta(
//			triangleList, rect, left_eye_rect, right_eye_rect, mouse_rect, tri_idx, ref_shape, &targets,
//			argumented_training_data, bldshps, arg_inner_bldshp_matrix);
//
//
//		for (DataPoint &dp : argumented_training_data)
//		{
//			Target_type offset =
//				stage_regressors[i].Apply_ta(dp, bldshps, tri_idx, arg_inner_bldshp_matrix);
//			/*Transform t = Procrustes(dp.init_shape, mean_shape);
//			t.Apply(&offset, false);*/
//			dp.init_shape = shape_adjustment(dp.init_shape, offset);
//		}
//
//		cout << "(^_^) Finish training  tr..... " << i + 1 << " regressor. Using "
//			<< (cv::getTickCount() - s) / cv::getTickFrequency()
//			<< "s. " << tp.T << " in total." << endl;
//		cout << "around " << (tp.T - i - 1)*(cv::getTickCount() - s) / cv::getTickFrequency() / 60
//			<< "minutes letf!\n";
//	}
//
//	puts("training dis exp.....");
//	for (int i = 0; i < tp.T; ++i)
//	{
//		long long s = cv::getTickCount();
//
//		vector<Target_type> targets =
//			ComputeTargets(argumented_training_data);
//		FILE *fp;
//		fp=fopen( "debug_target.txt", "a");
//		fprintf(fp, "out number: %d ++++++++++++++++++++++++++++++++++++++++\n", i);
//		fclose(fp);
//
//		stage_regressors[i].Regress_expdis(
//			triangleList, rect, left_eye_rect, right_eye_rect, mouse_rect, tri_idx, ref_shape, &targets,
//			argumented_training_data, bldshps, arg_inner_bldshp_matrix);
//
//
//		for (DataPoint &dp : argumented_training_data)
//		{
//			Target_type offset =
//				stage_regressors[i].Apply_expdis(dp, bldshps, tri_idx, arg_inner_bldshp_matrix);
//			/*Transform t = Procrustes(dp.init_shape, mean_shape);
//			t.Apply(&offset, false);*/
//			dp.init_shape = shape_adjustment(dp.init_shape, offset);
//		}
//
//		cout << "(^_^) Finish training dis exp....." << i + 1 << " regressor. Using "
//			<< (cv::getTickCount() - s) / cv::getTickFrequency()
//			<< "s. " << tp.T << " in total." << endl;
//		cout << "around " << (tp.T - i - 1)*(cv::getTickCount() - s) / cv::getTickFrequency() / 60
//			<< "minutes letf!\n";
//	}
//
//
//	puts("B");
//	system("pause");
//	cv::FileStorage model_file;
//	model_file.open(tp.output_model_pathname, cv::FileStorage::WRITE);
//	model_file << "ref_shape" << ref_shape;
//	//model_file << "test_init_shapes" << "[";
//	//for (auto it = test_init_shapes.begin(); it != test_init_shapes.end(); ++it)
//	//{
//	//	model_file << *it;
//	//}
//	//model_file << "]";
//	model_file << "stage_regressors" << "[";
//	for (auto it = stage_regressors.begin(); it != stage_regressors.end(); ++it)
//		model_file << *it;
//	model_file << "]";
//	model_file.release();
//}
//
//Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);
//void draw_land_img(cv::Mat image, std::vector<cv::Point2d> landmarks) {
//
//	for (int i = 0; i < 15; i++)
//		cv::circle(image, landmarks[i], 1, cv::Scalar(0, 250, 0), 2);
//	for (int i = 1; i < 15; i++)
//		cv::line(image, landmarks[i], landmarks[i - 1], cv::Scalar(255, 0, 0));
//
//	for (cv::Point2d landmark : landmarks)
//	{
//		cv::circle(image, landmark, 0.1, cv::Scalar(0, 250, 0), 2);
//	}
//
//}
//void test_upt_slt_angle(
//	DataPoint &data, Eigen::MatrixXf &bldshps,
//	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect) {
//	float pi = acos(-1);
//	float be = -pi / 2, step = pi / 100;
//	float en = pi / 2 + step - 0.000000001;
//
//
//	FILE *fp;
//	fp = fopen("test_updt_slt_dde_2d_point.txt", "w");
//	fprintf(fp, "%d\n", (int)((en - be) / step + 1) * 3);
//	fclose(fp);
//
//	fp = fopen("test_updt_slt_dde.txt", "w");
//	fprintf(fp, "%d\n", (int)((en - be) / step + 1) * 3);
//	fclose(fp);
//
//	Eigen::Vector3f angle;
//	angle = data.init_shape.angle;
//	data.shape.angle = data.init_shape.angle;
//	int i = 1, which = 0;
//	for (float ag = be; ag < en; ag += step, i++) {
//		angle(which) = ag;
//		data.init_shape.angle = angle;
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data);
//	}
//	printf("i %d\n", i);
//	angle(which) = data.shape.angle(which);
//	which = 1;
//	for (float ag = be; ag < en; ag += step, i++) {
//		angle(which) = ag;
//		data.init_shape.angle = angle;
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data);
//	}
//	printf("i %d\n", i);
//	angle(which) = data.shape.angle(which);
//	which = 2;
//	for (float ag = be; ag < en; ag += step, i++) {
//		angle(which) = ag;
//		data.init_shape.angle = angle;
//		update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data);
//	}
//	
//}
//void debug_inner_slt() {
//	DataPoint temp;
//	//std::string psp_name = "G:\\DDE\\server_lv\\lv_small_psp_inslt/lv_small_52";
//	std::string psp_name = "/home/weiliu/psp_f2obj/lv_small_psp_inslt/lv_small_52";
//	load_fitting_coef_one(psp_name + ".psp_f", temp);
//
//	temp.image = cv::imread("/home/weiliu/running_dde/psp_f_file/lv_small_first_frame.jpg");
//
//
//	
//	//cal_init_2d_land_ang_0ide_i(temp, training_data[i],arg_exp_land_matrix[training_data[i].ide_idx]);
//	load_bldshps(bldshps, bldshps_path);
//	std::vector<cv::Point2d> land_temp(G_land_num);
//	
//	cal_2d_land_i_ang_tg(land_temp, temp.shape, bldshps, temp);
//	draw_land_img(temp.image, land_temp);
//
//	cv::imshow("img",temp.image);
//	cv::waitKey(0);
//
//	load_slt(slt_line, slt_point_rect, slt_path, rect_path);
//	get_index_slt_point_rect(slt_line, slt_point_rect, temp);
//
//	temp.init_shape = temp.shape;
//	update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, temp);
//
//	cal_2d_land_i_ang_tg(land_temp, temp.shape, bldshps, temp);
//	draw_land_img(temp.image, land_temp);
//	cv::imshow("img", temp.image);
//	cv::waitKey(0);
//
//	test_upt_slt_angle(temp, bldshps,
//		slt_line, slt_point_rect);
//
//	exit(9);
//}
//
//int main()
//{
//	debug_inner_slt();
//
//		vector<DataPoint> training_data = GetTrainingData();
//		printf("training_data size: %d %.5f %.5f\n", training_data.size(),sin(30),sin(pi/6));
//		system("pause");
//		//train_data_normalize(training_data);
//		load_bldshps(bldshps, bldshps_path);
//		load_slt(slt_line, slt_point_rect, slt_path, rect_path);
//		
//		get_index_slt_point_rect(slt_line, slt_point_rect, training_data[0]);
//
//		DataPoint data = training_data[0];
//		data.init_shape = data.shape;
//
////yaw
//		//int st = -90, en = 90, step = 10;
//		//FILE *fp;
//		//fp=fopen( "test_updt_slt.txt", "w");
//		//fprintf(fp, "%d\n", (en - st) / step + 1);
//		//fclose(fp);
//		//for (int ang = st; ang <= en; ang += step) {
//		//	printf("now angle: %d\n", ang);
//		//	data.init_shape.angle(1) = ang * G_rand_angle_border;
//		//	update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data);
//		//}
////roll
//		int st = -30, en = 50, step = 10;
//		FILE *fp;
//		fp=fopen( "test_updt_slt.txt", "w");
//		fprintf(fp, "%d\n", (en - st) / step + 1);
//		fclose(fp);
//		for (int ang = st; ang <= en; ang += step) {
//			printf("now angle: %d\n", ang);
//			data.init_shape.angle(2) = ang * G_rand_angle_border;
//			update_slt_init_shape_DDE(bldshps, slt_line, slt_point_rect, data);
//		}
//
//		//vector<Eigen::MatrixXf> arg_inner_bldshp_matrix(training_data.size());
//
//		//cal_inner_bldshp_matrix(arg_inner_bldshp_matrix, bldshps, training_data);
//
//
//		//vector<DataPoint> argumented_training_data =
//		//	ArgumentData(training_data, slt_line, slt_point_rect, bldshps, arg_inner_bldshp_matrix);
//		//system("pause");
//		
//	
//}
//
///*
//1 grep -rl 'fopen_s(&fp,' ./ | xargs sed -i 's/fopen_s(&fp,/fp=fopen(/g'
//2
//3
//4 grep -rl 'fscanf_s' ./ | xargs sed -i 's/fscanf_s/fscanf/g'
//
//
//grep -rl 'fopen_s(&fpr,' ./ | xargs sed -i 's/fopen_s(&fpr,/fpr=fopen(/g'
//grep -rl 'fopen_s(&fpw,' ./ | xargs sed -i 's/fopen_s(&fpw,/fpw=fopen(/g'
//*/
//
