#include <iostream>
#include <string>
#include <io.h>

#include <opencv2/opencv.hpp>

#include "face_x.h"
#include "calculate_coeff_dde.h"
#define debug_def
#define from_video
#define show_feature_def

using namespace std;

const string kModelFileName = "model_r/model_all_5_face.xml.gz";
const string kModelFileName_dde = "model_r/model_dde_new_gauss.xml.gz";
const string kAlt2 = "model_r/haarcascade_frontalface_alt2.xml";
const string kTestImage = "./photo_test/video/lv_mp4/images/frame0.jpg";//real_time_test//22.png"; /test_samples/005_04_03_051_05.png";
const string videoImage = "./photo_test/video/lv_mp4/images/frame";

const string videolvsave = "./photo_test/video/lv_hard/lv_hard";

const std::string test_debug_lv_path = "./lv_file/lv_mp4_frame0.lv";// fitting_result_t66_pose_0.lv";
const string video_path = "D:\\sydney\\first\\code\\2017\\DDE\\FaceX/photo_test/video/lv_hard.avi";
const string image_se_path = "D:/sydney/first/data_me/test_lv/fw/Tester_1/TrainingPose/pose_%1d.jpg";
string land_video_save_path = "./photo_test/video/lv_hard_land.avi";
#ifdef win64

std::string fwhs_path_lv = "D:/sydney/first/data_me/test_lv/fw";
std::string lfw_path_lv = "D:/sydney/first/data_me/test_lv/lfw_image";
std::string gtav_path_lv = "D:/sydney/first/data_me/test_lv/GTAV_image";
std::string test_path = "D:/sydney/first/data_me/test";
std::string test_path_one = "D:/sydney/first/data_me/test_only_one";
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
std::string sg_vl_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_value_sqrt_77.txt";
std::string slt_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/sillht.txt";
std::string rect_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_point_rect.txt";
std::string save_coef_path = "./ide_fw_p1.lv";
std::string coef_path = "D:/sydney/first/data_me/fitting_coef/ide_fw_p1.lv";
std::string fwhs_path_p1 = "D:/sydney/first/data_me/test_lv";

//std::string debug_lv_save_path = "./lv_file/fitting_result_005_04_03_051_05.lv";
#endif // win64
#ifdef linux
std::string fwhs_path = "/home/weiliu/DDE/cal_coeff/data_me/FaceWarehouse";
std::string lfw_path = "/home/weiliu/DDE/cal_coeff/data_me/lfw_image";
std::string gtav_path = "/home/weiliu/DDE/cal_coeff/data_me/GTAV_image";
std::string test_path = "./test";
std::string test_path_one = "/home/weiliu/DDE/cal_coeff/data_me/test_only_one";
std::string test_path_two = "./data_me/test_only_two";
std::string test_path_three = "./data_me/test_only_three";
std::string bldshps_path = "./deal_data/blendshape_ide_svd_77.lv";
std::string sg_vl_path = "./deal_data/blendshape_ide_svd_value_sqrt_77.txt";
std::string slt_path = "./3d22d/sillht.txt";
std::string rect_path = "./3d22d/slt_point_rect.txt";
std::string save_coef_path = "../fitting_coef/ide_fw_p1.lv";
//std::string fwhs_path_p1 = "/home/weiliu/fitting_dde/1cal/data_me/fw_p1";
std::string fwhs_path_p1 = "D:/sydney/first/data_me/test_lv";
std::string fwhs_path_p2 = "/home/weiliu/fitting_dde/2cal/data_me/fw_p2";
std::string fwhs_path_p3 = "/home/weiliu/fitting_dde/3cal/data_me/fw_p3";
std::string fwhs_path_p4 = "/home/weiliu/fitting_dde/4cal/data_me/fw_p4";
std::string fwhs_path_p5 = "/home/weiliu/fitting_dde/5cal/data_me/fw_p5";
//std::string bldshps_path = "/home/weiliu/fitting_dde/cal/deal_data/blendshape_ide_svd_77.lv";


#endif // linux


#include <algorithm>
DataPoint pre_process(
	const FaceX & face_x, const DDEX &dde_x, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl, vector<DataPoint> &train_data, Eigen::MatrixX3i &tri_idx)
{
	//cout << "picture name:";
	//string pic_name;
	//cin >> pic_name;
#ifdef from_video
	cv::VideoCapture cap(video_path);
	cv::Mat image;
	cap >> image;
#else
	cv::Mat image = cv::imread(kTestImage);// +pic_name);
#endif // from_video
	cv::imshow("Alignment result", image);
	cv::waitKey();
	
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	cv::CascadeClassifier cc(kAlt2);
	if (cc.empty())
	{
		cout << "Cannot open model file " << kAlt2 << " for OpenCV face detector!" << endl;
		exit(1);
	}
	vector<cv::Rect> faces;
	double start_time = cv::getTickCount();
	cc.detectMultiScale(gray_image, faces);
	cout << "Detection time: " << (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << endl;

	/*for (cv::Rect face : faces)
	{*/
	cv::Rect face = faces[0];
	rect_scale(face, 1.5);

	start_time = cv::getTickCount();
	vector<cv::Point2d> landmarks = face_x.Alignment(gray_image, face);
	cout << "Alignment time: "
		<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << endl;
	show_image(image, face, landmarks);
	//save_land(landmarks);

	//fitting
	DataPoint data;
	data.image = gray_image;

	fit_solve(landmarks, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl, data);

	//testing the result of fitting
	cal_2d_land_i_0dis_ang(landmarks, bldshps, data);
#ifdef debug_def
	data.land_2d.rowwise() -= data.center;
	save_for_debug(data, test_debug_lv_path);
#endif
	show_image(image, face, landmarks);
	puts("fitting finished!");
	return data;

}

DataPoint debug_preprocess(std::string path_name) {
	DataPoint data;
#ifndef from_video
	data.image = cv::imread(kTestImage, cv::IMREAD_GRAYSCALE);
#endif // !from_video

	load_fitting_coef_one(path_name,data);
	data.landmarks.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++)
		data.landmarks[i_v].x = data.land_2d(i_v, 0), data.landmarks[i_v].y = data.image.rows - data.land_2d(i_v, 1);
	//std::cout << data.landmarks << "\n";
	//show_image_0rect(data.image, data.landmarks);
	//show_image_land_2d(data.image, data.land_2d);
	return data;
}

void Tracking_face(const FaceX & face_x)
{
	cout << "Press \"r\" to re-initialize the face location." << endl;
	cv::Mat frame;
	cv::Mat img;
	cv::VideoCapture vc(0);
	vc >> frame;
	cv::CascadeClassifier cc(kAlt2);
	vector<cv::Point2d> landmarks(face_x.landmarks_count());

	int flag = 0;
	vector<cv::Rect> faces;
	for (;;)
	{
		vc >> frame;
		cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
		cv::imshow("Gray image", img);

		vector<cv::Point2d> original_landmarks = landmarks;
		landmarks = face_x.Alignment(img, landmarks);

		for (int i = 0; i < landmarks.size(); ++i)
		{
			landmarks[i].x = (landmarks[i].x + original_landmarks[i].x) / 2;
			landmarks[i].y = (landmarks[i].y + original_landmarks[i].y) / 2;
		}

		for (cv::Point2d p : landmarks)
		{
			cv::circle(frame, p, 1, cv::Scalar(0, 255, 0), 2);
		}
		if (flag) cv::rectangle(frame, faces[0], cv::Scalar(100, 200, 255), 2);
		cv::imshow("\"r\" to re-initialize, \"q\" to exit", frame);
		int key = cv::waitKey(1);
		if (key == 'q')
			break;
		else if (key == 'r')
		{
			flag = 1;
			cc.detectMultiScale(img, faces);
			faces[0].x = std :: max(0, faces[0].x - 10);// faces[0].y = max(0, faces[0].y - 10);
			faces[0].width = std::min(img.rows - faces[0].x, faces[0].width + 25);
			faces[0].height = std::min(img.cols - faces[0].y, faces[0].height + 25);
			if (!faces.empty())
			{
				landmarks = face_x.Alignment(img, faces[0]);
			}
		}
	}
}

void DDE_run_test(
	DataPoint &data, const DDEX &dde_x, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl, vector<DataPoint> &train_data, Eigen::MatrixX3i &tri_idx) {

	puts("DDE running...");
	Eigen::MatrixXf exp_r_t_all_matrix;
	cal_exp_r_t_all_matrix(bldshps, data, exp_r_t_all_matrix);

	Target_type last_data[3];
	for (int test_round = 0; test_round < 500; test_round++) {
		printf("dde_round %d: \n", test_round);
		dde_x.dde(data, bldshps, tri_idx, train_data, jaw_land_corr, slt_line, slt_point_rect, exp_r_t_all_matrix);
		printf("dde_round %d: \n", test_round);
		puts("---------------------------------------------------");
		print_datapoint(data);
		show_image_0rect(data.image, data.landmarks);
		
		//system("pause");

		if (test_round % 10 == 0) {
			//show_image_0rect(data.image, data.landmarks);
			
			save_datapoint(data, kTestImage + "_" + to_string(test_round) + ".lv");
			puts("been saven");
			system("pause");
		}

//--------------------------------------post_processing
		/*
		last_data[0] = last_data[1]; last_data[1] = last_data[2]; last_data[2] = data.shape;
		if (test_round > 1) {
			ceres_post_processing(data, last_data[0], last_data[1], last_data[2], exp_r_t_point_matrix);
			update_2d_land(data,bldshps);
			//update_slt(
		}
		*/
	}

}

void DDE_video_test(
	DataPoint &data, const DDEX &dde_x, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl, vector<DataPoint> &train_data, Eigen::MatrixX3i &tri_idx) {

	if (_access(video_path.c_str(),0)==-1)
	{
		cout << "File does not exist" << endl;
		exit(-1);
	}
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) exit(2);//如果视频不能正常打开则返回
	cv::Mat rgb_image;
	cap >> rgb_image;
	
	cv::cvtColor(rgb_image, data.image, cv::COLOR_BGR2GRAY);
	for (int i_v = 0; i_v < G_land_num; i_v++)
		data.landmarks[i_v].x = data.land_2d(i_v, 0), data.landmarks[i_v].y = data.image.rows - data.land_2d(i_v, 1);
	dde_x.visualize_feature_cddt(rgb_image, tri_idx, data.landmarks);

	//cv::imshow("result", data.image);
	//cv::waitKey();

	puts("DDE running...");
	Eigen::MatrixXf exp_r_t_all_matrix;	
//	cv::resize(data.image, data.image,cv::Size(640, 480*3));
	cv::VideoWriter output_video(land_video_save_path, CV_FOURCC_DEFAULT, 25.0, cv::Size(data.image.cols, data.image.rows));
	//std::cout << data.image.cols << " " << data.image.rows << "\n" << cv::Size(data.image.cols, data.image.rows) << "\n";
	//output_video << data.image;
	//system("pause");
	cal_exp_r_t_all_matrix(bldshps, data, exp_r_t_all_matrix);
	Target_type last_data[3];

	for (int test_round = 1;/*&&test_round < 2000*/; test_round++) {
		cap >> rgb_image;
		if (rgb_image.empty()) break;
		cv::cvtColor(rgb_image, data.image, cv::COLOR_BGR2GRAY);
		printf("dde_round %d: \n", test_round);
		//data.image = cv::imread(videoImage+ to_string(test_round) + ".jpg", cv::IMREAD_GRAYSCALE);
		dde_x.dde(data, bldshps, tri_idx, train_data, jaw_land_corr, slt_line, slt_point_rect, exp_r_t_all_matrix);
		printf("dde_round %d: \n", test_round);
		puts("---------------------------------------------------");

		//show_image_0rect(data.image, data.landmarks);

		//--------------------------------------post_processing

		last_data[0] = last_data[1]; last_data[1] = last_data[2]; last_data[2] = data.shape;
		if (test_round > 3) {
			Eigen::MatrixXf exp_r_t_matrix;
			exp_r_t_matrix.resize(G_nShape, 3 * G_land_num);
			for (int i_exp = 0; i_exp < G_nShape; i_exp++)
				for (int i_v = 0; i_v < G_land_num; i_v++)
					for (int axis = 0; axis < 3; axis++)
						exp_r_t_matrix(i_exp, i_v * 3 + axis) = exp_r_t_all_matrix(i_exp, data.land_cor(i_v) * 3 + axis);
			ceres_post_processing(data, last_data[0], last_data[1], last_data[2], exp_r_t_matrix);
			//recal_dis_ang(data,bldshps);
			data.shape.exp(0) = 1;
			update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
			//update_slt(
		}
		last_data[2] = data.shape;
		////system("pause");
//	print_datapoint(data);
		save_datapoint(data, videolvsave + "_" + to_string(test_round) + ".lv");
		//print_datapoint(data);
		//show_image_0rect(data.image, data.landmarks);
			//save_video(data.image, data.landmarks, output_video);
		save_video(rgb_image, data.landmarks, output_video);
		//if (test_round % 30 == 1) {
		//	//show_image_0rect(data.image, data.landmarks);
		//	print_datapoint(data);
		//	show_image_0rect(data.image, data.landmarks);
		//	//system("pause");
		//}
		
	}

}









Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);
Eigen::VectorXf ide_sg_vl(G_iden_num);
Eigen::VectorXi inner_land_corr(G_inner_land_num);
Eigen::VectorXi jaw_land_corr(G_jaw_land_num);
std::vector<std::pair<int, int> > slt_point_rect[G_nVerts];
std::vector<int> slt_line[G_line_num];

void camera() {
	cv::Mat frame;
	cv::Mat img;
	cv::VideoCapture vc(0);
	vc >> frame;
	cv::VideoWriter output_video("lv_hard_glass.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25.0, cv::Size(640, 480));
	long long start_time = cv::getTickCount();
	for (;;)
	{
		vc >> frame;
		cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
		cv::imshow("Gray image", img);
		cv::imshow("gg", frame);
		cv::waitKey(20);
		output_video << frame;
		if ((cv::getTickCount() - start_time) / cv::getTickFrequency() > 45) break;
	}
	//exit(0);
}
//#include <dirent.h>
//#include <videoio.hpp>
//#include <video.hpp>
//#include<videostab.hpp>
void test_video() {
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) exit(2);//如果视频不能正常打开则返回
	cv::Mat rgb_image;
	
	puts("ppp");
	for (int test_round = 1; ; test_round++) {
		cap >> rgb_image;
		if (rgb_image.empty()) break;
		//printf("%d %d\n", test_round, fl);
		cv::imshow("gg", rgb_image);
		cv::waitKey(20);
		printf("%d\n", test_round);
	}
}
int main()
{
	//camera();
	//return 0;
	//test_video();
	//if (_access("test.avi",0)==-1)
	//{
	//	cout << "File does not exist" << endl;
	//	return -1;
	//}
	//cv::VideoCapture cap;
	////cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	//cap.open("test.avi");
	//if (!cap.isOpened())//如果视频不能正常打开则返回
	//	exit(2);
	//cv::Mat image;
	//puts("QAQ");
	//cap >> image;
	//cv::imshow("result", image);
	//cv::waitKey();

	try
	{	

#ifndef debug_def
		puts("initializing face...");
		FaceX face_x(kModelFileName);
		//Tracking_face(face_x);
#endif // !debug_def

		
		puts("initializing dde...");
		DDEX dde_x(kModelFileName_dde);
		load_inner_land_corr(inner_land_corr);
		load_jaw_land_corr(jaw_land_corr);
		//std::cout << inner_land_corr << '\n';
		load_slt(slt_line, slt_point_rect, slt_path, rect_path);
//		load_bldshps(bldshps, bldshps_path, ide_sg_vl, sg_vl_path);
		vector<DataPoint> training_data;
		training_data.clear();
		load_land_coef(fwhs_path_lv, ".jpg", training_data);
		load_land_coef(lfw_path_lv, ".jpg", training_data);
		load_land_coef(gtav_path_lv, ".bmp", training_data);
		printf("traindata size %d\n", training_data.size());
		//system("pause");

		Eigen::MatrixX3i tri_idx;
		std::vector<cv::Vec6f> triangleList;
		cv::Rect rect;
		std::vector<cv::Point2d> ref_shape = dde_x.get_ref_shape();
		
		Eigen::MatrixX2f points(G_land_num, 2);
		cal_del_tri(ref_shape, rect, triangleList, tri_idx);
		std::cout << ref_shape << "\n";
		std::cout << tri_idx << "\n";
		//system("pause");
		//printf("triangleList.size() %d:\n",triangleList.size());
		//for (int i = 0; i < triangleList.size(); i++) {
		//	printf("%d %.2f %.2f %.2f %.2f %.2f %.2f\n", i, triangleList[i][0], triangleList[i][1], triangleList[i][2], triangleList[i][3], triangleList[i][4], triangleList[i][5]);
		//	printf("%d %d %d %d\n", i, tri_idx(i, 0), tri_idx(i, 1), tri_idx(i, 2));

		//}
		//cout << "Choice: " << endl;
		//cout << "1. Align " << kTestImage << " in the current working directory." << endl;
		//cout << "2. Align video from web camera." << endl;
		//cout << "Please select one [1/2]: ";
		//int choice;
		//cin >> choice;
		//switch (choice)
		//{
		//case 1:
#ifdef debug_def
		DataPoint init_data = debug_preprocess(test_debug_lv_path);

		//DataPoint init_data = pre_process(face_x, dde_x, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl, training_data, tri_idx);
		//save_datapoint(init_data, test_debug_lv_path);
		//return 0;

		//init_data.shape.dis.rowwise() -= init_data.center;

		print_datapoint(init_data);
		

		//for (int i = 0; i < G_land_num; i++)
		//	points(i, 0) = ref_shape[i].x, points(i, 1) = ref_shape[i].y;
		//cal_del_tri(points, init_data.image, triangleList);


		//printf("%d %d\n", init_data.image.rows, init_data.image.cols);
		//show_image_0rect(init_data.image, ref_shape);
		//puts("asd");
#else
		DataPoint init_data = pre_process(face_x, dde_x, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl, training_data, tri_idx);
#endif // debug_def

#ifdef from_video
		DDE_video_test(init_data, dde_x, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl, training_data, tri_idx);
#else
		DDE_run_test(init_data, dde_x, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl, training_data, tri_idx);
#endif
		//	break;
		//case 2:
		//	//Tracking(face_x);
		//	break;
		//}
	}
	catch (const runtime_error& e)
	{
		cerr << e.what() << endl;
	}
}
