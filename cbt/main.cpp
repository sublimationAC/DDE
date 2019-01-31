#include <iostream>
#include <string>
#include <dirent.h>
#include <io.h>

#include <opencv2/opencv.hpp>

#include "face_x.h"
#include "calculate_coeff_dde.hpp"
#define debug_def
#define from_video
//#define norm_lk_face
//#define gs_filter
//#define md_filter
//#define show_feature_def
#define lk_rgb_def
#define lk_opencv_def

const int G_gs_window_size = 6;
const float G_gs_angle_sig = 3;
using namespace std;

const string kModelFileName = "model_r/model_all_5_face.xml.gz";
const string kModelFileName_dde = "model_r/model_dde_ta_ed_enh_norm_gauss.xml.gz";
const string kAlt2 = "model_r/haarcascade_frontalface_alt2.xml";
const string kTestImage = "./photo_test/test_samples//005_04_03_051_05.png";//video/lv_mp4/images/frame0.jpg";// /test_samples/005_04_03_051_05.png";
const string videoImage = "./photo_test/video/lv_mp4/images/frame";

const string videolvsave = "./photo_test/video/rubbish/r_";
const string gs_smooth_videolvsave = "./photo_test/video/lv_out_npmxini_pp204_gs_smooth/lv_out_npmxini_pp204_gs_smooth";

const std::string test_debug_lv_path = "./lv_file/lv_easy_pre.lv";// fitting_result_t66_pose_0.lv";
const string video_path = "./photo_test/video/lv_easy.avi";

string land_video_save_path = "./photo_test/video/lv_easy_npmxini_pp204.avi";
string land_gs_smooth_video_save_path = "./photo_test/video/lv_easy_mxini_pp204_gs_smooth.avi";


string lk_video_save_path = "./lv_easy_lk.avi";
#ifdef win64
const string image_se_path = "D:/sydney/first/data_me/test_lv/fw/Tester_1/TrainingPose/pose_%1d.jpg";
std::string fwhs_path_lv = "D:/sydney/first/data_me/test_lv/fw";
std::string lfw_path_lv = "D:/sydney/first/data_me/test_lv/lfw_image";
std::string gtav_path_lv = "D:/sydney/first/data_me/test_lv/GTAV_image";
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
std::string sg_vl_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_value_sqrt_77.txt";
std::string slt_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/sillht.txt";
std::string rect_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_point_rect.txt";
std::string inner_cor_path = "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3/cal_coeffience_Q_M_u_e_3/inner_jaw/inner_vertex_corr.txt";
std::string jaw_land_path = "D:\\sydney\\first\\code\\2017\\cal_coeffience_Q_M_u_e_3/cal_coeffience_Q_M_u_e_3/inner_jaw/jaw_vertex.txt";
///std::string save_coef_path = "./ide_fw_p1.lv";
///std::string coef_path = "D:/sydney/first/data_me/fitting_coef/ide_fw_p1.lv";
///std::string fwhs_path_p1 = "D:/sydney/first/data_me/test_lv";

//std::string debug_lv_save_path = "./lv_file/fitting_result_005_04_03_051_05.lv";
#endif // win64
#ifdef linux
std::string fwhs_path_lv = "/home/weiliu/fitting_dde/fitting_coef/fw";
std::string lfw_path_lv = "/home/weiliu/fitting_dde/fitting_coef/lfw_image";
std::string gtav_path_lv = "/home/weiliu/fitting_dde/fitting_coef/GTAV_image";

std::string sg_vl_path = "/home/weiliu/fitting_dde/cal/deal_data/blendshape_ide_svd_value_sqrt_77.txt";
std::string slt_path = "/home/weiliu/fitting_dde/cal/3d22d/sillht.txt";
std::string rect_path = "/home/weiliu/fitting_dde/cal/3d22d/slt_point_rect.txt";

std::string bldshps_path = "/home/weiliu/fitting_dde/cal/deal_data/blendshape_ide_svd_77.lv";


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
#ifdef win64
	cv::imshow("Alignment result", image);
	cv::waitKey();
#endif // win64

	
	
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
#ifdef win64
	show_image(image, face, landmarks);
#endif // win64

	
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

#ifdef win64
	show_image(image, face, landmarks);
#endif // win64
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
		dde_x.dde(data.image,data, bldshps, tri_idx, train_data, jaw_land_corr, slt_line, slt_point_rect, exp_r_t_all_matrix);
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

#ifdef win64
	if (_access(video_path.c_str(),0)==-1)
#endif
#ifdef linux
	if (access(video_path.c_str(), 0) == -1)
#endif
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

	cv::CascadeClassifier cc(kAlt2);
	vector<cv::Rect> faces;
	cc.detectMultiScale(data.image, faces);
#ifndef small_rect_def
	rect_scale(faces[0], 1.5);
#endif // !small_rect_def	
	normalize_gauss_face_rect(data.image, faces[0]);
#ifdef show_feature_def
	dde_x.visualize_feature_cddt(rgb_image, tri_idx, data.landmarks);
#endif // show_feature_def
	

	//cv::imshow("result", data.image);
	//cv::waitKey();

	puts("DDE running...");
	Eigen::MatrixXf exp_r_t_all_matrix;	
	cal_exp_r_t_all_matrix(bldshps, data, exp_r_t_all_matrix);

	update_training_data(data, train_data,exp_r_t_all_matrix);

	Target_type last_data[3];
//	cv::resize(data.image, data.image,cv::Size(640, 480*3));
	cv::VideoWriter output_video(land_video_save_path, CV_FOURCC_DEFAULT, 25.0, cv::Size(data.image.cols, data.image.rows));
	//std::cout << data.image.cols << " " << data.image.rows << "\n" << cv::Size(data.image.cols, data.image.rows) << "\n";
	//output_video << data.image;
	//system("pause");
	FILE *fp_t, *fp_a;
	fopen_s(&fp_t, "tslt_save.txt", "w");
	fopen_s(&fp_a, "angle_save.txt", "w");
	Target_type smooth_data[G_gs_window_size+1];
	//cv::Mat smooth_rgb_image[G_gs_window_size];
	cv::VideoWriter gs_smooth_output_video(land_gs_smooth_video_save_path, CV_FOURCC_DEFAULT, 25.0, cv::Size(data.image.cols, data.image.rows));
	cv::Mat lk_frame_last;
	DataPoint lk_data_last;
	for (int test_round = 1;/* test_round < 30*/; test_round++) {
		cap >> rgb_image;
		if (rgb_image.empty()) break;
		cv::cvtColor(rgb_image, data.image, cv::COLOR_BGR2GRAY);
		normalize_gauss_face_rect(data.image, faces[0]);
		/*cv::imshow("nm_gs", data.image);
		cv::waitKey(0);*/
		printf("dde_round %d: \n", test_round);
		//data.image = cv::imread(videoImage+ to_string(test_round) + ".jpg", cv::IMREAD_GRAYSCALE);
		dde_x.dde(rgb_image, data, bldshps, tri_idx, train_data, jaw_land_corr, slt_line, slt_point_rect, exp_r_t_all_matrix);
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

			lk_post_processing(lk_frame_last, rgb_image, lk_data_last, data);
			ceres_post_processing(data, last_data[0], last_data[1], last_data[2], exp_r_t_matrix);
			//recal_dis_ang(data,bldshps);
			data.shape.exp(0) = 1;
			update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
			system("pause");
			//update_slt(
		}
		lk_frame_last = rgb_image.clone();
		lk_data_last = data;

		last_data[2] = data.shape;
		////system("pause");
		puts("errorA");
		fprintf(fp_t, "%.10f %.10f %.10f\n", data.shape.tslt(0), data.shape.tslt(1), (last_data[1].tslt-data.shape.tslt).norm());
		puts("errorB");
		fprintf(fp_a, "%.10f %.10f %.10f %.10f\n", data.shape.angle(0)*180/pi , data.shape.angle(1) * 180 / pi, 
			data.shape.angle(2) * 180 / pi, (last_data[1].angle - data.shape.angle).norm());
		puts("errorC");

		print_datapoint(data);
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
		
		//cv::imshow("G_debug_up_image", G_debug_up_image);
		//cv::waitKey(10);
//------------------------------------------gauss---smooth--------------------------------------------------------
#ifdef usefilter


		for (int j = G_gs_window_size; j >0; j--) smooth_data[j] = smooth_data[j - 1];
		smooth_data[0] = data.shape;
		if (test_round > G_gs_window_size ) {
			DataPoint result=data;			
			result.shape.angle.setZero();
			result.shape.tslt.setZero();
#ifdef gs_filter
			float sum = 0;
			for (int j = 0; j <= G_gs_window_size; j++) {
				float coef = exp(-j * j / (2.0 * G_gs_angle_sig*G_gs_angle_sig));
				result.shape.angle.array() += smooth_data[j].angle.array()*coef;
				result.shape.tslt.array() += smooth_data[j].tslt.array()*coef;
				sum += coef;
			}
			result.shape.angle /= sum;
			result.shape.tslt /= sum;
#endif // gs_filter
#ifdef md_filter
			for (int i_a=0;i_a<G_angle_num;i_a++){
				std::vector<float> temp(G_gs_window_size + 1);
				for (int j = 0; j <= G_gs_window_size; j++)
					temp[j] = smooth_data[j].angle(i_a);
				sort(temp.begin(), temp.end());
				result.shape.angle(i_a) = temp[temp.size()/2];
			}
			for (int i_t = 0; i_t < G_tslt_num; i_t++) {
				std::vector<float> temp(G_gs_window_size + 1);
				for (int j = 0; j <= G_gs_window_size; j++)
					temp[j] = smooth_data[j].tslt(i_t);
				sort(temp.begin(), temp.end());
				result.shape.tslt(i_t) = temp[temp.size() / 2];
			}
#endif // md_filter

			
			dde_x.dde_onlyexpdis(rgb_image, result, bldshps, tri_idx, train_data, jaw_land_corr, slt_line, slt_point_rect, exp_r_t_all_matrix);


			smooth_data[0].dis = result.shape.dis;
			smooth_data[0].exp = result.shape.exp;

			result.shape.dis.setZero();
			result.shape.exp.setZero();
#ifdef gs_filter
			sum = 0;
			for (int j = 0; j <= G_gs_window_size; j++) {
				float coef = exp(-j * j / (2.0 * G_gs_angle_sig*G_gs_angle_sig));
				result.shape.dis.array() += smooth_data[j].dis.array()*coef;
				result.shape.exp.array() += smooth_data[j].exp.array()*coef;
				sum += coef;
			}

			result.shape.dis /= sum;
			result.shape.exp /= sum;
#endif // gs_filter
#ifdef md_filter
			for (int i_e = 0; i_e < G_nShape; i_e++) {
				std::vector<float> temp(G_gs_window_size + 1);
				for (int j = 0; j <= G_gs_window_size; j++)
					temp[j] = smooth_data[j].exp(i_e);
				sort(temp.begin(), temp.end());
				result.shape.exp(i_e) = temp[temp.size() / 2];
			}
			for (int i_d = 0; i_d < 2*G_land_num; i_d++) {
				std::vector<float> temp(G_gs_window_size + 1);
				for (int j = 0; j <= G_gs_window_size; j++)
					temp[j] = smooth_data[j].dis(i_d/2,i_d&1);
				sort(temp.begin(), temp.end());
				result.shape.dis(i_d / 2, i_d & 1) = temp[temp.size() / 2];
			}
#endif // md_filter


			
			//smooth_data[0] = data.shape;
			//result.shape
			data.shape = result.shape;
			data.shape.exp(0) = 1;
			update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
			
			save_datapoint(data, gs_smooth_videolvsave + "_" + to_string(test_round-G_gs_window_size) + ".lv");
			//print_datapoint(data);
			//show_image_0rect(data.image, data.landmarks);
				//save_video(data.image, data.landmarks, output_video);
			save_video(rgb_image, data.landmarks, gs_smooth_output_video);
		}
		
#endif // usefilter
		//for (int j = 0; j < G_gs_window_size - 1; j++) smooth_rgb_image[j] = smooth_rgb_image[j + 1];
		//smooth_rgb_image[G_gs_window_size - 1] = rgb_image.clone();
	}
	fclose(fp_t);
	fclose(fp_a);
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
	cv::VideoWriter output_video("lv_high.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10.0, cv::Size(640, 480));
	long long start_time = cv::getTickCount();
	for (;;)
	{
		vc >> frame;
		cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
		cv::imshow("Gray image", img);
		cv::imshow("gg", frame);
		cv::waitKey(20);
		output_video << frame;
		if ((cv::getTickCount() - start_time) / cv::getTickFrequency() > 10) break;
		std::cout << (cv::getTickCount() - start_time) / cv::getTickFrequency() << "\n";
	}
	puts("over");
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
void find_nearest(DataPoint &data, std::vector<DataPoint> &train_data) {
	puts("change nearest");
	float mi_land = 100000000;
	int idx = 0;
	for (int i = 0; i < train_data.size(); i++) {
		//float distance = (data.center - train_data[i].center).norm();
		float distance_land = 0;// ((data.land_2d.rowwise() - data.center) - (train_data[i].land_2d.rowwise() - train_data[i].center)).norm();
		for (int j = 0; j < G_land_num; j++) {
			//printf("%.5f %.5f %.5f %.5f\n", i);
			Eigen::RowVector2f temp = (data.land_2d.row(j) - data.center - (train_data[i].land_2d.row(j) - train_data[i].center));
			distance_land += temp.norm();
		}
		if (distance_land < mi_land) {
			idx = i;
			mi_land = distance_land;
			printf("idx %d dis%.10f center :%.10f %.10f train_center :%.10f %.10f\n",
				idx, mi_land, data.center(0), data.center(1), train_data[i].center(0), train_data[i].center(1));
		}
	}
	printf("nearest vertex: %d train dataset size: %d\n", idx, train_data.size());

}
void draw_land_train_pic(cv::Mat image,DataPoint &data, DataPoint &training_data, cv::Scalar color) {
	float loss_norm = 0, loss = 0, tot_norm = 0;;
	for (int i = 15; i < G_land_num; i++) {
		//printf("%.5f %.5f %.5f %.5f\n", i);
		Eigen::RowVector2f temp = (data.land_2d.row(i) - data.center - (training_data.land_2d.row(i) - training_data.center));
		loss_norm += temp.norm();
		loss += sqrt(temp(0)*temp(0) + temp(1)*temp(1));
		tot_norm += temp(0)*temp(0) + temp(1)*temp(1);
		temp = training_data.land_2d.row(i) - training_data.center + data.center;
		cv::circle(image, cv::Point2f(temp(0),image.rows- temp(1)), 0.1,color, 2);
	}
	printf("loss %.10f loss_norm %.10f tot %.10f\n", loss, loss_norm,sqrt(tot_norm));
	int i = 0;
	Eigen::RowVector2f temp = (data.land_2d.row(i) - data.center - (training_data.land_2d.row(i) - training_data.center));
	loss_norm = temp.norm();
	loss = sqrt(temp(0)*temp(0) + temp(1)*temp(1));
	printf("++loss %.10f loss_norm %.10f\n", loss, loss_norm);
}
void show_data_land() {

	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) exit(2);//如果视频不能正常打开则返回
	cv::Mat rgb_image;
	cap >> rgb_image;

	DataPoint data;
#ifndef from_video
	data.image = cv::imread(kTestImage, cv::IMREAD_GRAYSCALE);
#endif // !from_video

	load_fitting_coef_one(test_debug_lv_path, data);
	data.landmarks.resize(G_land_num);

	vector<DataPoint> train_data;
	train_data.clear();
	
	load_land_coef(fwhs_path_lv, ".jpg", train_data);
	//load_land_coef(lfw_path_lv, ".jpg", train_data);
	//load_land_coef(gtav_path_lv, ".bmp", train_data);
	printf("%d \n", train_data.size());
	/*print_datapoint(train_data[0]);
	print_datapoint(data);*/

	//std::cout << train_data[0].s << "\n-----------------------\n";
	//std::cout << data.s << "\n";
	//system("pause");
	Eigen::MatrixXf exp_r_t_all_matrix;
	load_bldshps(bldshps, bldshps_path, ide_sg_vl, sg_vl_path);

	//update_2d_ang_land(data, bldshps);

	cal_exp_r_t_all_matrix(bldshps, data, exp_r_t_all_matrix);

	update_training_data(data, train_data, exp_r_t_all_matrix);
	//find_nearest(data, training_data);
	//draw_land_train_pic(rgb_image, data,training_data[1605], cv::Scalar(0, 0, 250));
	//draw_land_train_pic(rgb_image,data, training_data[1585], cv::Scalar(250, 0, 0));

	for (int i=0;i< train_data.size();i++)
		draw_land_train_pic(rgb_image,data, train_data[i], cv::Scalar(0, 0, 250));
	cv::imshow("debug_nearest", rgb_image);
	cv::waitKey(0);

}

void test_cv_mat() {
	float sum = 0;
	for (int j = 0; j <= G_gs_window_size; j++) {
		float coef = exp(-j * j / (2.0 * G_gs_angle_sig*G_gs_angle_sig));
		printf("%.5f ", coef);
		sum += coef;
	}
	puts("");
	for (int j = 0; j <= G_gs_window_size; j++) {
		float coef = exp(-j * j / (2.0 * G_gs_angle_sig*G_gs_angle_sig));
		printf("%.5f ", coef/sum);
	}
	puts("");
	system("pause");
	for (int j = 0; j < G_gs_window_size; j++) {
		sum += exp(-(j - G_gs_window_size / 2)*(j - G_gs_window_size / 2) / (2.0 * G_gs_angle_sig*G_gs_angle_sig));
		printf("%.5f ", exp(-(j - G_gs_window_size / 2)*(j - G_gs_window_size / 2) / (2.0 * G_gs_angle_sig*G_gs_angle_sig)));
	}
	puts("");
	for (int j = 0; j < G_gs_window_size; j++) {
		
		printf("%.5f ", exp(-(j - G_gs_window_size / 2)*(j - G_gs_window_size / 2) / (2.0 * G_gs_angle_sig*G_gs_angle_sig))/sum);
	}
	puts("");
	system("pause");
	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) exit(2);//如果视频不能正常打开则返回
	cv::Mat rgb_image;
	cap >> rgb_image;
	cv::Mat smooth_rgb_image[11];
	for (int test_round = 1;/* test_round < 30*/; test_round++) {
		cap >> rgb_image;
		if (rgb_image.empty()) break;
		for (int j = 0; j < 10; j++) smooth_rgb_image[j] = smooth_rgb_image[j + 1];

		smooth_rgb_image[10] = rgb_image.clone();

		if (test_round > 10) {
			cv::circle(smooth_rgb_image[8], cv::Point(100, 100), 10, cv::Scalar(0, 250, 220), 2);
			cv::circle(smooth_rgb_image[10], cv::Point(100, 100), 10, cv::Scalar(0, 250, 220), 2);
			cv::imshow("test_cv_mat", smooth_rgb_image[7]);
			cv::waitKey(0);

			cv::imshow("test_cv_mat", smooth_rgb_image[8]);
			cv::waitKey(0);
			cv::imshow("test_cv_mat", smooth_rgb_image[9]);
			cv::waitKey(0);

			cv::imshow("test_cv_mat", smooth_rgb_image[10]);
			cv::waitKey(0);

		}
	}
	exit(0);
}
void test_sobel() {
	cv::Mat image = cv::imread(kTestImage),result;
	
	std::vector<cv::Mat> channel;
	cv::split(image, channel);  //分离色彩通道
	cv::imshow("img1", channel[0]);
	cv::imshow("img2", channel[1]);
	cv::imshow("img3", channel[2]);
	cv::imshow("result", image);
	cv::waitKey();
	
	for (int ch = 0; ch < 3; ch++) {
		result = channel[ch].clone();
		for (int i_x = 0; i_x < channel[ch].cols; i_x++)
			for (int i_y = 0; i_y < channel[ch].rows; i_y++) {
				std::pair<float, float> sb = cal_sobel(channel[ch], cv::Point(i_x, i_y));
				//if (i_x == 480) std::cout <<i_x << ' ' << channel[ch].cols <<' ' << max(min(sqrt(sb.first*sb.first + sb.second*sb.second), float(255)), float(0)) << "++\n";
				result.at<uchar>(i_y,i_x) = uchar(max(min(sqrt(sb.first*sb.first + sb.second*sb.second), float(255)), float(0)));

			}
		normalize_gauss_face_rect(result, cv::Rect(0, 0, result.cols, result.rows));
		cv::imshow("result", result);
		channel[ch] = result.clone();
		cv::waitKey();
	}

	cv::merge(channel, image); //合并色彩通道
	cv::imshow("result", image);
	cv::waitKey();
}

void test_lk() {
	DataPoint data = debug_preprocess(test_debug_lv_path);
	puts("testing LK...");

	cv::VideoCapture cap(video_path);
	if (!cap.isOpened()) exit(2);//如果视频不能正常打开则返回
	cv::Mat rgb_image;
	cap >> rgb_image;

	cv::cvtColor(rgb_image, data.image, cv::COLOR_BGR2GRAY);
	cv::CascadeClassifier cc(kAlt2);
	vector<cv::Rect> faces;
	cc.detectMultiScale(data.image, faces);

	rect_scale(faces[0], 1.5);
	
#ifdef norm_lk_face
	std::vector<cv::Mat> channel;
	cv::split(rgb_image, channel);
	for (int ch = 0; ch < channel.size(); ch++)
		normalize_gauss_face_rect(channel[ch], faces[0]);
	cv::merge(channel, rgb_image);
#endif // norm_lk_face
	

	for (int i_v = 0; i_v < G_land_num; i_v++)
		data.landmarks[i_v].x = data.land_2d(i_v, 0), data.landmarks[i_v].y = data.image.rows - data.land_2d(i_v, 1);

	std::vector<cv::Point2f> landmarks_last(G_land_num),landmarks_now(G_land_num), landmarks_inv(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++)
		landmarks_last[i_v].x = data.land_2d(i_v, 0), landmarks_last[i_v].y = data.image.rows - data.land_2d(i_v, 1);

	landmarks_now = landmarks_last;
	//for (int i_v = 0; i_v < G_land_num; i_v++) {
	//	
	//	std::cout << landmarks_last[i_v] << " <<< ldmk\n";
	//}

	cv::VideoWriter output_video(lk_video_save_path, CV_FOURCC_DEFAULT, 25.0, cv::Size(data.image.cols, data.image.rows));
#ifdef lk_rgb_def

	cv::Mat frame_last=rgb_image.clone();
#else
	cv::Mat frame_last = data.image;
#endif
	//show_image_0rect(rgb_image, landmarks_last);

	

	for (int test_round = 1;/* test_round < 30*/; test_round++) {
		cap >> rgb_image;
		if (rgb_image.empty()) break;

#ifndef lk_rgb_def
		cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2GRAY);
#endif // !lk_rgb_def

		

#ifdef norm_lk_face
		cv::split(rgb_image, channel);
		for (int ch = 0; ch < channel.size(); ch++)
			normalize_gauss_face_rect(channel[ch], faces[0]);
		cv::merge(channel, rgb_image);
#endif // norm_lk_face


		printf("test_round %d: \n", test_round);
		//if (test_round == 15) {
		//	cv::imshow("now", rgb_image);
		//	cv::imshow("last", frame_last);
		//	//cv::waitKey();

			int i_v = 55;
			cv::Point p = landmarks_last[i_v];
			

#ifdef lk_opencv_def
			std::vector<uchar> lk_sts(G_land_num);
			std::vector<float> lk_err(G_land_num);
			cv::calcOpticalFlowPyrLK(frame_last, rgb_image, landmarks_last, landmarks_now, lk_sts, lk_err);
			cv::calcOpticalFlowPyrLK(rgb_image,frame_last, landmarks_now, landmarks_inv,  lk_sts, lk_err);
			//for (int i_v = 0; i_v < G_land_num; i_v++)
			//	if (dis_cv_pt(landmarks_inv[i_v], landmarks_last[i_v]) > 5) landmarks_now[i_v] = landmarks_last[i_v];
			cv::Point p_next = landmarks_now[i_v];
			cv::Point p_inv = landmarks_inv[i_v];
			printf("%d %.5f\n", lk_sts[i_v], lk_err[i_v]);
			std::cout << p << "<-last  next->" << p_next << "  inv -> " << p_inv <<  "\n";
#else
			std::cout << landmarks_last[i_v] << " ldmk " << p << " <<< p\n";
			cv::Point p_next = lk_get_pos_next(10, p, frame_last, rgb_image);
			cv::Point p_inv = lk_get_pos_next(10, p_next, rgb_image, frame_last);
			std::cout << p << "<-last  next->" << p_next << "  inv -> " << p_inv << "\n";
			
#endif // lk_opencv_def

			


			cv::circle(rgb_image, p, 1, cv::Scalar(255,0, 0), 2);
			cv::circle(rgb_image, p_next, 1, cv::Scalar(0, 255, 0), 2);
			cv::circle(rgb_image, p_inv, 1, cv::Scalar(0, 0, 255), 2);

			
			cv::imshow("res", rgb_image);
			cv::waitKey();
			landmarks_now[i_v] = p_next;
		//}
		//for (int i_v = 0; i_v < G_land_num; i_v++) {
		//
		//	cv::Point p=landmarks_last[i_v];
		//	cv::Point p_next = lk_get_pos_next(G_lk_batch_size, p, frame_last, rgb_image);
		//	cv::Point p_inv = lk_get_pos_next(G_lk_batch_size, p_next, rgb_image, frame_last);
		//	std::cout << p << "<-last  next->" << p_next << "  inv:" << p_inv << "\n";
		//	landmarks_now[i_v] = p_next;
		//}
		//
		//show_image_0rect(rgb_image, landmarks_now);
	
		save_video(rgb_image, landmarks_now, output_video);
		landmarks_last = landmarks_now;
		frame_last = rgb_image.clone();
	}

}

int main()
{
	test_lk();
	return 0;

	//test_sobel();
	//test_cv_mat();
	
	//show_data_land();

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

	//show_data_land();
#ifdef win64
	try
	{	
#endif
#ifndef debug_def
		puts("initializing face...");
		FaceX face_x(kModelFileName);
		//Tracking_face(face_x);
#endif // !debug_def

		
		puts("initializing dde...");
		DDEX dde_x(kModelFileName_dde);
		load_inner_land_corr(inner_land_corr,inner_cor_path);
		load_jaw_land_corr(jaw_land_corr,jaw_land_path);
		//std::cout << inner_land_corr << '\n';
		load_slt(slt_line, slt_point_rect, slt_path, rect_path);
		load_bldshps(bldshps, bldshps_path, ide_sg_vl, sg_vl_path);
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

		/*DataPoint init_data = pre_process(face_x, dde_x, bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl, training_data, tri_idx);
		save_datapoint(init_data, test_debug_lv_path);
		return 0;*/

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
#ifdef win64
	}
	catch (const runtime_error& e)
	{
		cerr << e.what() << endl;
	}
#endif // !linux
	return 0;
}
/*
1 grep -rl 'fopen_s(&fp,' ./ | xargs sed -i 's/fopen_s(&fp,/fp=fopen(/g'
2
3
4 grep -rl 'fscanf_s' ./ | xargs sed -i 's/fscanf_s/fscanf/g'


grep -rl 'fopen_s(&fpr,' ./ | xargs sed -i 's/fopen_s(&fpr,/fpr=fopen(/g'
grep -rl 'fopen_s(&fpw,' ./ | xargs sed -i 's/fopen_s(&fpw,/fpw=fopen(/g'
*/

/*
		for (int j = 0; j < G_gs_window_size-1; j++) smooth_data[j] = smooth_data[j + 1];
		smooth_data[G_gs_window_size-1] = data.shape;
		for (int j = 0; j < G_gs_window_size-1; j++) smooth_rgb_image[j] = smooth_rgb_image[j + 1];
		smooth_rgb_image[G_gs_window_size-1] = rgb_image.clone();


		if (test_round > G_gs_window_size-1) {
			DataPoint result=data;
			result.shape= smooth_data[G_gs_window_size / 2];


			result.shape.angle.setZero();
			float sum = 0;
			for (int j = 0; j < G_gs_window_size ; j++) {
				result.shape.angle.array() += smooth_data[j].angle.array()*exp(-(j - G_gs_window_size/2)*(j - G_gs_window_size/2) / 2.0);
				sum += exp(-(j - G_gs_window_size/2)*(j - G_gs_window_size/2) / 2.0);
			}
			result.shape.angle.array() /= sum;
			cv::cvtColor(smooth_rgb_image[G_gs_window_size/2], result.image, cv::COLOR_BGR2GRAY);
			puts("A");
			dde_x.dde_onlyexpdis(smooth_rgb_image[G_gs_window_size/2], result, bldshps, tri_idx, train_data, jaw_land_corr, slt_line, slt_point_rect, exp_r_t_all_matrix);

			save_datapoint(result, gs_smooth_videolvsave + "_" + to_string(test_round-G_gs_window_size+1) + ".lv");
			//print_datapoint(data);
			//show_image_0rect(data.image, data.landmarks);
				//save_video(data.image, data.landmarks, output_video);
			save_video(smooth_rgb_image[G_gs_window_size/2], result.landmarks, gs_smooth_output_video);
		}
*/