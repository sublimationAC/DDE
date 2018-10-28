#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "calculate_coeff_dde.h"
#include "face_x.h"

using namespace std;

const string kModelFileName = "model_all_30_changed.xml.gz";
const string kModelFileName_dde = "model_dde.xml.gz";
const string kAlt2 = "haarcascade_frontalface_alt2.xml";
const string kTestImage = "./photo_test/test_samples/22.png";

//#define win64
#define linux


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


#include <algorithm>
void pre_process(
	const FaceX & face_x,const DDEX &dde_x,Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl,vector<DataPoint> &train_data,Eigen::MatrixX3i &tri_idx)
{
	//cout << "picture name:";
	//string pic_name;
	//cin >> pic_name;
	cv::Mat image = cv::imread(kTestImage);// +pic_name);
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, CV_BGR2GRAY);
	cv::CascadeClassifier cc(kAlt2);
	if(cc.empty())
	{
		cout << "Cannot open model file " << kAlt2 << " for OpenCV face detector!" << endl;
		return;
	}
	vector<cv::Rect> faces;
	double start_time = cv::getTickCount();
	cc.detectMultiScale(gray_image, faces);
	cout << "Detection time: " << (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << endl;

	/*for (cv::Rect face : faces)
	{*/
	cv::Rect face =faces[0];
		face.x = max(0, face.x - 10);// face.y = max(0, face.y - 10);
		face.width = std :: min(gray_image.rows - face.x, face.width + 25);
		face.height = std :: min(gray_image.cols - face.y, face.height + 25);
		cv::rectangle(image, face, cv::Scalar(0, 0, 255), 2);
		start_time = cv::getTickCount();
		vector<cv::Point2d> landmarks = face_x.Alignment(gray_image, face);
		cout << "Alignment time: " 
			<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
			<< "s" << endl;
		for (cv::Point2d landmark : landmarks)
		{
			cv::circle(image, landmark, 0.1, cv::Scalar(0, 255, 0), 2);
		}
	//}
	cv::imshow("Alignment result", image);
	cv::waitKey();

	//fitting
	DataPoint data;
	data.image = image;
	
	fit_solve(landmarks,bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl, data);

	dde_x.dde(data, bldshps,tri_idx,train_data);

}

void Tracking(const FaceX & face_x)
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
		int key = cv::waitKey(10);
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

Eigen::MatrixXf bldshps(G_iden_num, G_nShape * 3 * G_nVerts);
Eigen::VectorXf ide_sg_vl(G_iden_num);
Eigen::VectorXi inner_land_corr(G_inner_land_num);
Eigen::VectorXi jaw_land_corr(G_jaw_land_num);
std::vector<std::pair<int, int> > slt_point_rect[G_nVerts];
std::vector<int> slt_line[G_line_num];
Eigen::VectorXf ide_sg_vl(G_iden_num);
int main()
{
	try
	{
		FaceX face_x(kModelFileName);
		DDEX dde_x(kModelFileName_dde);
		load_inner_land_corr(inner_land_corr);
		load_jaw_land_corr(jaw_land_corr);
		//std::cout << inner_land_corr << '\n';
		load_slt(slt_line, slt_point_rect, slt_path, rect_path);
		load_bldshps(bldshps, bldshps_path, ide_sg_vl, sg_vl_path);
		vector<DataPoint> training_data;
		training_data.clear();
		load_img_land_coef(fwhs_path, ".jpg", training_data);

		Eigen::MatrixX3i tri_idx;
		std::vector<cv::Vec6f> triangleList;
		cv::Rect rect;
		std::vector<cv::Point2d> ref_shape = dde_x.get_ref_shape();
		cal_del_tri(ref_shape, rect, triangleList, tri_idx);

		cout << "Choice: " << endl;
		cout << "1. Align " << kTestImage << " in the current working directory." << endl;
		cout << "2. Align video from web camera." << endl;
		cout << "Please select one [1/2]: ";
		int choice;
		cin >> choice;
		switch (choice)
		{
		case 1:
			pre_process(face_x,dde_x,bldshps, inner_land_corr, jaw_land_corr, slt_line, slt_point_rect, ide_sg_vl,training_data, tri_idx);
			break;
		case 2:
			Tracking(face_x);
			break;
		}
	}
	catch (const runtime_error& e)
	{
		cerr << e.what() << endl;
	}
}
