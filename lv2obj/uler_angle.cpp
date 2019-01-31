//#include <dirent.h>
//#include <iostream>
//#include <vector>
//#include <string>
//#include <utility>
//
//#include <Eigen/Core>
//#include <Eigen/LU>
//#include <Eigen/Geometry>
//
//#include <opencv2/opencv.hpp>
//#define pi acos(-1)
//
//const int G_land_num = 74;
//const int G_train_pic_id_num = 3300;
//const int G_nShape = 47;
//const int G_nVerts = 11510;
//const int G_nFaces = 11540;
//const int G_test_num = 77;
//const int G_iden_num = 77;
//const int G_inner_land_num = 59;
//const int G_line_num = 50;
//const int G_jaw_land_num = 20;
//#define normalization
//struct Target_type {
//	Eigen::VectorXf exp;
//	Eigen::RowVector3f tslt;
//	Eigen::Matrix3f rot;
//	Eigen::MatrixX2f dis;
//	Eigen::Vector3f angle;
//};
//
//struct DataPoint
//{
//	cv::Mat image;
//	cv::Rect face_rect;
//	std::vector<cv::Point2d> landmarks;
//	//std::vector<cv::Point2d> init_shape;
//	Target_type shape, init_shape;
//	Eigen::VectorXf user;
//	Eigen::RowVector2f center;
//	Eigen::MatrixX2f land_2d;
//#ifdef posit
//	float f;
//#endif // posit
//#ifdef normalization
//	Eigen::MatrixX3f s;
//#endif
//
//	Eigen::VectorXi land_cor;
//};
//
//
//void load_lv(std::string name, DataPoint &temp) {
//	std::cout << "load coefficients...file:" << name << "\n";
//	FILE *fp;
//	fopen_s(&fp, name.c_str(), "rb");
//
//	temp.user.resize(G_iden_num);
//	for (int j = 0; j < G_iden_num; j++)
//		fread(&temp.user(j), sizeof(float), 1, fp);
//	std::cout << temp.user << "\n";
//	//system("pause");
//	temp.land_2d.resize(G_land_num, 2);
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		fread(&temp.land_2d(i_v, 0), sizeof(float), 1, fp);
//		fread(&temp.land_2d(i_v, 1), sizeof(float), 1, fp);
//	}
//
//
//	fread(&temp.center(0), sizeof(float), 1, fp);
//	fread(&temp.center(1), sizeof(float), 1, fp);
//
//	temp.shape.exp.resize(G_nShape);
//	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
//		fread(&temp.shape.exp(i_shape), sizeof(float), 1, fp);
//
//	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
//		fread(&temp.shape.rot(i, j), sizeof(float), 1, fp);
//
//	for (int i = 0; i < 3; i++) fread(&temp.shape.tslt(i), sizeof(float), 1, fp);
//
//	temp.land_cor.resize(G_land_num);
//	for (int i_v = 0; i_v < G_land_num; i_v++) fread(&temp.land_cor(i_v), sizeof(int), 1, fp);
//
//	temp.s.resize(2, 3);
//	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
//		fread(&temp.s(i, j), sizeof(float), 1, fp);
//
//	temp.shape.dis.resize(G_land_num, 2);
//	for (int i_v = 0; i_v < G_land_num; i_v++) {
//		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
//		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
//	}
//	std::cout << temp.shape.dis << "\n";
//	//system("pause");
//	fclose(fp);
//	puts("load successful!");
//}
//
//
////assume the be could not be more than 90
//void cal_uler_angle(Eigen::Matrix3f R) {
//	Eigen::Vector3f x, y, z,t;
//	x = R.row(0).transpose();
//	y = R.row(1).transpose();
//	z = R.row(2).transpose();
//	float al, be, ga, gaw;
//	if (fabs(1 - z(2)*z(2)) < 1e-3) {
//		ga=gaw=be = 0;
//		al = acos(x(0));
//		if (y(0) < 0) al = 2 * pi - al;
//	}
//	else {
//		
//		be = acos(z(2));
//		al = acos(std::max(std::min(float(1.0),z(1) / sqrt(1 - z(2)*z(2))),float(-1.0)));
//		
//		if (z(0) < 0) al = 2 * pi - al;//according to the sin(al)
//
//
//		t(0) = cos(al), t(1) = sin(al), t(2) = 0;
//		t.normalize();
//		x.normalize();
//		//t.normalized();
//		ga = acos(t.dot(x));
//		gaw = acos(std::max(std::min(float(1.0), -y(2) / sqrt(1 - z(2)*z(2))), float(-1.0)));
//
//		printf("%.10f %.10f %.10f\n", -y(2), sqrt(1 - z(2)*z(2)), -y(2) / sqrt(1 - z(2)*z(2)));
//		if (x(2) < 0) ga = 2 * pi - ga, gaw = 2 * pi - gaw;//according to the sin(ga)
//	}
//	std::cout << R << "\n----------------------\n";
//	printf("%.10f %.10f %.10f %.10f %.10f\n",z(2), al/pi*180, be / pi * 180, ga / pi * 180, gaw / pi * 180);
//	system("pause");
//}
//
//Eigen::Matrix3f get_r_from_angle(float angle, int axis) {
//	Eigen::Matrix3f ans;
//	ans.setZero();
//	ans(axis, axis) = 1;
//	int idx_x = 0, idx_y = 1;
//	if (axis == 0)
//		idx_x = 1, idx_y = 2;
//	else
//		if (axis == 2)
//			idx_x = 0, idx_y = 1;
//		else
//			idx_x = 0, idx_y = 2;
//	ans(idx_x, idx_x) = cos(angle), ans(idx_x, idx_y) = -sin(angle), ans(idx_y, idx_x) = sin(angle), ans(idx_y, idx_y) = cos(angle);
//	return ans;
//}
//
//
//Eigen::Matrix3f get_r_from_angle(const Eigen::Vector3f &angle) {
//	Eigen::Matrix3f ans;
//	float Sa = sin(angle(0)), Ca = cos(angle(0)), Sb = sin(angle(1)),
//		Cb = cos(angle(1)), Sc = sin(angle(2)), Cc = cos(angle(2));
//
//	ans(0, 0) = Ca * Cc - Sa * Cb*Sc;
//	ans(0, 1) = -Sa * Cc - Ca * Cb*Sc;
//	ans(0, 2) = Sb * Sc;
//	ans(1, 0) = Ca * Sc + Sa * Cb*Cc;
//	ans(1, 1) = -Sa * Sc + Ca * Cb*Cc;
//	ans(1, 2) = -Sb * Cc;
//	ans(2, 0) = Sa * Sb;
//	ans(2, 1) = Ca * Sb;
//	ans(2, 2) = Cb;
//	return ans;
//}
//
////assume the be could not be more than 90
//Eigen::Vector3f cal_uler_angle_zyx(Eigen::Matrix3f R) {
//	Eigen::Vector3f x, y, z, t;
//	x = R.row(0).transpose();
//	y = R.row(1).transpose();
//	z = R.row(2).transpose();
//	float al, be, ga;
//	if (fabs(1 - x(2)*x(2)) < 1e-3) {
//		be = asin(x(2));
//		al = ga = 0;
//		exit(1);
//	}
//	else {
//
//		be = asin(std::max(std::min(1.0, double(x(2))), -1.0));
//		al = asin(std::max(std::min(1.0, double(-x(1) / sqrt(1 - x(2)*x(2)))), -1.0));
//		ga = asin(std::max(std::min(1.0, double(-y(2) / sqrt(1 - x(2)*x(2)))), -1.0));
//
//	}
//	std::cout << R << "\n----------------------\n";
//	printf("%.10f %.10f %.10f %.10f\n", x(2), al / pi * 180, be / pi * 180, ga / pi * 180);
//	Eigen::Vector3f ans;
//	ans << al, be, ga;
//	return ans;
//	//system("pause");
//}
//
//Eigen::Matrix3f get_r_from_angle_zyx(const Eigen::Vector3f &angle) {
//	Eigen::Matrix3f ans;
//	float Sa = sin(angle(0)), Ca = cos(angle(0)), Sb = sin(angle(1)),
//		Cb = cos(angle(1)), Sc = sin(angle(2)), Cc = cos(angle(2));
//
//	ans(0, 0) = Ca * Cb; 
//	ans(0, 1) = -Sa * Cb; 
//	ans(0, 2) = Sb;
//	ans(1, 0) = Sa * Cc + Ca * Sb*Sc; 
//	ans(1, 1) = Ca * Cc - Sa * Sb*Sc; 
//	ans(1, 2) = -Cb * Sc;
//	ans(2, 0) = Sa * Sc - Ca * Sb*Cc;
//	ans(2, 1) = Ca * Sc + Sa * Sb*Cc;
//	ans(2, 2) = Cb * Cc;
//	return ans;
//}
//
//
//
//
//void test_r(DataPoint data) {
//	Eigen::Matrix3f rot;
//	//rot = get_r_from_angle(data.shape.tslt(2), 2)*get_r_from_angle(data.shape.tslt(1), 0)*get_r_from_angle(data.shape.tslt(0), 2);
//	//std::cout << rot << "\n";
//	std::cout << get_r_from_angle_zyx(data.shape.angle) << "\n";
//	system("pause");
//}
//
//int main() {
//
//	DataPoint data;
//	/*data.shape.angle << 1, 20, 0.5;
//	test_r(data);*/
//
//
//	load_lv("data/lv_mp4_1.lv",data);//data/test_debug_lv_005_04_03_051_05
//	//lv_mp4_1.
//	data.shape.angle=cal_uler_angle_zyx(data.shape.rot);
//	test_r(data);
//	return 0;
//}
////g++ -Wall -std=c++11 `pkg-config --cflags opencv` -o deal deal_falut.cpp `pkg-config --libs opencv`