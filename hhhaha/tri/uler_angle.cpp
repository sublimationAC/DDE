#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#define pi acos(-1)

const int G_land_num = 74;
const int G_train_pic_id_num = 3300;
const int G_nShape = 47;
const int G_nVerts = 11510;
const int G_nFaces = 11540;
const int G_test_num = 77;
const int G_iden_num = 77;
const int G_inner_land_num = 59;
const int G_line_num = 50;
const int G_jaw_land_num = 20;
#define normalization
struct Target_type {
	Eigen::VectorXf exp;
	Eigen::RowVector3f tslt;
	Eigen::Matrix3f rot;
	Eigen::MatrixX2f dis;

};

struct DataPoint
{
	cv::Mat image;
	cv::Rect face_rect;
	std::vector<cv::Point2d> landmarks;
	//std::vector<cv::Point2d> init_shape;
	Target_type shape, init_shape;
	Eigen::VectorXf user;
	Eigen::RowVector2f center;
	Eigen::MatrixX2f land_2d;
#ifdef posit
	float f;
#endif // posit
#ifdef normalization
	Eigen::MatrixX3f s;
#endif

	Eigen::VectorXi land_cor;
};


void load_lv(std::string name, DataPoint &temp) {
	std::cout << "load coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");

	temp.user.resize(G_iden_num);
	for (int j = 0; j < G_iden_num; j++)
		fread(&temp.user(j), sizeof(float), 1, fp);
	std::cout << temp.user << "\n";
	system("pause");
	temp.land_2d.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.land_2d(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.land_2d(i_v, 1), sizeof(float), 1, fp);
	}


	fread(&temp.center(0), sizeof(float), 1, fp);
	fread(&temp.center(1), sizeof(float), 1, fp);

	temp.shape.exp.resize(G_nShape);
	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		fread(&temp.shape.exp(i_shape), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		fread(&temp.shape.rot(i, j), sizeof(float), 1, fp);

	for (int i = 0; i < 3; i++) fread(&temp.shape.tslt(i), sizeof(float), 1, fp);

	temp.land_cor.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) fread(&temp.land_cor(i_v), sizeof(int), 1, fp);

	temp.s.resize(2, 3);
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fread(&temp.s(i, j), sizeof(float), 1, fp);

	temp.shape.dis.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}
	std::cout << temp.shape.dis << "\n";
	system("pause");
	fclose(fp);
	puts("load successful!");
}


//assume the be could not be more than 90
void cal_uler_angle(Eigen::Matrix3f R) {
	Eigen::Vector3f x, y, z,t;
	x = R.row(0).transpose();
	y = R.row(1).transpose();
	z = R.row(2).transpose();
	float al, be, ga, gaw;
	if (fabs(1 - z(2)*z(2)) < 1e-3) {
		ga=gaw=be = 0;
		al = acos(x(0));
		if (y(0) < 0) al = 2 * pi - al;
	}
	else {
		
		be = acos(z(2));
		al = acos(std::max(std::min(float(1.0),z(1) / sqrt(1 - z(2)*z(2))),float(-1.0)));
		
		if (z(0) < 0) al = 2 * pi - al;//according to the sin(al)


		t(0) = cos(al), t(1) = sin(al), t(2) = 0;
		t.normalize();
		x.normalize();
		//t.normalized();
		ga = acos(t.dot(x));
		gaw = acos(std::max(std::min(float(1.0), -y(2) / sqrt(1 - z(2)*z(2))), float(-1.0)));

		printf("%.10f %.10f %.10f\n", -y(2), sqrt(1 - z(2)*z(2)), -y(2) / sqrt(1 - z(2)*z(2)));
		if (x(2) < 0) ga = 2 * pi - ga, gaw = 2 * pi - gaw;//according to the sin(ga)
	}
	std::cout << R << "\n----------------------\n";
	printf("%.10f %.10f %.10f %.10f %.10f\n",z(2), al/pi*180, be / pi * 180, ga / pi * 180, gaw / pi * 180);
	system("pause");
}

int main() {

	DataPoint data;
	load_lv("./test/pose_4_t108.lv",data);//data/test_debug_lv_005_04_03_051_05
	cal_uler_angle(data.shape.rot);
	
	return 0;
}
//g++ -Wall -std=c++11 `pkg-config --cflags opencv` -o deal deal_falut.cpp `pkg-config --libs opencv`