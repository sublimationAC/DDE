#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
const int G_land_num = 73;
const int G_train_pic_id_num = 3300;
const int G_nShape = 47;
const int G_nVerts = 11510;
const int G_nFaces = 11540;
const int G_test_num = 77;
const int G_iden_num = 77;
const int G_inner_land_num = 58;


//#define normalization
#define perspective

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
	
#ifdef perspective
	
	float fcs;
#endif // posit
#ifdef normalization
	Eigen::MatrixX3f s;
#endif

	Eigen::VectorXi land_cor;
};
void data_err_cnt_one(std::string name, Eigen::VectorXi &err_cnt) {
	std::cout << "load coefficients...file:" << name << "\n";
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");
	DataPoint temp;
	temp.user.resize(G_iden_num);
	for (int j = 0; j < G_iden_num; j++)
		fread(&temp.user(j), sizeof(float), 1, fp);

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

#ifdef normalization
	temp.s.resize(2, 3);
	for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
		fread(&temp.s(i, j), sizeof(float), 1, fp);
#endif // normalization

#ifdef perspective
	fread(&temp.fcs, sizeof(float), 1, fp);
	std::cout << "f/tslt_z: " << temp.fcs / temp.shape.tslt(2) << " " << temp.shape.tslt(2) / temp.fcs << "\n";

	//temp.fcs = temp.fcs / temp.shape.tslt(2) * 10;
	//temp.shape.tslt(2) = 10;

#endif // perspective

	temp.shape.dis.resize(G_land_num, 2);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		fread(&temp.shape.dis(i_v, 0), sizeof(float), 1, fp);
		fread(&temp.shape.dis(i_v, 1), sizeof(float), 1, fp);
	}

	fclose(fp);

	puts("load successfully");
	

	float err = sqrt(temp.shape.dis.squaredNorm() / G_land_num / 2);
	//float err = temp.user.sum()*10;

	printf("err %.5f\n", err);
	if (err<50) err_cnt((int)(err))++;
	else {
		err_cnt(49)++;
		std::cout << temp.shape.dis << "\n";
		exit(9);		
	}

}

//
int cnt = 0;
void data_err_cnt(std::string path, Eigen::VectorXi &err_cnt) {

	struct dirent **namelist;
	int n;
	n = scandir(path.c_str(), &namelist, 0, alphasort);
	if (n < 0)
	{
		std::cout << "scandir return " << n << "\n";
		perror("Cannot open .");
		exit(1);
	}
	else
	{
		int index = 0;
		struct dirent *dp;
		while (index < n)
		{
			dp = namelist[index];
			std::cout << dp->d_name << ' ' << strlen(dp->d_name) << "\n";
			if (dp->d_name[0] == '.') {
				free(namelist[index]);
				index++;
				continue;
			}
			if (dp->d_type == DT_DIR) {
				data_err_cnt(path + "/" + dp->d_name, err_cnt);
			}
			else {
				int len = strlen(dp->d_name);
				if (dp->d_name[len - 3] == 'p' && dp->d_name[len - 4] == 's' &&
					dp->d_name[len - 1] == 'f' && dp->d_name[len - 2] == '_') {
					////	
					std::string p = path + "/" + dp->d_name;
					data_err_cnt_one(p, err_cnt);
					std::cout << "cnt" << cnt << "\n";
					cnt++;
				}
			}
			free(namelist[index]);
			index++;
		}
		free(namelist);
	}

}




std::string fwhs_path = "./fw";
//std::string fwhs_path_p = "./data_me/fw_p1";
std::string lfw_path = "./lfw_image";
std::string gtav_path = "./GTAV_image";
std::string test_path = "D:/sydney/first/data_me/test_lv";




int main() {
	Eigen::VectorXi err_cnt(50);
	err_cnt.setZero();
	data_err_cnt(fwhs_path, err_cnt);

	//data_err_cnt(lfw_path,err_cnt);
	//data_err_cnt(gtav_path,err_cnt);
	
	std::cout << "cnt" << cnt << "\n";
	std::cout << "err cnt" << err_cnt.transpose() << "\n";
	return 0;
}
//g++ -Wall -std=c++11 `pkg-config --cflags opencv` -o deal deal_falut.cpp `pkg-config --libs opencv`

