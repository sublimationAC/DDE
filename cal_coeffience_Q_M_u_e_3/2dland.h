#pragma once
//#include "math_headers.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <vector>
#include <utility>
#include <cv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstring>
#include <cstdio>
#define win64
//#define linux
#ifdef win64
	#include <io.h> 
#endif // win64
#ifdef linux
	#include <dirent.h>
#endif // linux

//#define posit
#define normalization


#define EPSILON 1e-6


const int G_land_num = 74;
const int G_train_pic_id_num = 3300;
const int G_nShape = 47;
const int G_nVerts = 11510;
const int G_nFaces = 11540;
const int G_test_num = 77;
const int G_iden_num = 25;
const int G_inner_land_num = 59;
const int G_line_num = 50;
const int G_jaw_land_num = 20;

struct iden
{
	int num = 0;
	Eigen::MatrixX2f land_2d;
	Eigen::MatrixX2f center;
	Eigen::MatrixXf exp;
	Eigen::VectorXf user;
	Eigen::MatrixX3f rot;
	Eigen::MatrixX3f tslt;
	Eigen::MatrixXi land_cor;
	Eigen::MatrixX3f s;
	float fcs;
};

void load_img_land(std::string path, std::string sfx, iden *ide, int &id_idx,std::vector< std::vector<cv::Mat_<uchar> > > &imgs);
int load_img_land_same_id(std::string path, std::string sfx, iden *ide, int id_idx, std::vector<cv::Mat_<uchar> > &img_temp);

void load_land(std::string p, iden *ide, int id_idx);
void load_img(std::string p, cv::Mat_<uchar> &temp);

void test_data_2dland(
	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
	iden *ide, int id_idx, int img_idx);

void load_inner_land_corr(Eigen::VectorXi &cor);
void load_jaw_land_corr(Eigen::VectorXi &jaw_cor);
void load_slt(
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	std::string path_slt, std::string path_rect);

void test_3d22dland(cv::Mat_<uchar> img, std::string path, iden *ide, int id_idx, int exp_idx);