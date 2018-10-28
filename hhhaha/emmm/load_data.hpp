#pragma once
#include "dde_x.h"

#ifdef win64

std::string fwhs_path = "D:/sydney/first/data_me/FaceWarehouse";
std::string lfw_path = "D:/sydney/first/data_me/lfw_image";
std::string gtav_path = "D:/sydney/first/data_me/GTAV_image";
std::string test_path = "D:/sydney/first/data_me/test";
std::string test_path_one = "D:/sydney/first/data_me/test_only_one";
std::string bldshps_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_77.lv";
std::string sg_vl_path = "D:\\sydney\\first\\code\\2017\\deal_data_2\\deal_data/blendshape_ide_svd_value_sqrt_77.txt";
std::string slt_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/sillht.txt";
std::string rect_path = "D:\\openframework\\of_v0.10.0_vs2017_release\\apps\\3d22d\\3d22d/slt_point_rect.txt";
std::string save_coef_path = "./ide_fw_p1.lv";

#endif // win64
#ifdef linux
std::string fwhs_path = "./data_me/FaceWarehouse";
std::string fwhs_path_p = "./data_me/fw_p1";
std::string lfw_path = "./data_me/lfw_image";
std::string gtav_path = "./data_me/GTAV_image";
std::string test_path = "./data_me/test";
std::string test_path_one = "./data_me/test_only_one";
std::string test_path_two = "./data_me/test_only_two";
std::string test_path_three = "./data_me/test_only_three";
std::string bldshps_path = "./deal_data/blendshape_ide_svd_77.lv";
std::string sg_vl_path = "./deal_data/blendshape_ide_svd_value_sqrt_77.txt";
std::string slt_path = "./3d22d/sillht.txt";
std::string rect_path = "./3d22d/slt_point_rect.txt";
std::string save_coef_path = "../fitting_coef/ide_fw_p1.lv";
#endif // linux



void load_img_land_coef(std::string path, std::string sfx, std::vector<DataPoint> &img);

void load_land(std::string p, DataPoint &temp);

void load_img(std::string p, DataPoint &temp);

void load_land(std::string p, DataPoint &temp);

void cal_rect(DataPoint &temp);

void load_fitting_coef_one(std::string name, DataPoint &temp);

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name, Eigen::VectorXf &ide_sg_vl, std::string sg_vl_path);

void load_inner_land_corr(Eigen::VectorXi &cor);
void load_jaw_land_corr(Eigen::VectorXi &jaw_cor);
void load_slt(
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	std::string path_slt, std::string path_rect);
