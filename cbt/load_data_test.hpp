#pragma once
#include "dde_x.hpp"



void load_land_coef(std::string path, std::string sfx, std::vector<DataPoint> &img);

void load_land(std::string p, DataPoint &temp);

void load_img(std::string p, DataPoint &temp);

void load_land(std::string p, DataPoint &temp);

void cal_rect(DataPoint &temp);

void load_fitting_coef_one(std::string name, DataPoint &temp);

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name, Eigen::VectorXf &ide_sg_vl, std::string sg_vl_path);

void load_inner_land_corr(Eigen::VectorXi &cor, std::string &name);
void load_jaw_land_corr(Eigen::VectorXi &jaw_cor, const std::string &name);
void load_slt(
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	std::string path_slt, std::string path_rect);
