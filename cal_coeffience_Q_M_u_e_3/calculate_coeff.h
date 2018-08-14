#pragma once
#include "2dland.h"

void init_r_t_pq(iden *ide,int ide_num);

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name);

void init_exp_ide(iden *ide, int ide_num);
float cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, Eigen::MatrixXf &slt);

void cal_rt_posit(float f,iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor, int id_idx,int exp_idx);
void cal_inner_bldshps(iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f bs_in,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);