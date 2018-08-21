#pragma once
#include "2dland.h"

void init_exp_ide_r_t_pq(iden *ide, int ide_num);

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name);

void cal_f(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect);
void init_exp_ide(iden *ide, int train_id_num);

float pre_cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor,
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, int id_idx);

void cal_rt_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);

void cal_inner_bldshps(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &bs_in,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);

float cal_3d_vtx(
	iden *ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, int vtx_idx, int axis);

void test_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);

void update_slt(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::VectorXi &out_land_cor);

void test_slt(float f ,iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx);

float cal_3dpaper_exp(
	float f, iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor);

void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result);













//bfgs
float bfgs_exp_one(float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::VectorXf &exp);