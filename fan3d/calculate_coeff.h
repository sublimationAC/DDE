#pragma once
#include "2dland.h"

void init_exp_ide_r_t_pq(iden *ide, int ide_num, Eigen::VectorXi &land_cor);

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name, Eigen::VectorXf &ide_sg_vl, std::string sg_vl_path);
void print_bldshps(Eigen::MatrixXf &bldshps);


//void cal_f(
//	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
//	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
//	Eigen::VectorXf &ide_sg_vl);
//void solve(
//	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
//	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
//	Eigen::VectorXf &ide_sg_vl);
//
//
//float pre_cal_exp_ide_R_t(
//	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor, Eigen::VectorXi &jaw_land_corr,
//	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, int id_idx,
//	Eigen::VectorXf &ide_sg_vl);
void init_exp_ide(iden *ide, int train_id_num ,int id_idx);

//void cal_rt_posit(
//	float f, iden *ide, Eigen::MatrixXf &bldshps,
//	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);
//void test_posit(
//	float f, iden *ide, Eigen::MatrixXf &bldshps,
//	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);
//
//void cal_rt_normalization(
//	iden *ide, Eigen::MatrixXf &bldshps,
//	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);
//void test_normalization(
//	iden *ide, Eigen::MatrixXf &bldshps,
//	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);
//
//void cal_inner_bldshps(
//	iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &bs_in,
//	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);

float cal_3d_vtx(
	iden *ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, int vtx_idx, int axis);


void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result);


void cal_id_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result);

//float cal_fixed_exp_same_ide(
//	float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx,
//	Eigen::VectorXf &ide_sg_vl);




//ceres
float ceres_exp_one(
	iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::RowVectorXf &exp, Eigen::Matrix3f &rot);

float ceres_user_one(
	iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &id_point, Eigen::VectorXf &user, Eigen::Matrix3f &rot,
	Eigen::VectorXf &ide_sg_vl);

//float ceres_user_fixed_exp(
//	iden *ide, int id_idx, Eigen::MatrixXf &id_point_fix_exp, Eigen::VectorXf &user,
//	Eigen::VectorXf &ide_sg_vl);



//bfgs
//float bfgs_exp_one(float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::VectorXf &exp);



void test_coef_land(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx);
void test_coef_mesh(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx);
void test_2dland(float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx);
void cal_mesh_land(Eigen::MatrixXf &bldshps);
void cal_mesh_land_exp_only(Eigen::MatrixXf &bldshps);


void solve_3d(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &land_corr, Eigen::VectorXf &ide_sg_vl);

float cal_exp_ide_R_3d(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &land_cor, int id_idx,
	Eigen::VectorXf &ide_sg_vl);

void cal_3d_R_t(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx);
void test_cal_3d_R_t(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx);



float cal_3d_exp(
	iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor);
float cal_3d_ide(
	iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::VectorXf &ide_sg_vl);

Eigen::Vector3f get_uler_angle_zyx(Eigen::Matrix3f R);