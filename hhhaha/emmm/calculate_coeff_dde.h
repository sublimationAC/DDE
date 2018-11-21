#pragma once
#include "post_processing.hpp"

struct iden
{
	int num = 0;
	Eigen::VectorXf user;

	Eigen::MatrixX2f land_2d;
	Eigen::MatrixX2f center;
	Eigen::MatrixXf exp;
	Eigen::MatrixX3f rot;
	Eigen::MatrixX3f tslt;
	Eigen::MatrixXi land_cor;
	Eigen::MatrixX3f s;
	Eigen::MatrixX2f dis;
	float fcs;
};

void fit_solve(
	std::vector<cv::Point2d> &landmarks, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl, DataPoint &data);

void init_exp_ide_r_t_pq(iden *ide, int ide_num);



void cal_f(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl);
void solve(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl);


float pre_cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor, Eigen::VectorXi &jaw_land_corr,
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, int id_idx,
	Eigen::VectorXf &ide_sg_vl);
void init_exp_ide(iden *ide, int train_id_num ,int id_idx);

void cal_rt_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);
void test_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);

void cal_rt_normalization(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);
void test_normalization(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);

void cal_inner_bldshps(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &bs_in,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx);

float cal_3d_vtx(
	iden *ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, int vtx_idx, int axis);


void update_slt(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::VectorXi &out_land_cor,
	Eigen::VectorXi &jaw_land_corr);

void test_slt(float f ,iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx);

float cal_3dpaper_exp(
	float f, iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor);

void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result);

float cal_3dpaper_ide(
	float f, iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::VectorXf &ide_sg_vl);

void cal_id_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result);

float cal_fixed_exp_same_ide(
	float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx,
	Eigen::VectorXf &ide_sg_vl);




//ceres
float ceres_exp_one(float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::RowVectorXf &exp);

float ceres_user_one(
	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &id_point, Eigen::VectorXf &user,
	Eigen::VectorXf &ide_sg_vl);

float ceres_user_fixed_exp(
	float focus, iden *ide, int id_idx, Eigen::MatrixXf &id_point_fix_exp, Eigen::VectorXf &user,
	Eigen::VectorXf &ide_sg_vl);



//bfgs
//float bfgs_exp_one(float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::VectorXf &exp);



void test_coef_land(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx);
void test_coef_mesh(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx);
void test_2dland(float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx);
void cal_mesh_land(Eigen::MatrixXf &bldshps);
void cal_mesh_land_exp_only(Eigen::MatrixXf &bldshps);
