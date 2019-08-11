#pragma once
#include <igl/read_triangle_mesh.h>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

//#define normalization
#define perspective
//#define test_svd_vec_def

const int G_iden_num = 77;
const int G_land_num = 73;
const int G_nShape = 47;
const int G_line_num = 84;

struct Target_type {
	Eigen::VectorXf exp;
	Eigen::RowVector3f tslt;
	Eigen::Matrix3f rot;
	Eigen::MatrixX2f dis;

};

struct DataPoint
{

	Target_type shape;
	Eigen::VectorXf user;
	Eigen::RowVector2f center;
	Eigen::MatrixX2f land_2d;
#ifdef perspective
	float f;
#endif // posit
#ifdef normalization
	Eigen::MatrixX3f s;
#endif

	Eigen::VectorXi land_cor;
};

void load_land_cor_from_lv(std::string name, Eigen::VectorXi &land_cor);
void load_land_cor_from_psp_f(std::string name, Eigen::VectorXi &land_cor);

void get_tst_slt_pts(Eigen::MatrixXi &slt_pts);