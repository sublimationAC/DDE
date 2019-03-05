#pragma once
#include <igl/read_triangle_mesh.h>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#define normalization


const int G_iden_num = 77;
const int G_land_num = 74;
const int G_nShape = 47;

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
#ifdef posit
	float f;
#endif // posit
#ifdef normalization
	Eigen::MatrixX3f s;
#endif

	Eigen::VectorXi land_cor;
};

void load_land_cor_from_lv(std::string name, Eigen::VectorXi &land_cor);