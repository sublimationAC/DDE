//#pragma once

#include "math_headers.h"
#include <igl/read_triangle_mesh.h>
#include <cstring>
const int G_inner_land_num = 62;
const int G_line_num = 70;
const int G_nVerts = 11510;
const int G_land_num = 74;

struct Mesh_my
{
	int num_vtx = 0, num_rect = 0;
	EigenMatrixXs vtx;
	EigenMatrixXs norm_vtx;
	EigenMatrixXi rect;

};

void init_mesh(std::string name, Mesh_my &mesh);
void cal_norm(Mesh_my &mesh);
void draw_mesh(Mesh_my &mesh);
void draw_line(Mesh_my &mesh,double agl);
void check_2d_3d_corr(Mesh_my &mesh, Eigen::VectorXi &cor);
bool check_slt_line(Mesh_my &mesh, int i);
void get_silhouette_vertex(Mesh_my &mesh);
void check_2d_3d_out_corr(Mesh_my &mesh);
void cal_nor_vec(Eigen::RowVector3d &nor, Eigen::RowVector3d a, Eigen::RowVector3d b, Eigen::RowVector3d o);
void test_slt();
void get_coef_land(Eigen::MatrixX3f &coef_land);
void test_coef_land(Eigen::MatrixX3f &coef_land, int idx);
void get_coef_mesh(Eigen::MatrixX3f &coef_mesh);
void test_coef_mesh(Mesh_my &mesh, Eigen::MatrixX3f &coef_mesh, int idx);
