//#pragma once

#include "math_headers.h"
#include <igl/read_triangle_mesh.h>
#include <cstring>
const int G_inner_land_num = 59;
const int G_jaw_land_num = 20;
const int G_line_num = 50;
const int G_nVerts = 11510;
const int G_land_num = 74;//68

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
void check_2d_3d_inner_jaw_corr(Mesh_my &mesh, Eigen::VectorXi &cor);
bool check_slt_line(Mesh_my &mesh, int i);
void get_silhouette_vertex(Mesh_my &mesh);
void check_2d_3d_out_corr(Mesh_my &mesh);
void cal_nor_vec(Eigen::RowVector3d &nor, Eigen::RowVector3d a, Eigen::RowVector3d b, Eigen::RowVector3d o);
void test_slt();
void get_coef_land(Eigen::MatrixX3f &coef_land, std::string name);
void test_coef_land(Eigen::MatrixX3f &coef_land, int idx);
void get_coef_mesh(Eigen::MatrixX3f &coef_mesh,std::string name);
void test_coef_mesh(Mesh_my &mesh, Eigen::MatrixX3f &coef_mesh, int idx);
void smooth_mesh(Mesh_my &mesh, int iteration);
bool check_mouse_vtx(Mesh_my &mesh, int i);
void get_mouse_data(Mesh_my &mesh);
void test_mouse(Mesh_my &mesh);

void check_3d_fan_corr(Mesh_my &mesh, Eigen::VectorXi &cor);

void get_positive_point(Mesh_my &mesh);

void test_update_slt_norm(
	Mesh_my &mesh, Eigen::MatrixX3f &norm_line);
void draw_test_slt_norm(Eigen::MatrixX3f test_slt_norm);