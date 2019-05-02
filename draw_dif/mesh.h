//#pragma once

#include "math_headers.h"
#include <igl/read_triangle_mesh.h>
#include <cstring>
const int G_inner_land_num = 58;
const int G_jaw_land_num = 20;

const int G_nVerts = 11510;
const int G_land_num = 73;//68


struct Mesh_my
{
	int num_vtx = 0, num_rect = 0;
	EigenMatrixXs vtx;
	EigenMatrixXs norm_vtx;
	EigenMatrixXi rect;

};

void init_mesh(std::string name, Mesh_my &mesh);

void get_coef_land(Eigen::MatrixX3f &coef_land, std::string name);
void test_coef_land(Eigen::MatrixX3f &coef_land, int idx);
void get_coef_mesh(Eigen::MatrixX3f &coef_mesh, std::string name);
void test_coef_mesh(Mesh_my &mesh, Eigen::MatrixX3f &coef_mesh, int idx);

void cal_norm(Mesh_my &mesh);
void draw_mesh(Mesh_my &mesh);
void draw_mesh_point(Mesh_my &mesh);
void draw_line(Mesh_my &mesh,double agl);


void get_mima(Mesh_my &mesh, Mesh_my &mesh_ref, float &mi, float &ma, int axis);