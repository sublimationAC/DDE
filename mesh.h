//#pragma once

#include "math_headers.h"
#include <igl/read_triangle_mesh.h>
#include <cstring>

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
void check_2d_3d_corr(Mesh_my &mesh);