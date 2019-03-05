#include "utils.h"
const float EPSILON = 1e-6;

struct Mesh_my
{
	int num_vtx=0, num_rect =0;
	Eigen::MatrixXd vtx;
	Eigen::MatrixXd norm_vtx;
	Eigen::MatrixXi rect;

};

void init(std:: string &name, Mesh_my &mesh, double scale, std::string &lv_name, Eigen::VectorXi &land_cor);
void cal_norm(Mesh_my &mesh);
void itplt(Mesh_my *pre_mesh, Mesh_my *nex_mesh, Mesh_my &mesh, int num, int per_frame);
void draw_mesh(Mesh_my &mesh);

void draw_land(Mesh_my &mesh, Eigen::VectorXi &land_cor);