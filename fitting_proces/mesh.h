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
void init_mesh(std::string name, Mesh_my &mesh);

void cal_norm(Mesh_my &mesh);
void itplt(Mesh_my *pre_mesh, Mesh_my *nex_mesh, Mesh_my &mesh, int num, int per_frame);

void get_coef_land(Eigen::MatrixX3f &coef_land, std::string name, float scale);
void get_coef_mesh(Eigen::MatrixX3f &coef_mesh, int &tot_obj_num, std::string name, float scale);
void test_coef_mesh(Mesh_my &mesh_std, Mesh_my &mesh, Eigen::MatrixX3f &land, Eigen::MatrixX3f &coef_mesh, Eigen::MatrixX3f &coef_land, int idx);

void draw_mesh(Mesh_my &mesh);

void draw_land(Mesh_my &mesh, Eigen::VectorXi &land_cor);
void draw_land(Eigen::MatrixX3f &land);

void draw_tst_slt_pts(Eigen::MatrixXi &slt_pts, int idx, Mesh_my mesh);