#include <igl/read_triangle_mesh.h>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
const float EPSILON = 1e-6;


struct Mesh_my
{
	int num_vtx=0, num_rect =0;
	Eigen::MatrixXd vtx;
	Eigen::MatrixXd norm_vtx;
	Eigen::MatrixXi rect;

};

void init(std:: string &name, Mesh_my &mesh, double scale);
void cal_norm(Mesh_my &mesh);
void itplt(Mesh_my *pre_mesh, Mesh_my *nex_mesh, Mesh_my &mesh, int num, int per_frame);
void draw_mesh(Mesh_my &mesh);

