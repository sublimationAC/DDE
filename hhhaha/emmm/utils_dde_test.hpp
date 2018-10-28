#pragma once
#include <vector>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#define win64
//#define linux
#define normalization
//#define posit
#define EPSILON 1e-6


const int G_land_num = 74;
const int G_train_pic_id_num = 3300;
const int G_nShape = 47;
const int G_nVerts = 11510;
const int G_nFaces = 11540;
const int G_test_num = 77;
const int G_iden_num = 77;
const int G_inner_land_num = 59;
const int G_line_num = 50;
const int G_jaw_land_num = 20;

const float G_rand_rot_border = 0.1;
const float G_rand_tslt_border = 10;
const float G_rand_s_border = 10;
const float G_rand_f_border = 100;

const int G_rnd_rot = 5;
const int G_rnd_tslt = 5;
const int G_rnd_exp = 15;
const int G_rnd_user = 5;
const int G_rnd_camr = 5;
const int G_trn_factor = 35;

const int G_target_type_size = G_nShape + 3 + 3 * 3 + 2 * G_land_num;
const int G_dde_K = 100;

struct Target_type {
	Eigen::VectorXf exp;
	Eigen::Vector3f tslt;
	Eigen::Matrix3f rot;
	Eigen::MatrixX2f dis;

};

struct DataPoint
{
	cv::Mat image;
	//cv::Rect face_rect;
	std::vector<cv::Point2d> landmarks;
	//std::vector<cv::Point2d> init_shape;
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

float cal_3d_vtx(
	Eigen::MatrixXf &bldshps,
	Eigen::VectorXf &user, Eigen::VectorXf &exp, int vtx_idx, int axis);

void recal_dis(DataPoint &data, Eigen::MatrixXf &bldshps);

void cal_mesh(DataPoint &data, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &mesh);
void cal_3d_land(DataPoint &data, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &land_3d);

void test_load_mesh(std::vector <DataPoint> &data, Eigen::MatrixXf bldshps, int idx, std::string path);
void test_load_3d_land(std::vector <DataPoint> &data, Eigen::MatrixXf bldshps, int idx, std::string path);

void draw_point(cv::Mat& img, cv::Point2f fp, cv::Scalar color);

void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color);
void draw_voronoi(cv::Mat& img, cv::Subdiv2D& subdiv);

void test_del_tri(Eigen::MatrixX2f &points, cv::Mat& img, cv::Subdiv2D subdiv);

void cal_del_tri(
	Eigen::MatrixX2f points, cv::Mat& img, std::vector<cv::Vec6f> &triangleList);

void cal_del_tri(
	const std::vector<cv::Point2d> &points, cv::Rect &rect,
	std::vector<cv::Vec6f> &triangleList, Eigen::MatrixX3i &tri_idx);
//void test_del_tri(
//	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
//	iden *ide, std::vector<std::pair<int, std::pair <int, int> > > &del_tri);


float dis_cv_pt(CvPoint pointO, CvPoint pointA);
float cal_cv_area(CvPoint point0, CvPoint point1, CvPoint point2);

void update_2d_land(DataPoint &data, Eigen::MatrixXf &bldshps);

Target_type shape_difference(const Target_type &s1, const Target_type &s2);
Target_type shape_adjustment(Target_type &shape, Target_type &offset);