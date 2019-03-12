#pragma once
#include "utils_train.h"


float cal_3d_vtx(
	Eigen::MatrixXf &bldshps,
	Eigen::VectorXf &user, Eigen::VectorXf &exp, int vtx_idx, int axis);

float cal_3d_vtx_0ide(
	Eigen::MatrixXf &exp_matrix, Eigen::VectorXf &exp, int vtx_idx, int axis);

//void recal_dis(DataPoint &data, Eigen::MatrixXf &bldshps);
void recal_dis_ang_sqz_optmz(DataPoint &data, Eigen::MatrixXf &bldshps);
void recal_dis_ang_0ide(DataPoint &data, Eigen::MatrixXf &exp_matrix);

void cal_mesh(DataPoint &data, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &mesh);
void cal_3d_land(DataPoint &data, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &land_3d);

void test_load_mesh(std::vector <DataPoint> &data, Eigen::MatrixXf &bldshps, int idx, std::string &path);
void test_load_3d_land(std::vector <DataPoint> &data, Eigen::MatrixXf &bldshps, int idx, std::string &path);

void draw_point(cv::Mat& img, cv::Point2f fp, cv::Scalar color);

void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color);
void draw_voronoi(cv::Mat& img, cv::Subdiv2D& subdiv);

void test_del_tri(Eigen::MatrixX2f &points, cv::Mat& img, cv::Subdiv2D &subdiv);

void cal_del_tri(
	Eigen::MatrixX2f &points, cv::Mat& img, std::vector<cv::Vec6f> &triangleList);

void cal_del_tri(
	const std::vector<cv::Point2d> &points, cv::Rect &rect,
	std::vector<cv::Vec6f> &triangleList, Eigen::MatrixX3i &tri_idx);
//void test_del_tri(
//	std::vector < std::vector <cv::Mat_<uchar> > >& imgs,
//	iden *ide, std::vector<std::pair<int, std::pair <int, int> > > &del_tri);


double dis_cv_pt(cv::Point2d pointO, cv::Point2d pointA);
double cal_cv_area(cv::Point2d point0, cv::Point2d point1, cv::Point2d point2);

//void cal_init_2d_land_i(std::vector<cv::Point2d> &ans, const DataPoint &data, Eigen::MatrixXf &bldshps);
void cal_init_2d_land_ang_i_sqz_optmz(std::vector<cv::Point2d> &ans, const DataPoint &data, Eigen::MatrixXf &bldshps);
void cal_init_2d_land_ang_0ide_i(
	std::vector<cv::Point2d> &ans, const DataPoint &data, Eigen::MatrixXf &exp_matrix);

void get_init_land_ang_0ide_i(
	std::vector<cv::Point2d> &ans, const DataPoint &data,
	Eigen::MatrixXf &bldshps);

Target_type shape_difference(const Target_type &s1, const Target_type &s2);
Target_type shape_adjustment(Target_type &shape, Target_type &offset);
std::vector<double> shape_adjustment(std::vector<double> &shape, Target_type &offset, char which);


std::vector<cv::Point2d> mean_shape(std::vector<std::vector<cv::Point2d>> shapes,
	const TrainingParameters &tp);

void print_datapoint(DataPoint &data);
void print_target(Target_type &data);

void target2vector(Target_type &data, Eigen::VectorXf &ans);
void vector2target(Eigen::VectorXf &data, Target_type &ans);

//Eigen::RowVector3f get_uler_angle(Eigen::Matrix3f R);
//
////Eigen::Matrix3f get_r_from_angle(float angle, int axis);
//Eigen::Matrix3f get_r_from_angle(const Eigen::Vector3f &angle);

Eigen::Vector3f get_uler_angle_zyx(Eigen::Matrix3f R);
Eigen::Matrix3f get_r_from_angle_zyx(const Eigen::Vector3f &angle);

void cal_left_eye_rect(const std::vector<cv::Point2d> &ref_shape, cv::Rect &left_eye_rect);
void cal_right_eye_rect(const std::vector<cv::Point2d> &ref_shape, cv::Rect &right_eye_rect);
void cal_mouse_rect(const std::vector<cv::Point2d> &ref_shape, cv::Rect &mouse_rect);

void rect_scale(cv::Rect &rect, double scale);

void normalize_gauss_face_rect(cv::Mat image,cv::Rect &rect);

uchar get_batch_feature(cv::Mat img, cv::Point p);

uchar get_sobel_batch_feature(cv::Mat img, cv::Point p);

void load_slt(
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	std::string path_slt, std::string path_rect);
void update_slt_init_shape(
	Eigen::MatrixXf &bldshps, std::vector<int> *slt_line,
	std::vector<std::pair<int, int> > *slt_point_rect, DataPoint &data);

void get_index_slt_point_rect(
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	const DataPoint &data);

float cal_3d_vtx_optmz_last(
	Eigen::MatrixXf &bldshps, Eigen::VectorXf &user, Eigen::VectorXf &exp, int vtx_idx, int axis);