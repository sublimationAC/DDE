
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

const float G_rand_tslt_border = 20;
const float G_rand_s_border = 10;
const float G_rand_f_border = 100;
const float G_rand_exp_border = 0.6;

const int G_rnd_rot = 5;
const int G_rnd_tslt = 5;
const int G_rnd_exp = 15;
const int G_rnd_user = 5;
const int G_rnd_camr = 5;
const int G_trn_factor = 35;


//const int G_target_type_size= G_nShape + 3 + 3 * 3 + 2 * G_land_num;
const int G_target_type_size = G_nShape-1 + 2 + 3 + 2 * G_land_num;
const int G_ta_size = 2 + 3;
const int G_disexp_size = 2 * G_land_num + G_nShape - 1;
const int G_tslt_num = 2;
const int G_angle_num = 3;

const float G_norm_face_rect_ave = 120;
const float G_norm_face_rect_sig = 60;

#define normalization
#define EPSILON 1e-3

#define norm_point_def


#ifndef FACE_X_UTILS_TRAIN_H_
#define FACE_X_UTILS_TRAIN_H_

#include <vector>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include<math.h>

const float pi = acos(-1);
const float G_rand_angle_border = 15.0* pi / 180 ;


struct TrainingParameters
{
	/* General parameters */
	std::string training_data_root = "";
	int landmark_count = -1;
	int left_eye_index_x = -1;
	int left_eye_index_y = -1;
	int right_eye_index_x = -1;
	int right_eye_index_y = -1;
	std::string output_model_pathname = "";

	/* Model parameters */
	int T = -1;
	int K = -1;
	int P = -1;
	double Kappa = -1;
	int F = -1;
	int Beta = -1;
	int TestInitShapeCount = -1;
	int ArgumentDataFactor = -1;
	int Base = -1;
	int Q = -1;
};

struct Target_type {
	Eigen::VectorXf exp;
	Eigen::RowVector3f tslt;
	//Eigen::Matrix3f rot;
	Eigen::MatrixX2f dis;
	Eigen::RowVector3f angle;
	
};

struct DataPoint
{
	cv::Mat image;
	cv::Rect face_rect;
	std::vector<cv::Point2d> landmarks;
	//std::vector<cv::Point2d> init_shape;
	Target_type shape, init_shape;
	Eigen:: VectorXf user;
	Eigen::RowVector2f center;
	Eigen::MatrixX2f land_2d;
	int ide_idx = 0;
#ifdef posit
		float f;
#endif // posit
#ifdef normalization
	Eigen::MatrixX3f s;
#endif
	
	Eigen::VectorXi land_cor;
};



struct Transform
{
	cv::Matx22d scale_rotation;
	cv::Matx21d translation;

	void Apply(std::vector<cv::Point2d> *x, bool need_translation = true);
};

template<typename T>
inline T Sqr(T a)
{
	return a * a;
}

Transform Procrustes(const std::vector<cv::Point2d> &x,
	const std::vector<cv::Point2d> &y);

std::vector<cv::Point2d> MeanShape(std::vector<std::vector<cv::Point2d>> shapes, 
	const TrainingParameters &tp);

std::vector<cv::Point2d> ShapeAdjustment(const std::vector<cv::Point2d> &shape,
	const std::vector<cv::Point2d> &offset);

double Covariance(double *x, double * y, const int size);

std::vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
	const std::vector<cv::Point2d> original_landmarks,
	cv::Rect new_face_rect);

std::vector<std::pair<int, double>> OMP(cv::Mat x, cv::Mat base, int coeff_count);

std::string TrimStr(const std::string &s, const std::string &space = "\t ");



#endif