#pragma once
#include "2dland.h"

float admm_exp_one(
	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::RowVectorXf &exp);

float admm_user_one(
	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &user_point, Eigen::VectorXf &user, float lmd);