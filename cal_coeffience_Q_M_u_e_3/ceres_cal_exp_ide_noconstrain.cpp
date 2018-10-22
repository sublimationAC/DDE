// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "calculate_coeff.h"
#define set_user_bound
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

float beta_sum_user = 200;

struct ceres_cal_exp {
	ceres_cal_exp(
		const float f_, const iden* ide_, const int id_idx_, const int exp_idx_,
		const Eigen::MatrixXf exp_point_) :
		f(f_), ide(ide_), id_idx(id_idx_), exp_idx(exp_idx_), exp_point(exp_point_) {}

	template <typename T> bool operator()(const T* const x_exp, T *residual) const {
		//puts("TT");
		//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
		//Eigen::Vector3d tslt;
		T tslt[3];
		for (int j = 0; j < 3; j++)
			tslt[j] = (T)(ide[id_idx].tslt(exp_idx, j));

		/*std::cout << rot << '\n';
		std::cout << tslt << '\n';*/
		//T ans = (T)0;
		for (int i_v = 0; i_v < G_land_num; i_v++) {
			T V[3];
			for (int j = 0; j < 3; j++) V[j] = (T)(0);
			//printf("%d\n", i_v);
			//V.setZero();

			//puts("QAQ");

			for (int axis = 0; axis < 3; axis++)
				for (int i_shape = 0; i_shape < G_nShape; i_shape++)
					V[axis] = V[axis] + ((double)(exp_point(i_shape, i_v * 3 + axis)))*x_exp[i_shape];
			//std::cout << V.transpose() << '\n';
			//puts("TAT");
			for (int j = 0; j<3; j++)
				V[j] = V[j] + tslt[j];

			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2]);
			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2]);
#ifdef posit
			residual[i_v * 2] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2];
			residual[i_v * 2 + 1] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2];
#endif
#ifdef normalization
			residual[i_v * 2] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) + ide[id_idx].center(exp_idx, 0) - V[0];
			residual[i_v * 2 + 1] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) + ide[id_idx].center(exp_idx, 1) - V[1];
#endif

			//printf("+++++++++%.6f \n", f / V(2));
			/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
			ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
			V[0] * (double)f / V[2], V[1] * (double)f / V[2]);*/
		}
		//residual[G_land_num * 2] = (T)0;
		//printf("- - %.6f\n", ans);
		//fprintf(fp, "%.6f\n", ans);
		return true;
	}

private:
	const float f;
	const iden *ide;
	const int id_idx;
	const int exp_idx;
	const Eigen::MatrixXf exp_point;
};

float ceres_exp_one(
	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::RowVectorXf &exp) {

	puts("optimizing expression coeffients only by ceres");
	//google::InitGoogleLogging(argv[0]);
	double x_exp[G_nShape];
	for (int i = 0; i < G_nShape; i++) x_exp[i] = exp(i);

	Problem problem;
	problem.AddResidualBlock(
		new AutoDiffCostFunction<ceres_cal_exp, G_land_num * 2, G_nShape>(
			new ceres_cal_exp(focus, ide, id_idx, exp_idx, exp_point)),
		NULL, x_exp);
	for (int i = 0; i < G_nShape; i++) {
		problem.SetParameterLowerBound(x_exp, i, 0.0);
		problem.SetParameterUpperBound(x_exp, i, 1.0);
	}

	Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	Solver::Summary summary;
	Solve(options, &problem, &summary);
	/*std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial : " << exp.transpose() << "\n";*/
	for (int i = 0; i < G_nShape; i++)  exp(i) = (float)x_exp[i];
	//std::cout << "Final   : " << exp.transpose() << "\n";

	//system("pause");
	return (float)(summary.final_cost);
	//return 0;
}


struct ceres_cal_user {
	ceres_cal_user(
		const float f_, const iden* ide_, const int id_idx_, const int exp_idx_,
		const Eigen::MatrixXf id_point_, const Eigen::VectorXf ide_sg_vl_) :
		f(f_), ide(ide_), id_idx(id_idx_), exp_idx(exp_idx_), id_point(id_point_), ide_sg_vl(ide_sg_vl_) {}

	template <typename T> bool operator()(const T* const x_user, T *residual) const {
		//puts("TT");
		//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
		//Eigen::Vector3d tslt;
		T tslt[3];
		for (int j = 0; j < 3; j++)
			tslt[j] = (T)(ide[id_idx].tslt(exp_idx, j));

		/*std::cout << rot << '\n';
		std::cout << tslt << '\n';*/
			
		for (int i_v = 0; i_v < G_land_num; i_v++) {
			T V[3];
			for (int j = 0; j < 3; j++) V[j] = (T)(0);
			//printf("%d\n", i_v);
			//V.setZero();

			//puts("QAQ");

			for (int axis = 0; axis < 3; axis++)
				for (int i_id = 0; i_id < G_iden_num; i_id++)
					V[axis] = V[axis] + ((double)(id_point(i_id, i_v * 3 + axis)))*x_user[i_id];
			//std::cout << V.transpose() << '\n';
			//puts("TAT");
			for (int j = 0; j<3; j++)
				V[j] = V[j] + tslt[j];
			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2]);
			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2]);
#ifdef posit
			residual[i_v * 2] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2];
			residual[i_v * 2 + 1] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2];
#endif
#ifdef normalization
			residual[i_v * 2] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) + ide[id_idx].center(exp_idx, 0) - V[0];
			residual[i_v * 2 + 1] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) + ide[id_idx].center(exp_idx, 1) - V[1];
#endif
			//printf("+++++++++%.6f \n", f / V(2));
			/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
			ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
			V[0] * (double)f / V[2], V[1] * (double)f / V[2]);*/
		}
		T ans = (T)0;
		for (int i_id = 0; i_id < G_iden_num; i_id++)
			ans += (T)x_user[i_id];
		printf("- - %.6f\n", ans);
		residual[G_land_num * 2] = (ans - (T)1)*(ans - (T)1)*(T)beta_sum_user;
		//fprintf(fp, "%.6f\n", ans);

		return true;
	}

private:
	const float f;
	const iden *ide;
	const int id_idx;
	const int exp_idx;
	const Eigen::MatrixXf id_point;
	const Eigen::VectorXf ide_sg_vl;
};

float ceres_user_one(
	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &id_point, Eigen::VectorXf &user,
	Eigen::VectorXf &ide_sg_vl) {

	puts("optimizing identity coeffients only by ceres");
	//google::InitGoogleLogging(argv[0]);
	double x_user[G_iden_num];
	for (int i = 0; i < G_iden_num; i++) x_user[i] = user(i);
	//puts("A");
	Problem problem;
	problem.AddResidualBlock(
		new AutoDiffCostFunction<ceres_cal_user, G_land_num * 2+1, G_iden_num>(
			new ceres_cal_user(focus, ide, id_idx, exp_idx, id_point, ide_sg_vl)),
		NULL, x_user);
	//puts("B");

	Solver::Options options;

#ifdef set_user_bound
	for (int i = 0; i < G_iden_num; i++) {
		problem.SetParameterLowerBound(x_user, i, 0.0);
		problem.SetParameterUpperBound(x_user, i, 1.0);
	}
	options.max_num_iterations = 25;
#endif // set_user_bound
	//puts("C");

	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	Solver::Summary summary;
	//puts("D");
	Solve(options, &problem, &summary);
	/*std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial : " << user.transpose() << "\n";*/
	for (int i = 0; i < G_iden_num; i++)  user(i) = (float)x_user[i];
	//std::cout << "Final   : " << user.transpose() << "\n";

	//system("pause");
	return (float)(summary.final_cost);
}


float ceres_user_fixed_exp(
	float focus, iden *ide, int id_idx, Eigen::MatrixXf &id_point_fix_exp, Eigen::VectorXf &user,
	Eigen::VectorXf &ide_sg_vl) {


	puts("optimizing identity coeffients with fixed expression by ceres");
	//google::InitGoogleLogging(argv[0]);
	double x_user[G_iden_num];
	for (int i = 0; i < G_iden_num; i++) x_user[i] = user(i);
	//puts("A");
	Problem problem;
	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		problem.AddResidualBlock(
			new AutoDiffCostFunction<ceres_cal_user, G_land_num * 2 + 1, G_iden_num>(
				new ceres_cal_user(focus, ide, id_idx, i_exp,
					id_point_fix_exp.block(i_exp*G_iden_num, 0, G_iden_num, G_land_num * 3), ide_sg_vl)),
			NULL, x_user);
	}

	//puts("C");
	Solver::Options options;

#ifdef set_user_bound
	for (int i = 0; i < G_iden_num; i++) {
		problem.SetParameterLowerBound(x_user, i, 0.0);
		problem.SetParameterUpperBound(x_user, i, 1.0);
	}
	options.max_num_iterations = 25;
#endif // set_user_bound


	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	Solver::Summary summary;
	//puts("D");
	Solve(options, &problem, &summary);
	/*std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial : " << user.transpose() << "\n";*/
	for (int i = 0; i < G_iden_num; i++)  user(i) = (float)x_user[i];
	//std::cout << "Final   : " << user.transpose() << "\n";

	//system("pause");
	return (float)(summary.final_cost);
}