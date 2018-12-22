#include"post_processing.hpp"

#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

const float omg_reg = 5;
const float omg_tm = 1;

struct ceres_cal_exp_r_t {
	ceres_cal_exp_r_t(
		const float f_, const DataPoint data_, const Eigen::VectorXf last_2_v_, const Eigen::VectorXf last_1_v_,
		const Eigen::VectorXf now_v_, const Eigen::MatrixXf exp_r_t_point_matrix_) :
		f(f_), data(data_), last_2_v(last_2_v_), last_1_v(last_1_v_), now_v(now_v_), exp_r_t_point_matrix(exp_r_t_point_matrix_) {}

	template <typename T> bool operator()(const T* const x_exp_t_a_d, T *residual) const {
		//puts("TT");
		//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
		//Eigen::Vector3d tslt;
		T tslt[3];
		tslt[2] = (T)0;
		for (int j = 0; j < G_tslt_num; j++)
			tslt[j] = (x_exp_t_a_d[G_nShape - 1 + j]);
		T angle[3];
		for (int i = 0; i < 3; i++) angle[i] = (x_exp_t_a_d[G_nShape - 1 + G_tslt_num + i]);
		T R[3][3];
		T Sa = sin(angle[0]), Ca = cos(angle[0]), Sb = sin(angle[1]),
			Cb = cos(angle[1]), Sc = sin(angle[2]), Cc = cos(angle[2]);

		R[0][0] = Ca * Cb;
		R[0][1] = -Sa * Cb;
		R[0][2] = Sb;
		R[1][0] = Sa * Cc + Ca * Sb*Sc;
		R[1][1] = Ca * Cc - Sa * Sb*Sc;
		R[1][2] = -Cb * Sc;
		R[2][0] = Sa * Sc - Ca * Sb*Cc;
		R[2][1] = Ca * Sc + Sa * Sb*Cc;
		R[2][2] = Cb * Cc;
		/*std::cout << rot << '\n';
		std::cout << tslt << '\n';*/

		for (int i_v = 0; i_v < G_land_num; i_v++) {
			T V[3], P[3];
			//for (int j = 0; j < 3; j++) V[j] = (T)(0);

			//printf("%d\n", i_v);
			//V.setZero();

			//puts("QAQ");
			for (int j = 0; j < 3; j++) V[j] = (T)(exp_r_t_point_matrix(0, i_v * 3 + j));
			for (int axis = 0; axis < 3; axis++)
				for (int i_exp = 1; i_exp < G_nShape; i_exp++)
					V[axis] = V[axis] + ((double)(exp_r_t_point_matrix(i_exp, i_v * 3 + axis)))*x_exp_t_a_d[i_exp - 1];

			for (int axis = 0; axis < 3; axis++) {
				P[axis] = (T)0;
				for (int j = 0; j < 3; j++)
					P[axis] += R[axis][j] * V[j];
			}

			for (int axis = 0; axis < 2; axis++) {
				V[axis] = (T)0;
				for (int j = 0; j < 3; j++)
					V[axis] += ((double)data.s(axis, j)) * P[j];
			}

			//std::cout << V.transpose() << '\n';
			//puts("TAT");
			for (int j = 0; j < 3; j++)
				V[j] = V[j] + tslt[j];
			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2]);
			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2]);
#ifdef posit
			residual[i_v * 2] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2];
			residual[i_v * 2 + 1] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2];
#endif
#ifdef normalization
			residual[i_v * 2] = (double)data.land_2d(i_v, 0) - V[0];
			residual[i_v * 2 + 1] = (double)data.land_2d(i_v, 1) - V[1];
#endif
			//printf("+++++++++%.6f \n", f / V(2));
			/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
			ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
			V[0] * (double)f / V[2], V[1] * (double)f / V[2]);*/
		}

		for (int i = 0; i < G_target_type_size; i++)
			residual[G_land_num * 2 + i] =((T) omg_reg)*(x_exp_t_a_d[i] - (T)now_v(i));
		for (int i = 0; i < G_target_type_size; i++)
			residual[G_land_num * 2 + G_target_type_size + i] = ((T)omg_tm) * (x_exp_t_a_d[i] + (T)last_2_v(i) -(T)( 2 * last_1_v(i)));


		//fprintf(fp, "%.6f\n", ans);

		return true;
	}

private:
	const float f;
	DataPoint data;
	const Eigen::VectorXf last_2_v, last_1_v, now_v;
	const Eigen::MatrixXf exp_r_t_point_matrix;
	//const Eigen::VectorXf ide_sg_vl;
};


double ceres_post_processing(
	DataPoint &data,Target_type &last_2,Target_type &last_1,Target_type &now,Eigen::MatrixXf &exp_r_t_point_matrix) {

	puts("post-processing  ----  optimizing exp&r&t&dis coeffients with fixed identity&Q by ceres");
	//print_target(last_2);
	//print_target(last_1);
	//system("pause");
	//google::InitGoogleLogging(argv[0]);
	double x_exp_t_a_d[G_target_type_size];
	Eigen::VectorXf last_2_v, last_1_v, now_v;
	target2vector(last_2, last_2_v); target2vector(last_1, last_1_v); target2vector(now, now_v);
	for (int i = 0; i < G_target_type_size; i++) x_exp_t_a_d[i] = now_v(i);

	//puts("A");
	Problem problem;
	problem.AddResidualBlock(
			new AutoDiffCostFunction<ceres_cal_exp_r_t, G_land_num * 2 + 2* G_target_type_size, G_target_type_size>(
				new ceres_cal_exp_r_t(0,data,last_2_v,last_1_v,now_v,exp_r_t_point_matrix)),
			NULL, x_exp_t_a_d);
	

	//puts("C");
	Solver::Options options;

	for (int i = 0; i < G_nShape-1; i++) {
		problem.SetParameterLowerBound(x_exp_t_a_d, i, 0.0);
		problem.SetParameterUpperBound(x_exp_t_a_d, i, 1.0);
	}
	options.max_num_iterations = 25;



	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	Solver::Summary summary;
	//puts("D");
	Solve(options, &problem, &summary);
	/*std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial : " << user.transpose() << "\n";*/
	for (int i = 0; i < G_target_type_size; i++) now_v(i) = x_exp_t_a_d[i];
	vector2target(now_v, now);
	data.shape = now;
	//std::cout << "Final   : " << user.transpose() << "\n";

	//system("pause");
	return (float)(summary.final_cost);

	//return 0;
}