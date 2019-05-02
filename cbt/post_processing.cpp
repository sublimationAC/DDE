#include"post_processing.hpp"

#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

const float omg_reg_tslt = 5;
const float omg_tm_tslt = omg_reg_tslt/5;
const float omg_reg_angle = 5;
const float omg_tm_angle = omg_reg_angle / 5;
const float omg_reg_exp = 5;
const float omg_tm_exp = omg_reg_exp / 5;

const float pp_beta_l1_sum_exp = 6;
const float pp_beta_l2_exp = 3;

DataPoint G_data;
//struct ceres_cal_tslt {
//	ceres_cal_tslt(
//		const float f_, const Eigen::RowVector3f last_2_tslt_, const Eigen::RowVector3f last_1_tslt_,
//		const Eigen::RowVector3f now_tslt_, const Eigen::MatrixXf exp_r_t_point_matrix_) :
//		f(f_), last_2_tslt(last_2_tslt_), last_1_tslt(last_1_tslt_), now_tslt(now_tslt_), exp_r_t_point_matrix(exp_r_t_point_matrix_) {}
//
//	template <typename T> bool operator()(const T* const x_tslt, T *residual) const {
//		//puts("TT");
//		//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
//		//Eigen::Vector3d tslt;
//		T tslt[3];
//		tslt[2] = (T)0;
//		for (int j = 0; j < G_tslt_num; j++)
//			tslt[j] = x_tslt[j];
//		T angle[3];
//		for (int i = 0; i < 3; i++) angle[i] = (T)(G_data.shape.angle(i));
//		T R[3][3];
//		T Sa = sin(angle[0]), Ca = cos(angle[0]), Sb = sin(angle[1]),
//			Cb = cos(angle[1]), Sc = sin(angle[2]), Cc = cos(angle[2]);
//
//		R[0][0] = Ca * Cb;
//		R[0][1] = -Sa * Cb;
//		R[0][2] = Sb;
//		R[1][0] = Sa * Cc + Ca * Sb*Sc;
//		R[1][1] = Ca * Cc - Sa * Sb*Sc;
//		R[1][2] = -Cb * Sc;
//		R[2][0] = Sa * Sc - Ca * Sb*Cc;
//		R[2][1] = Ca * Sc + Sa * Sb*Cc;
//		R[2][2] = Cb * Cc;
//		/*std::cout << rot << '\n';
//		std::cout << tslt << '\n';*/
//		T ans = (T)0;
//		for (int i_v = 0; i_v < G_land_num; i_v++) {
//			T V[3], P[3];
//			//for (int j = 0; j < 3; j++) V[j] = (T)(0);
//
//			//printf("%d\n", i_v);
//			//V.setZero();
//
//			//puts("QAQ");
//			for (int j = 0; j < 3; j++) V[j] = (T)(exp_r_t_point_matrix(0, i_v * 3 + j));
//			for (int axis = 0; axis < 3; axis++)
//				for (int i_exp = 1; i_exp < G_nShape; i_exp++)
//					V[axis] = V[axis] + ((double)(exp_r_t_point_matrix(i_exp, i_v * 3 + axis)))*G_data.shape.exp(i_exp);
//
//			for (int axis = 0; axis < 3; axis++) {
//				P[axis] = (T)0;
//				for (int j = 0; j < 3; j++)
//					P[axis] += R[axis][j] * V[j];
//			}
//#ifdef normalization
//			for (int axis = 0; axis < 2; axis++) {
//				V[axis] = (T)0;
//				for (int j = 0; j < 3; j++)
//					V[axis] += ((double)G_data.s(axis, j)) * P[j];
//			}
//#endif // normalization
//#ifdef perspective
//			for (int axis = 0; axis < 3; axis++) V[axis] = P[axis];
//#endif // perspective
//
//
//			//std::cout << V.transpose() << '\n';
//			//puts("TAT");
//			for (int j = 0; j < G_tslt_num; j++)
//				V[j] = V[j] + tslt[j];
//
//#ifdef perspective
//			residual[i_v * 2] = (double)G_data.land_2d(i_v, 0) - V[0] * (double)f / V[2]- (double)G_data.center(0);
//			residual[i_v * 2 + 1] = (double)G_data.land_2d(i_v, 1) - V[1] * (double)f / V[2] - (double)G_data.center(0);
//#endif
//#ifdef normalization
//			residual[i_v * 2] = (double)G_data.land_2d(i_v, 0) - V[0];
//			residual[i_v * 2 + 1] = (double)G_data.land_2d(i_v, 1) - V[1];
//#endif
//			ans += residual[i_v * 2] * residual[i_v * 2];
//			ans += residual[i_v * 2 + 1] * residual[i_v * 2 + 1];
//			//printf("+++++++++%.6f \n", f / V(2));
//			/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
//			ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
//			V[0] * (double)f / V[2], V[1] * (double)f / V[2]);*/
//		}
//		printf("tslt %.5f %.5f ans %.5f\n", x_tslt[0], x_tslt[1], ans);
//		T ans_reg = (T)0, ans_tm = (T)0;
//		for (int i = 0; i < G_tslt_num; i++) {
//			residual[G_land_num * 2 + i] = ((T)omg_reg_tslt)*(x_tslt[i] - (T)now_tslt(i));
//			ans_reg += residual[G_land_num * 2 + i] * residual[G_land_num * 2 + i];
//		}
//		for (int i = 0; i < G_tslt_num; i++) {
//			residual[G_land_num * 2 + G_tslt_num + i] = ((T)omg_tm_tslt) * (x_tslt[i] + (T)last_2_tslt(i) - (T)(2 * last_1_tslt(i)));
//			ans_tm += residual[G_land_num * 2 + G_tslt_num + i] * residual[G_land_num * 2 + G_tslt_num + i];
//		}
//		printf("tslt reg %.5f tem %.5f\n", ans_reg,ans_tm);
//
//		return true;
//	}
//
//
//private:
//	const float f;
//	const Eigen::RowVector3f last_2_tslt, last_1_tslt, now_tslt;
//	const Eigen::MatrixXf exp_r_t_point_matrix;
//	//const Eigen::VectorXf ide_sg_vl;
//};
////
//struct ceres_cal_angle {
//	ceres_cal_angle(
//		const float f_, const Eigen::Vector3f last_2_angle_, const Eigen::Vector3f last_1_angle_,
//		const Eigen::Vector3f now_angle_, const Eigen::MatrixXf exp_r_t_point_matrix_) :
//		f(f_), last_2_angle(last_2_angle_), last_1_angle(last_1_angle_), now_angle(now_angle_), exp_r_t_point_matrix(exp_r_t_point_matrix_) {}
//
//	template <typename T> bool operator()(const T* const x_angle, T *residual) const {
//		//puts("TT");
//		//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
//		//Eigen::Vector3d tslt;
//		T tslt[3];
//		tslt[2] = (T)0;
//		for (int j = 0; j < G_tslt_num; j++)
//			tslt[j] = (T)(G_data.shape.tslt(j));
//		T angle[3];
//		for (int i = 0; i < 3; i++) angle[i] = (x_angle[i]);
//		T R[3][3];
//		T Sa = sin(angle[0]), Ca = cos(angle[0]), Sb = sin(angle[1]),
//			Cb = cos(angle[1]), Sc = sin(angle[2]), Cc = cos(angle[2]);
//
//		R[0][0] = Ca * Cb;
//		R[0][1] = -Sa * Cb;
//		R[0][2] = Sb;
//		R[1][0] = Sa * Cc + Ca * Sb*Sc;
//		R[1][1] = Ca * Cc - Sa * Sb*Sc;
//		R[1][2] = -Cb * Sc;
//		R[2][0] = Sa * Sc - Ca * Sb*Cc;
//		R[2][1] = Ca * Sc + Sa * Sb*Cc;
//		R[2][2] = Cb * Cc;
//		/*std::cout << rot << '\n';
//		std::cout << tslt << '\n';*/
//		T ans = (T)0;
//		for (int i_v = 0; i_v < G_land_num; i_v++) {
//			T V[3], P[3];
//			//for (int j = 0; j < 3; j++) V[j] = (T)(0);
//
//			//printf("%d\n", i_v);
//			//V.setZero();
//
//			//puts("QAQ");
//			for (int j = 0; j < 3; j++) V[j] = (T)(exp_r_t_point_matrix(0, i_v * 3 + j));
//			for (int axis = 0; axis < 3; axis++)
//				for (int i_exp = 1; i_exp < G_nShape; i_exp++)
//					V[axis] = V[axis] + ((double)(exp_r_t_point_matrix(i_exp, i_v * 3 + axis)))*G_data.shape.exp(i_exp);
//
//			for (int axis = 0; axis < 3; axis++) {
//				P[axis] = (T)0;
//				for (int j = 0; j < 3; j++)
//					P[axis] += R[axis][j] * V[j];
//			}
//#ifdef normalization
//			for (int axis = 0; axis < 2; axis++) {
//				V[axis] = (T)0;
//				for (int j = 0; j < 3; j++)
//					V[axis] += ((double)G_data.s(axis, j)) * P[j];
//			}
//#endif
//#ifdef perspective
//			for (int axis = 0; axis < 3; axis++) V[axis] = P[axis];
//#endif // perspective
//			//std::cout << V.transpose() << '\n';
//			//puts("TAT");
//			for (int j = 0; j < G_tslt_num; j++)
//				V[j] = V[j] + tslt[j];
//
//#ifdef perspective
//			residual[i_v * 2] = (double)G_data.land_2d(i_v, 0) - V[0] * (double)f / V[2] - (double)G_data.center(0);
//			residual[i_v * 2 + 1] = (double)G_data.land_2d(i_v, 1) - V[1] * (double)f / V[2] - (double)G_data.center(1);
//#endif
//#ifdef normalization
//			residual[i_v * 2] = (double)G_data.land_2d(i_v, 0) - V[0];
//			residual[i_v * 2 + 1] = (double)G_data.land_2d(i_v, 1) - V[1];
//#endif
//			ans += residual[i_v * 2] * residual[i_v * 2];
//			ans += residual[i_v * 2 + 1] * residual[i_v * 2 + 1];
//			//printf("+++++++++%.6f \n", f / V(2));
//			/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
//			ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
//			V[0] * (double)f / V[2], V[1] * (double)f / V[2]);*/
//		}
//		printf("angle %.5f %.5f ans %.5f\n", x_angle[0], x_angle[1], ans);
//		T ans_reg = (T)0, ans_tm = (T)0;
//		for (int i = 0; i < G_angle_num; i++) {
//			residual[G_land_num * 2 + i] = ((T)omg_reg_angle)*(x_angle[i] - (T)now_angle(i));
//			ans_reg += residual[G_land_num * 2 + i] * residual[G_land_num * 2 + i];
//		}
//		for (int i = 0; i < G_angle_num; i++) {
//			residual[G_land_num * 2 + G_angle_num + i] = ((T)omg_tm_angle) * (x_angle[i] + (T)last_2_angle(i) - (T)(2 * last_1_angle(i)));
//			ans_tm += residual[G_land_num * 2 + G_angle_num + i] * residual[G_land_num * 2 + G_angle_num + i];
//		}
//		printf("angle reg %.5f tem %.5f\n", ans_reg, ans_tm);
//
//		return true;
//	}
//
//
//private:
//	const float f;
//	const Eigen::Vector3f last_2_angle, last_1_angle, now_angle;
//	const Eigen::MatrixXf exp_r_t_point_matrix;
//	//const Eigen::VectorXf ide_sg_vl;
//};

struct ceres_postp_exp {
	ceres_postp_exp(
		const float f_, const Eigen::VectorXf last_2_exp_, const Eigen::VectorXf last_1_exp_,
		const Eigen::VectorXf now_exp_, const Eigen::MatrixXf exp_r_t_point_matrix_,
		const Eigen::MatrixX2f land_2d_) :
		f(f_), last_2_exp(last_2_exp_), last_1_exp(last_1_exp_), 
		now_exp(now_exp_), exp_r_t_point_matrix(exp_r_t_point_matrix_), land_2d(land_2d_){}

	template <typename T> bool operator()(const T* const x_exp, T *residual) const {
		assert(last_2_exp.size() == 47);
		assert(last_1_exp.size() == 47);
		assert(now_exp.size() == 47);

		puts("TT");
		//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
		//Eigen::Vector3d tslt;
		T tslt[3];
		tslt[2] = (T)0;
		for (int j = 0; j < G_tslt_num; j++)
			tslt[j] = (T)(G_data.shape.tslt(j));
		T angle[3];
		for (int i = 0; i < 3; i++) angle[i] = (T)(G_data.shape.angle(i));
		T R[3][3];
		T Sa = sin(angle[0]), Ca = cos(angle[0]), Sb = sin(angle[1]),
			Cb = cos(angle[1]), Sc = sin(angle[2]), Cc = cos(angle[2]);
		puts("A");
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
		T ans = (T)0;
		puts("A");
		for (int i_v = 0; i_v < G_land_num; i_v++) {
			T V[3], P[3];
			//for (int j = 0; j < 3; j++) V[j] = (T)(0);

			printf("%d\n", i_v);
			//V.setZero();

			//puts("QAQ");
			for (int j = 0; j < 3; j++) V[j] = (T)(exp_r_t_point_matrix(0, i_v * 3 + j));
			for (int axis = 0; axis < 3; axis++)
				for (int i_exp = 1; i_exp < G_nShape; i_exp++)
					V[axis] = V[axis] + ((double)(exp_r_t_point_matrix(i_exp, i_v * 3 + axis)))*x_exp[i_exp-1];

			for (int axis = 0; axis < 3; axis++) {
				P[axis] = (T)0;
				for (int j = 0; j < 3; j++)
					P[axis] += R[axis][j] * V[j];
			}
#ifdef normalization
			for (int axis = 0; axis < 2; axis++) {
				V[axis] = (T)0;
				for (int j = 0; j < 3; j++)
					V[axis] += ((double)G_data.s(axis, j)) * P[j];
			}
#endif
#ifdef perspective
			for (int axis = 0; axis < 3; axis++) V[axis] = P[axis];
#endif // perspective
			//std::cout << V.transpose() << '\n';
			//puts("TAT");
			for (int j = 0; j < G_tslt_num; j++)
				V[j] = V[j] + tslt[j];

#ifdef perspective
			residual[i_v * 2] = (double)G_data.land_2d(i_v, 0) - V[0] * (double)f / V[2] - (double)G_data.center(0);
			residual[i_v * 2 + 1] = (double)G_data.land_2d(i_v, 1) - V[1] * (double)f / V[2] - (double)G_data.center(1);
#endif
#ifdef normalization
			printf("%d %.5f %.5f\n", i_v, G_data.land_2d(i_v, 0), G_data.land_2d(i_v, 1));
			residual[i_v * 2] = (T)0;//(T)G_data.land_2d(i_v, 0) - V[0];
			residual[i_v * 2 + 1] = (T)0;//(T)G_data.land_2d(i_v, 1) - V[1];
#endif
			puts("B");
			ans += residual[i_v * 2] * residual[i_v * 2];
			ans += residual[i_v * 2 + 1] * residual[i_v * 2 + 1];
			//printf("+++++++++%.6f \n", f / V(2));
			/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
			ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
			V[0] * (double)f / V[2], V[1] * (double)f / V[2]);*/
		}
		printf("exp %.5f %.5f ans %.5f\n", x_exp[0], x_exp[1], ans);
		puts("C");
		T ans_reg = (T)0, ans_tm = (T)0;
		printf("%d %d %d %d %d %d\n", last_1_exp.rows(), last_1_exp.cols(), last_2_exp.rows(), last_2_exp.cols()
			, now_exp.rows(), now_exp.cols());
		for (int i = 0; i < G_nShape-1; i++) {
			residual[G_land_num * 2 + i] = ((T)omg_reg_exp)*(x_exp[i] - (T)now_exp(i+1));
			ans_reg += residual[G_land_num * 2 + i] * residual[G_land_num * 2 + i];
		}
		puts("D");
		for (int i = 0; i < G_nShape - 1; i++) {
			printf("%d %.5f\n", i, last_2_exp(i + 1));
			residual[G_land_num * 2 + G_nShape - 1 + i] = ((T)omg_tm_exp) * ((x_exp[i] + (T)last_2_exp(i + 1)) -(T)(2 * last_1_exp(i + 1)));
			ans_tm += residual[G_land_num * 2 + G_nShape - 1 + i] * residual[G_land_num * 2 + G_nShape - 1 + i];
		}
		printf("exp reg %.5f tem %.5f\n", ans_reg, ans_tm);
		residual[G_land_num * 2+ 2*G_nShape - 2] = T(0);
		for (int i = 0; i < G_nShape - 1; i++) {
			residual[G_land_num * 2 + 2 * G_nShape - 2] += x_exp[i] * (T)pp_beta_l1_sum_exp;
			residual[G_land_num * 2 + 2 * G_nShape - 2 + 1 + i] = x_exp[i] * (T)pp_beta_l2_exp;
		}

		return true;
	}


private:
	const float f;
	const Eigen::VectorXf last_2_exp, last_1_exp, now_exp;
	const Eigen::MatrixXf exp_r_t_point_matrix;
	const Eigen::MatrixX2f land_2d;
	//const Eigen::VectorXf ide_sg_vl;
};
void postp_rt_pnp(
	DataPoint &data, Eigen::MatrixXf &exp_r_t_point_matrix) {

	puts("solve pnp...");

	std::vector<cv::Point2f> land_2d; land_2d.clear();
	std::vector<cv::Point3f> land_3d; land_3d.clear();

	int temp_num = 0;
	/*land_in.resize(G_land_num, 2);
		land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2);*/
	for (int i_v = 0; i_v < G_land_num; i_v++)
		land_2d.push_back(cv::Point2f(data.land_2d(i_v, 0), data.land_2d(i_v, 1)));
	//std::cout << land_in.transpose() << '\n';
	//bs_in.resize(G_land_num, 3);
	land_3d.resize(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		v.setZero();
		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
			for (int axis = 0; axis < 3; axis++)
				v(axis) += data.shape.exp(i_shape)*exp_r_t_point_matrix(i_shape, i_v * 3 + axis);
		land_3d[i_v] = cv::Point3f(v(0), v(1), v(2));
		
	}



	// Camera internals

	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
		data.fcs, 0, data.center(0), 0, data.fcs, data.center(1), 0, 0, 1);
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

	std::cout << "Camera Matrix \n" << camera_matrix << "\n";
	// Output rotation and translation
	cv::Mat rotation_vector; // Rotation in axis-angle form
	cv::Mat translation_vector;

	// Solve for pose
	puts("ini 3d:");
	for (int i = 0; i < 6; i++)
		std::cout << i << ' ' << land_3d[i] << "\n";
	puts("ini 2d:");
	for (int i = 0; i < 6; i++)
		std::cout << i << ' ' << land_2d[i] << "\n";
	cv::solvePnP(land_3d, land_2d, camera_matrix, dist_coeffs, rotation_vector, translation_vector,
		0, CV_EPNP);

	for (int axis = 0; axis < 3; axis++)
		data.shape.tslt(axis) = translation_vector.at<double>(axis);

	std::cout << "rotation_vector:\n" << rotation_vector << "\n\n";

	cv::Mat rot;
	cv::Rodrigues(rotation_vector, rot);
	//Eigen::Map<Eigen::Matrix3d> R(rot.ptr <double>(), rot.rows, rot.cols);
	Eigen::Matrix3f R;


	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			R(i,j)= rot.at<double>(i, j);


	data.shape.angle = get_uler_angle_zyx(R);


	puts("-----------------------------------------");
	std::cout << rot << "\n";
	/*std::cout << R << "\n";
	system("pause");*/
	std::cout << R << '\n';

}

double ceres_post_processing(
	DataPoint &data,Target_type &last_2,Target_type &last_1,Target_type &now,Eigen::MatrixXf &exp_r_t_point_matrix) {
	G_data = data;
	print_datapoint(G_data);
	//system("pause");
	puts("post-processing  ----  optimizing exp&r&t&dis coeffients with fixed identity&Q by ceres");
	//print_target(last_2);
	//print_target(last_1);
	//system("pause");
	//google::InitGoogleLogging(argv[0]);
	double x_exp[G_nShape-1];

	for (int i = 0; i < G_nShape - 1; i++) x_exp[i] = now.exp(i + 1);

	

	//puts("A");
	Problem problem_exp;

	problem_exp.AddResidualBlock(
		new AutoDiffCostFunction<ceres_postp_exp, G_land_num * 2 + 3 * G_nShape-3 +1 ,G_nShape-1>(
			new ceres_postp_exp(data.fcs, last_2.exp, last_1.exp, now.exp, exp_r_t_point_matrix, data.land_2d)),
		NULL, x_exp);
	
	
	for (int i = 0; i < G_nShape-1; i++) {
		problem_exp.SetParameterLowerBound(x_exp, i, 0.0);
		problem_exp.SetParameterUpperBound(x_exp, i, 1.0);
	}
	Solver::Options options;
	options.max_num_iterations = 25;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	Solver::Summary summary;
	//Solve(options, &problem_exp, &summary);
	puts("ADDDDout");
	//puts("D");
	for (int ite = 0; ite < 3; ite++) {
		printf("t&a optmz :\ntslt  +++");
		std::cout << G_data.shape.tslt.transpose() << "\n";
		printf("angle : ");
		std::cout << G_data.shape.angle.transpose() << "\n";
		postp_rt_pnp(G_data, exp_r_t_point_matrix);
		printf("after t&a optmz :\ntslt  +++");
		std::cout << G_data.shape.tslt.transpose() << "\n";
		printf("angle : ");
		std::cout << G_data.shape.angle.transpose() << "\n";

		puts("optmz exp:");
		Solve(options, &problem_exp, &summary);		
		for (int i = 0; i < G_nShape - 1; i++) G_data.shape.exp(i + 1) = x_exp[i];
		system("pause");
		

		
		
	//	Solve(options, &problem_dis, &summary);
	}

	/*std::cout << summary.BriefReport() << "\n";
	std::cout << "Initial : " << user.transpose() << "\n";*/
	/*for (int i = 0; i < G_target_type_size; i++) now_v(i) = x_exp_t_a_d[i];
	vector2target(now_v, now);*/
	now = G_data.shape;
	
	//for (int i = 0; i < 2 * G_land_num; i++) now.dis(i / 2, i & 1)= x_dis[i];
	puts("calculating displacement...");
	Eigen::MatrixX2f land(G_land_num, 2);
	Eigen::Matrix3f rot = get_r_from_angle_zyx(now.angle);

#ifdef perspective
	Eigen::Vector3f T = data.shape.tslt;
#endif // perspective

#ifdef normalization
	Eigen::RowVector2f T = data.shape.tslt.block(0, 0, 1, 2);
#endif // normalization

	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f v;
		
		for (int j = 0; j < 3; j++) v(j) = exp_r_t_point_matrix(0, i_v * 3 + j);
		for (int axis = 0; axis < 3; axis++)
			for (int i_exp = 1; i_exp < G_nShape; i_exp++)
				v(axis) = v(axis) + ((exp_r_t_point_matrix(i_exp, i_v * 3 + axis)))*now.exp(i_exp);
#ifdef perspective
		v = rot * v + T;
		land(i_v,0) = data.fcs * v(0)/v(2) + data.center(0);
		land(i_v, 1) = data.fcs * v(1) / v(2) + data.center(1);
#endif // perspective
#ifdef normalization
		land.row(i_v) = ((data.s) * (rot * v)).transpose() + T;
#endif // normalization
	}

	now.dis.array() = data.land_2d.array() - land.array();

	now.land_cor = data.land_cor;
	data.shape = now;
	//std::cout << "Final   : " << user.transpose() << "\n";

	
	//system("pause");
	return (float)(summary.final_cost);

	//return 0;
}

void lk_post_processing(
	cv::Mat frame_last,cv::Mat frame_now,  DataPoint &data_last, DataPoint &data_now) {

	
	//Eigen::MatrixX2f landmarks_now, landmarks_last;
	//landmarks_now.resize(G_land_num, 2);
	//landmarks_last.resize(G_land_num, 2);
	//for (int i_v = 0; i_v < G_land_num; i_v++) {
	//	landmarks_now(i_v, 0) = data.land_2d(i_v, 0);landmarks_now(i_v, 1) = frame_now.rows - data.land_2d(i_v, 1);
	//	landmarks_last(i_v, 0) = land_2d_last(i_v, 0); landmarks_last(i_v, 1) = frame_last.rows - land_2d_last(i_v, 1);
	//}
	//
	//for (int i_v = 0; i_v < G_land_num; i_v++) {
	//	cv::Point2d p(landmarks_last(i_v, 0), landmarks_last(i_v, 1));
	//	cv::Point2d p_next=lk_get_pos_next(G_lk_batch_size, p, frame_last, frame_now);
	//	cv::Point2d p_inv= lk_get_pos_next(G_lk_batch_size, p_next, frame_now, frame_last);
	//	std::cout << p << "<-last  next->" << p_next << "  inv:" << p_inv << "\n";
	//	if (dis_cv_pt(p, p_inv) < 3) trust[i_v] = 1;
	//	else trust[i_v] = 0;
	//}

	std::vector<cv::Point2f> landmarks_last(G_land_num), landmarks_now(G_land_num), landmarks_inv(G_land_num), landmarks_now_g(G_land_num);
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		landmarks_last[i_v].x = data_last.land_2d(i_v, 0);
		landmarks_last[i_v].y = data_last.image.rows - data_last.land_2d(i_v, 1);
		landmarks_now_g[i_v].x = data_now.land_2d(i_v, 0);
		landmarks_now_g[i_v].y = data_now.image.rows - data_now.land_2d(i_v, 1);
	}

	std::vector<uchar> lk_sts(G_land_num);
	std::vector<float> lk_err(G_land_num);
	cv::calcOpticalFlowPyrLK(frame_last, frame_now, landmarks_last, landmarks_now, lk_sts, lk_err);
	cv::calcOpticalFlowPyrLK(frame_now, frame_last, landmarks_now, landmarks_inv, lk_sts, lk_err);

	for (int i_v = 0; i_v < G_land_num; i_v++)
		if (dis_cv_pt(landmarks_inv[i_v], landmarks_last[i_v]) < 3 
			&& dis_cv_pt(landmarks_now[i_v], landmarks_now_g[i_v]) < 3)
			data_now.land_2d(i_v, 0) = landmarks_now[i_v].x,
			data_now.land_2d(i_v, 1) = data_now.image.rows- landmarks_now[i_v].y;	
}
