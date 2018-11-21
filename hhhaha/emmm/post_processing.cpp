//#include"post_processing.hpp"
//
//#include "ceres/ceres.h"
//#include "glog/logging.h"
//using ceres::AutoDiffCostFunction;
//using ceres::CostFunction;
//using ceres::Problem;
//using ceres::Solver;
//using ceres::Solve;
//
//const float omg_reg = 5;
//const float omg_tm = 1;
//
//struct ceres_cal_exp_r_t {
//	ceres_cal_exp_r_t(
//		const float f_, const DataPoint data_, const Eigen::VectorXf last_2_v_, const Eigen::VectorXf last_1_v_,
//		const Eigen::VectorXf now_v_, const Eigen::MatrixXf exp_r_t_point_) :
//		f(f_), data(data_), last_2_v(last_2_v_), last_1_v(last_1_v_), now_v(now_v_), exp_r_t_point(exp_r_t_point_) {}
//
//	template <typename T> bool operator()(const T* const x_exp_r_t, T *residual) const {
//		//puts("TT");
//		//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
//		//Eigen::Vector3d tslt;
//		T tslt[3];
//		for (int j = 0; j < 3; j++)
//			tslt[j] = (T)(x_exp_r_t[G_nShape+j]);
//		T R[];
//
//		/*std::cout << rot << '\n';
//		std::cout << tslt << '\n';*/
//
//		for (int i_v = 0; i_v < G_land_num; i_v++) {
//			T V[3];
//			for (int j = 0; j < 3; j++) V[j] = (T)(0);
//			//printf("%d\n", i_v);
//			//V.setZero();
//
//			//puts("QAQ");
//
//			for (int axis = 0; axis < 3; axis++)
//				for (int i_id = 0; i_id < G_iden_num; i_id++)
//					V[axis] = V[axis] + ((double)(id_point(i_id, i_v * 3 + axis)))*x_user[i_id];
//			//std::cout << V.transpose() << '\n';
//			//puts("TAT");
//			for (int j = 0; j < 3; j++)
//				V[j] = V[j] + tslt[j];
//			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2]);
//			//ans += ((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2])*((double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2]);
//#ifdef posit
//			residual[i_v * 2] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V[0] * (double)f / V[2];
//			residual[i_v * 2 + 1] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V[1] * (double)f / V[2];
//#endif
//#ifdef normalization
//			residual[i_v * 2] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) + ide[id_idx].center(exp_idx, 0) - V[0];
//			residual[i_v * 2 + 1] = (double)ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) + ide[id_idx].center(exp_idx, 1) - V[1];
//#endif
//			//printf("+++++++++%.6f \n", f / V(2));
//			/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
//			ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
//			V[0] * (double)f / V[2], V[1] * (double)f / V[2]);*/
//		}
//
//		for (int i=0;i<G_target_type_size;i++)
//			residual[G_land_num * 2 + i] = (double)omg_reg*(x_exp_r_t[i]-now_v(i));
//		for (int i=0;i<G_target_type_size;i++)
//			residual[G_land_num*2+G_target_type_size+i]= (double)omg_tm*(x_exp_r_t[i] +last_2_v(i)-2*last_1_v(i));
//
//		
//		//fprintf(fp, "%.6f\n", ans);
//
//		return true;
//	}
//
//private:
//	const float f;
//	DataPoint data;
//	const Eigen::VectorXf last_2_v,last_1_v,now_v;
//	const Eigen::MatrixXf exp_r_t_point;
//	//const Eigen::VectorXf ide_sg_vl;
//};
//
//
//double ceres_post_processing(
//	DataPoint &data,Target_type &last_2,Target_type &last_1,Target_type &now,Eigen::MatrixXf &exp_r_t_point_matrix) {
//
//	puts("optimizing exp&r&t coeffients with fixed identity&Q by ceres");
//	//google::InitGoogleLogging(argv[0]);
//	double x_exp_r_t[G_target_type_size];
//	Eigen::VectorXf last_2_v, last_1_v, now_v;
//	target2vector(last_2, last_2_v); target2vector(last_1, last_1_v); target2vector(now, now_v);
//	for (int i = 0; i < G_target_type_size; i++) x_exp_r_t[i] = now_v(i);
//
//	//puts("A");
//	Problem problem;
//	problem.AddResidualBlock(
//			new AutoDiffCostFunction<ceres_cal_exp_r_t, G_land_num * 2 + 2* G_target_type_size, G_target_type_size>(
//				new ceres_cal_exp_r_t(0,data,last_2_v,last_1_v,now_v,exp_r_t_point_matrix)),
//			NULL, x_exp_r_t);
//	
//
//	//puts("C");
//	Solver::Options options;
//
//	for (int i = 0; i < G_nShape; i++) {
//		problem.SetParameterLowerBound(x_exp_r_t, i, 0.0);
//		problem.SetParameterUpperBound(x_exp_r_t, i, 1.0);
//	}
//	options.max_num_iterations = 25;
//
//
//
//	options.linear_solver_type = ceres::DENSE_QR;
//	options.minimizer_progress_to_stdout = true;
//
//	Solver::Summary summary;
//	//puts("D");
//	Solve(options, &problem, &summary);
//	/*std::cout << summary.BriefReport() << "\n";
//	std::cout << "Initial : " << user.transpose() << "\n";*/
//	for (int i = 0; i < G_target_type_size; i++) now_v(i) = x_exp_r_t[i];
//	vector2target(now_v, now);
//	data.shape = now;
//	//std::cout << "Final   : " << user.transpose() << "\n";
//
//	//system("pause");
//	return (float)(summary.final_cost);
//
//	//return 0;
//}