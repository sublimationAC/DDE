#include <iostream>
#include "meta.h"
#include "boundedproblem.h"
#include "solver/lbfgsbsolver.h"
#include "calculate_coeff.h"

// to use CppNumericalSolvers just use the namespace "cppoptlib"
//FILE *fp;
namespace cppoptlib {

	
	template<typename T>
	class cal_exp : public BoundedProblem<T> {
	public:
		using Superclass = BoundedProblem<T>;
		using typename Superclass::TVector;
		using TMatrix = typename Superclass::THessian;
		
		const float f;
		const iden *ide;
		const int id_idx;
		const int exp_idx;
		const Eigen::MatrixXf exp_point;
		//

	public:
		cal_exp(
			const float f_, const iden* ide_, const int id_idx_, const int exp_idx_,
			const Eigen::MatrixXf exp_point_) :
			Superclass(G_nShape),
			ide(ide_), f(f_), id_idx(id_idx_), exp_idx(exp_idx_), exp_point(exp_point_) {}

		T value(const TVector &exp ) {
			//puts("TT");
			//Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
			Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
			/*std::cout << rot << '\n';
			std::cout << tslt << '\n';*/
			float ans = 0;
			for (int i_v = 0; i_v < G_land_num; i_v++) {
				Eigen::Vector3f V;
				//printf("%d\n", i_v);
				V.setZero();
				
				//puts("QAQ");

				for (int axis = 0; axis < 3; axis++)
					for (int i_shape = 0; i_shape < G_nShape; i_shape++)
						V(axis) += exp(i_shape)*exp_point(i_shape, i_v * 3 + axis);
				//std::cout << V.transpose() << '\n';
				//puts("TAT");
				V = V + tslt;

				ans += (ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V(0)*f / V(2))*(ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V(0)*f / V(2));
				ans += (ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V(1)*f / V(2))*(ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V(1)*f / V(2));

				printf("+++++++++%.10f \n", f / V(2));
				/*printf("%d %.5f %.5f tr %.5f %.5f\n", i_v,
					ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0), ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1),
					V(0)*f / V(2), V(1)*f / V(2));*/
			}
			//printf("- - %.10f\n", ans);
			//fprintf(fp, "%.10f\n", ans);
			return ans;
		}

	//	void gradient(const TVector &exp, TVector &grad) {

	//		Eigen::Matrix3Xf point(3, G_land_num);
	//		point.setZero();
	//		for (int i_v = 0; i_v < G_land_num; i_v++) {
	//			//printf("%d\n", i_v);
	//			for (int axis = 0; axis < 3; axis++)
	//				for (int i_shape = 0; i_shape < G_nShape; i_shape++)
	//					point(axis, i_v) += exp(i_shape)*exp_point(i_shape, i_v * 3 + axis);
	//		}
	//		//std::cout << point.transpose() << '\n';
	//		point.colwise() += ide[id_idx].tslt.row(exp_idx).transpose();
	//		//std::cout << point.transpose() << '\n';
	//		for (int i_grad = 0; i_grad < G_nShape; i_grad++) {
	//			float temp = 0;
	//			for (int i_v = 0; i_v < G_land_num; i_v++)
	//				for (int axis = 0; axis < 2; axis++)
	//					temp +=
	//					2 * f*(f*point(axis, i_v) / point(2, i_v) - ide[id_idx].land_2d(G_land_num*exp_idx + i_v, axis))*
	//					(exp_point(i_grad, i_v * 3 + axis)*point(2, i_v) -
	//					 exp_point(i_grad, i_v * 3 + 2)*point(axis, i_v)) / point(2, i_v) / point(2, i_v);
	//			//printf("+%d %.10f\n", i_grad, temp);
	//			grad(i_grad) = temp;
	//		}
	//	}
	};

}
float bfgs_exp_one(float focus,iden *ide,int id_idx,int exp_idx,Eigen::MatrixXf &exp_point,Eigen::VectorXf &exp){

	puts("bfgsing expressin coeffients...");
	
	//fopen_s(&fp, "bfgs_data.txt", "w");
	typedef float T;
	typedef cppoptlib::cal_exp<T> TNNLS;
	typedef typename TNNLS::TVector TVector;
	typedef typename TNNLS::TMatrix TMatrix;
	// create model X*b for arbitrary b

	// perform non-negative least squares
	//puts("A");
	TNNLS f(focus, ide, id_idx, exp_idx, exp_point);
	//printf("5===%.10f\n", f(exp));


	/*bool probably_correct = f.checkGradient(exp);
	printf("check grad %d\n", probably_correct);
	Eigen::VectorXf g(G_nShape),texp;
	f.gradient(exp,g);
	float temp = 0.000001;
	for (int i = 0; i < 10; i++) {
		temp -= 0.0000001;
		texp = exp;
		texp(0) += temp;
		printf("%d %.10f %.10f\n", i, g(0), (f(texp) - f(exp)) / temp);
	}*/

	//puts("B");


	f.setLowerBound(TVector::Zero(G_nShape));
	f.setUpperBound(TVector::Ones(G_nShape));
	//// create initial guess (make sure it's valid >= 0)

	// init L-BFGS-B for box-constrained solving
	//puts("C");
	cppoptlib::LbfgsbSolver<TNNLS> solver;
	//Eigen::VectorXf beta=exp;
	//for (int i=0;i<G_nShape;i++) beta(i)= exp(i);
	//puts("D");
	solver.minimize(f, exp);
	//puts("E");
	std::cout << "final b = " << exp.transpose() << "\tloss:" << f(exp) << std::endl;
	std::cout << "f in argmin " << f(exp) << std::endl;
	std::cout << "Solver status: " << solver.status() << std::endl;
	std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;


	//exp = beta;
	system("pause");
	//fclose(fp);
	return f(exp);
	//return 0;
}
