#include <iostream>
#include "meta.h"
#include "boundedproblem.h"
#include "solver/lbfgsbsolver.h"
#include "calculate_coeff.h"

// to use CppNumericalSolvers just use the namespace "cppoptlib"
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
			const float f_, const iden* ide_,const int id_idx_, const int exp_idx_, 
			const Eigen::MatrixXf exp_point_) :
			ide(ide_),f(f_),id_idx(id_idx_),exp_idx(exp_idx_),exp_point(exp_point_){}

		T value(const TVector &exp ) {
			float ans = 0;
			for (int i_v = 0; i_v < G_land_num; i_v++) {
				Eigen::Vector3f V;
				V.setZero();
				Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
				Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx);
				for (int axis = 0; axis < 3; axis++)
					for (int i_exp = 0; i_exp < G_nShape; i_exp++)
						V(axis) += exp(i_v)*exp_point(i_exp, i_v * 3 + axis);
				V = rot * V + tslt;

				ans += (ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V(0)*f / V(2))*(ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - V(0)*f / V(2));
				ans += (ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V(1)*f / V(2))*(ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - V(1)*f / V(2));
			}
			return ans;
		}

		/*void gradient(const TVector &beta, TVector &grad) {
			grad = X.transpose() * 2 * (X*beta - y);
		}*/
	};

}
float bfgs_exp_one(float focus,iden *ide,int id_idx,int exp_idx,Eigen::MatrixXf &exp_point,Eigen::VectorXf &exp){

	typedef float T;
	typedef cppoptlib::cal_exp<T> TNNLS;
	typedef typename TNNLS::TVector TVector;
	typedef typename TNNLS::TMatrix TMatrix;
	// create model X*b for arbitrary b

	// perform non-negative least squares
	TNNLS f(focus, ide, id_idx, exp_idx, exp_point);

	f.setLowerBound(TVector::Zero(G_nShape));
	f.setUpperBound(TVector::Ones(G_nShape));
	//// create initial guess (make sure it's valid >= 0)

	// init L-BFGS-B for box-constrained solving
	cppoptlib::LbfgsbSolver<TNNLS> solver;
	//Eigen::VectorXf beta=exp;
	//for (int i=0;i<G_nShape;i++) beta(i)= exp(i);
	solver.minimize(f, exp);
	std::cout << "final b = " << exp.transpose() << "\tloss:" << f(exp) << std::endl;
	std::cout << "f in argmin " << f(exp) << std::endl;
	std::cout << "Solver status: " << solver.status() << std::endl;
	std::cout << "Final criteria values: " << std::endl << solver.criteria() << std::endl;
	//exp = beta;
	//system("pause");
	return f(exp);
	//return 0;
}
