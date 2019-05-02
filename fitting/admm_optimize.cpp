#include "admm_optimize.h"

//float admm_all(
//	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &bldshps) {
//
//	int ite = 10;
//	for (int o=0;o<ite;o++){



float evaluate_expuser(float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &expuser_point, Eigen::RowVectorXf &expuser) {

	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
	float ans = 0;
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::Vector3f V = (expuser * expuser_point.block(0, i_v * 3, expuser_point.rows(), 3)).transpose();
		V = V + tslt;

		float dis0 = ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - ide[id_idx].center(exp_idx, 0) - V(0) * focus / V(2);
		float dis1 = ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - ide[id_idx].center(exp_idx, 1) - V(1) * focus / V(2);
		printf("%d %.5f %.5f\n", i_v, dis0, dis1);
		ans += dis0 * dis0 + dis1 * dis1;
	}
	return ans;
}

void drvt_expuser(
	float focus, iden *ide, int id_idx, int exp_idx, 
	Eigen::MatrixXf &expuser_point, Eigen::RowVectorXf &expuser,Eigen::RowVectorXf &dexpuser) {
	
	dexpuser.resize(1, expuser.cols());
	dexpuser.setZero();
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
	float ans = 0;
	for (int i_v = 0; i_v < G_land_num; i_v++) {
		Eigen::MatrixX3f RB = expuser_point.block(0, i_v * 3, expuser_point.rows(), 3);
		Eigen::Vector3f V = (expuser * RB).transpose();
		V = V + tslt;

		float dis0 = ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 0) - ide[id_idx].center(exp_idx, 0) - V(0) * focus / V(2);
		float dis1 = ide[id_idx].land_2d(G_land_num*exp_idx + i_v, 1) - ide[id_idx].center(exp_idx, 1) - V(1) * focus / V(2);

		dexpuser += -dis0 / (V(2)*V(2))*focus*(V(2)*(RB.col(0).transpose()) - V(0)*(RB.col(2).transpose()));
		dexpuser += -dis1 / (V(2)*V(2))*focus*(V(2)*(RB.col(1).transpose()) - V(1)*(RB.col(2).transpose()));

	}
	dexpuser.normalize();
}

float admm_exp_one(
	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &exp_point, Eigen::RowVectorXf &exp) {
	puts("admming exp:");

	const int ite = 30;
	float alpha = 0.01;
	for (int o = 0; o < ite; o++) {
		printf("bf ide: %d err:%.5f\n", o, evaluate_expuser(focus, ide, id_idx, exp_idx, exp_point, exp));
		Eigen::RowVectorXf dexp;
		drvt_expuser(focus, ide, id_idx, exp_idx, exp_point, exp, dexp);
		dexp(0) = 0;
		dexp.normalize();

		float now = evaluate_expuser(focus, ide, id_idx, exp_idx, exp_point, exp);
		Eigen::RowVectorXf tdexp;
		tdexp.resize(1, G_nShape);
		float dt = 0.0001;
		for (int i = 1; i < G_nShape; i++) {
			exp(i) += dt;
			tdexp(i) = (evaluate_expuser(focus, ide, id_idx, exp_idx, exp_point, exp) - now) / dt;
			exp(i) -= dt;
		}
		tdexp(0) = 0;
		tdexp.normalize();
		std::cout << "dexp:" << dexp << "\n";
		std::cout << "tdexp:" << tdexp << "\n";
		
		exp -= alpha * dexp;

		for (int i = 1; i < G_nShape; i++) {
			exp(i) = min((float)(1.0), exp(i));
			exp(i) = max((float)0, exp(i));
		}

		printf("ide: %d err:%.5f\n", o, evaluate_expuser(focus, ide, id_idx, exp_idx, exp_point, exp));
	}
	return evaluate_expuser(focus, ide, id_idx, exp_idx, exp_point, exp);
}

float admm_user_one(
	float focus, iden *ide, int id_idx, int exp_idx, Eigen::MatrixXf &user_point, Eigen::VectorXf &user,float lmd) {
	puts("admming user:");

	Eigen::RowVectorXf row_user = user.transpose();

	printf("%d %d\n", user_point.rows(), user_point.cols());
	const int ite = 30;
	float alpha = 0.01;
	for (int o = 0; o < ite; o++) {
		float evlt = evaluate_expuser(focus, ide, id_idx, exp_idx, user_point, row_user);		
		evlt += lmd * (row_user.sum()-1);

		printf("bf ide: %d err:%.5f\n", o, evlt);
		Eigen::RowVectorXf duser;
		drvt_expuser(focus, ide, id_idx, exp_idx, user_point, row_user, duser);
		duser.array() += lmd;
		duser.normalize();

		float now = evaluate_expuser(focus, ide, id_idx, exp_idx, user_point, row_user)+ lmd * (row_user.sum() - 1);
		Eigen::RowVectorXf tduser;
		tduser.resize(1, G_iden_num);
		float dt = 0.0001;
		for (int i = 0; i < G_iden_num; i++) {
			row_user(i) += dt;
			tduser(i) =
				(evaluate_expuser(focus, ide, id_idx, exp_idx, user_point, row_user) + lmd * (row_user.sum() - 1)
					- now) / dt;
			row_user(i) -= dt;
		}
		tduser.normalize();
		std::cout << "duser:" << duser << "\n";
		std::cout << "tduser:" << tduser << "\n";

		row_user -= alpha * duser;
		for (int i = 0; i < G_iden_num; i++) {
			row_user(i) = min((float)(1.0), row_user(i));
			row_user(i) = max((float)0, row_user(i));
		}
		printf("ide: %d err:%.5f\n", o, evaluate_expuser(focus, ide, id_idx, exp_idx, user_point, row_user) + lmd * (row_user.sum() - 1));
	}
	user = row_user.transpose();
	return evaluate_expuser(focus, ide, id_idx, exp_idx, user_point, row_user) + lmd * (row_user.sum() - 1);
}