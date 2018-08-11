#include "calculate_coeff.h"

void init_exp_ide_r_t_pq(iden *ide, int ide_num) {
	puts("initializing coeffients(R,t,pq)...");
	for (int i = 0; i < ide_num; i++) {
		ide[i].center.resize(ide[i].num);
		ide[i].exp.resize(ide[i].num, G_nShape);
		ide[i].user.resize(ide[i].num, G_iden_num);
		ide[i].rot.resize(3 * ide[i].num);
		ide[i].tslt.resize(ide[i].num);
		ide[i].center.setZero();
		for (int j = 0; j < ide[i].num; j++) {
			for (int k = 0; k < G_land_num; k++)
				ide[i].center.row(j) += ide[i].land_2d.row(j*G_land_num + k);
			ide[i].center.array() /= G_land_num;
			for (int k = 0; k < G_land_num; k++)
				ide[i].land_2d.row(j*G_land_num + k)-= ide[i].center.row(j);
		}
	}
}

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name) {
	puts("loading blendshapes...");
	std::cout << name << std::endl;
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");
	for (int i = 0; i < G_iden_num; i++) {
		for (int j = 0; j < G_nShape*G_nVerts * 3; j++) 
			fread(&bldshps(i, j), sizeof(float), 1, fp);
	}
	fclose(fp);
}

void cal_f(iden *ide, Eigen::MatrixXf &bldshps) {
	float L = 0, R = 3000, er_L, er_R;
	er_L = cal_exp_ide_R_t(L, ide, bldshps);
	er_R = cal_exp_ide_R_t(R, ide, bldshps);
	for (int rounds = 0; rounds < 50; rounds++) {
		printf("%.5f %.5f %.5f %.5f\n", L, er_L, R, er_R);
		float mid_l, mid_r, er_mid_l, er_mid_r;
		mid_l = L * 2 / 3 + R / 3;
		mid_r = L / 3 + R * 2 / 3;
		er_mid_l = cal_exp_ide_R_t(mid_l, ide, bldshps);
		er_mid_r = cal_exp_ide_R_t(mid_r, ide, bldshps);
		if (er_mid_l < er_mid_r) R = mid_r, er_R = er_mid_r;
		else L = mid_l, er_L = er_mid_l;
	}
}

void init_exp_ide(iden *ide, int ide_num) {
	puts("initializing coeffients(identitiy,expression)...");
	for (int i = 0; i < ide_num; i++) {
		ide[i].exp.array() = 1.0 / G_nShape;
		ide[i].user.array() = 1.0 / G_iden_num;
	}
}

float cal_exp_ide_R_t(float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor) {
	init_exp_ide(ide, G_train_pic_id_num);
	float error = 0;
	for (int rounds = 0; rounds < 5; rounds++) {
		for (int id_idx = 0; id_idx < G_train_pic_id_num; i++) {
			
			//update_out_land_corr();
			///////////////////////////////////////////////paper's solution
			for (int exp_idx = 0; exp_idx < ide[id_idx].num; exp_idx++) {
				cal_rt_posit(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
				cal_3dpaper_ide();
				cal_3dpaper_exp();
			}
			cal_ide();
		}
		error = cal_err();
		printf("%d %.10f\n", rounds, error);
	}
	return error;
}


void cal_rt_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs_in(G_inner_land_num);
	Eigen::MatrixX2f land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num + 16, 0, exp_idx*G_land_num + G_land_num - 1, 1);
	Eigen::RowVector2f m0 = ide[id_idx].land_2d.row(exp_idx*G_land_num + 15);
	cal_inner_bldshps(ide,bldshps, bs_in,inner_land_cor, id_idx, exp_idx);
	Eigen::Matrix3f B = bs_in.transpose()*bs_in;
	Eigen::VectorXf ep(land_in.rows()), ep_last;
	ep.setZero();
	Eigen::Vector3f I, J;
	float s;
	do {
		ep_last = ep;
		Eigen::MatrixX2f xy;
		xy.col(0)= ((1 + ep.array()).array()*land_in.col(0).array()) - m0(0);
		xy.col(1) = ((1 + ep.array()).array()*land_in.col(1).array()) - m0(1);
		Eigen::MatrixX2f ans = B.inverse()*bs_in.transpose()*xy;
		I = ans.col(0), J = ans.col(1);
		float s1 = I.norm(), s2 = J.norm();
		s = (s1 + s2) / 2;
		I.normalize(); J.normalize;
		ep = s / f * ((bs_in*(I.cross(J))).array());
	} while ((ep - ep_last).norm() > 0.1);
	ide[id_idx].tslt.row(exp_idx) = Eigen::RowVector3f(m0(0),m0(1),f);
	ide[id_idx].tslt.row(exp_idx).array() /= s;
	ide[id_idx].rot.row(3 * exp_idx) = I;
	ide[id_idx].rot.row(3 * exp_idx+1) = J;
	ide[id_idx].rot.row(3 * exp_idx+2) = I.cross(J);

}

void cal_inner_bldshps(iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f bs_in,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {
	bs_in.setZero();
	for (int i = 0; i < G_inner_land_num; i++)
		for (int j = 0; j < 3; j++)
			for (int i_id = 0; i_id < G_iden_num; i_id++)
				for (int i_exp = 0; i_exp < G_nShape; i_exp++)
					bs_in(i, j) += ide[id_idx].exp(exp_idx, i_exp)*ide[id_idx].user(exp_idx, i_id)
					*bldshps(i_id, 3 * G_nVerts*i_exp + inner_land_cor(i) * 3 + j);
	bs_in = bs_in.block(1, 0, G_inner_land_num, 2).rowwise() - bs_in.row(0);
}

void cal_3dpaper_ide(iden *ide,Eigen ::
