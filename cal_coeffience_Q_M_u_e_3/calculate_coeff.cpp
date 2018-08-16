#include "calculate_coeff.h"

void init_exp_ide_r_t_pq(iden *ide, int ide_num) {

	puts("initializing coeffients(R,t,pq)...");
	for (int i = 0; i < ide_num; i++) {
		ide[i].center.resize(ide[i].num,2);
		ide[i].exp.resize(ide[i].num, G_nShape);
		ide[i].user.resize(ide[i].num, G_iden_num);
		ide[i].rot.resize(3 * ide[i].num,3);
		ide[i].tslt.resize(ide[i].num,3);
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

void cal_f(
	iden *ide, Eigen::MatrixXf &bldshps,Eigen::VectorXi inner_land_corr,
	std :: vector<int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect) {

	float L = 0, R = 3000, er_L, er_R;
	er_L = cal_exp_ide_R_t(L, ide, bldshps, inner_land_corr, slt_line,slt_point_rect);
	er_R = cal_exp_ide_R_t(R, ide, bldshps, inner_land_corr, slt_line, slt_point_rect);
	for (int rounds = 0; rounds < 50; rounds++) {
		printf("%.5f %.5f %.5f %.5f\n", L, er_L, R, er_R);
		float mid_l, mid_r, er_mid_l, er_mid_r;
		mid_l = L * 2 / 3 + R / 3;
		mid_r = L / 3 + R * 2 / 3;
		er_mid_l = cal_exp_ide_R_t(mid_l, ide, bldshps, inner_land_corr, slt_line, slt_point_rect);
		er_mid_r = cal_exp_ide_R_t(mid_r, ide, bldshps, inner_land_corr, slt_line, slt_point_rect);
		if (er_mid_l < er_mid_r) R = mid_r, er_R = er_mid_r;
		else L = mid_l, er_L = er_mid_l;
	}
}

void init_exp_ide(iden *ide, int train_id_num) {

	puts("initializing coeffients(identitiy,expression)...");
	for (int i = 0; i < train_id_num; i++) {
		ide[i].exp.array() = 1.0 / G_nShape;
		ide[i].user.array() = 1.0 / G_iden_num;
	}
}

float cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor,
	std::vector <int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect) {

	init_exp_ide(ide, G_train_pic_id_num);
	float error = 0;
	for (int rounds = 0; rounds < 5; rounds++) {
		for (int id_idx = 0; id_idx < G_train_pic_id_num; id_idx++) {		
			///////////////////////////////////////////////paper's solution
			for (int exp_idx = 0; exp_idx < ide[id_idx].num; exp_idx++) {
				cal_rt_posit(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
				Eigen::VectorXi out_land_cor(15);
				update_slt(f,ide,bldshps,id_idx,exp_idx,slt_line, slt_point_rect,out_land_cor);
				out_land_cor(6) = inner_land_cor(59),out_land_cor(7) = inner_land_cor(60),out_land_cor(8) = inner_land_cor(61);
				cal_3dpaper_exp();
				cal_3dpaper_ide();
			}
			cal_fixed_exp_same_ide();
		}
		//error = cal_err();
		printf("%d %.10f\n", rounds, error);
	}
	return error;
}


void cal_rt_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs_in;
	bs_in.resize(G_inner_land_num, 3);
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
		I.normalize(); J.normalize();
		ep = s / f * ((bs_in*(I.cross(J))).array());
	} while ((ep - ep_last).norm() > 0.1);
	ide[id_idx].tslt.row(exp_idx) = Eigen::RowVector3f(m0(0),m0(1),f);
	ide[id_idx].tslt.row(exp_idx).array() /= s;
	ide[id_idx].rot.row(3 * exp_idx) = I;
	ide[id_idx].rot.row(3 * exp_idx+1) = J;
	ide[id_idx].rot.row(3 * exp_idx+2) = I.cross(J);

}
float cal_3d_vtx(
	iden *ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, int vtx_idx, int axis) {
	float ans = 0;
	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_exp = 0; i_exp < G_nShape; i_exp++)
			ans += ide[id_idx].exp(exp_idx, i_exp)*ide[id_idx].user(exp_idx, i_id)
			*bldshps(i_id, 3 * G_nVerts*i_exp + vtx_idx * 3 + axis);
	return ans;
}

void cal_inner_bldshps(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f bs_in,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	for (int i = 0; i < G_inner_land_num; i++)
		for (int j = 0; j < 3; j++)
			bs_in(i, j) = cal_3d_vtx(ide,bldshps,id_idx,exp_idx,inner_land_cor(i),j);
	bs_in = bs_in.block(1, 0, G_inner_land_num, 2).rowwise() - bs_in.row(0);
}

bool use[G_nVerts];
void update_slt(
	float f, iden* ide, Eigen::MatrixXf bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::VectorXf out_land_cor) {
	////////////////////////////////project

	Eigen::Matrix3f R;
	Eigen::Vector3f T = ide[id_idx].tslt.row(exp_idx);
	for (int i = 0; i < 3; i++)
		R.row(i) = ide[id_idx].rot.row(3 * exp_idx + i);


	Eigen::VectorXi slt_cddt(G_line_num);
	Eigen::MatrixX3f slt_cddt_cdnt(G_line_num, 3);
	for (int i = 0; i < G_line_num; i++) {
		float min_v_n = 10000;
		int min_idx;
		Eigen::Vector3f cdnt;
		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
			int x = slt_line[i][j];
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++) 
				point[0](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
			for (int k = 0, sz = slt_point_rect[x].size(); k < sz; k++) {
				for (int axis = 0; axis < 3; axis++) {				
					point[1](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].first, axis);
					point[2](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].second, axis);
				}
				for (int i = 0; i < 3; i++) point[i] = R * point[i] + T;
				V[0] = point[1] - point[0];
				V[1] = point[2] - point[0];
				V[0] = V[0].cross(V[1]);
				V[0].normalize();
				nor = nor + V[0];
			}
			if (nor.norm() > EPSILON) nor.normalize();
			if (fabs(nor(3)) < min_v_n) min_v_n = fabs(nor(3)), min_idx = x, cdnt = point[0];
		}
		slt_cddt(i) = min_idx;
		cdnt(0) = cdnt(0)*f / cdnt(2);
		cdnt(1) = cdnt(1)*f / cdnt(2);
		slt_cddt_cdnt.row(i) = cdnt.transpose();
	}

	for (int i = 0; i < 15; i++) {
		if (i == 6 || i == 7 || i == 8) continue;
		float min_dis = 10000;
		int min_idx;
		for (int j = 0; j < G_line_num; j++) {
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 1));
			if (temp < min_dis) min_dis = temp, min_idx = j;
		}
		out_land_cor(i) = slt_cddt(min_idx);
	}
}
					
void cal_3dpaper_exp(
	float f, iden* ide, Eigen::MatrixXf bldshps, int id_idx, int exp_idx, Eigen::VectorXf inner_land_cor,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::VectorXf out_land_cor){

	for (int i_exp;i_exp<G_nShape;i_exp++){

