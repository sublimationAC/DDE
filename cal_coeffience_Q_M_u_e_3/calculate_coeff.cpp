#include "calculate_coeff.h"

void init_exp_ide_r_t_pq(iden *ide, int ide_num) {

	puts("initializing coeffients(R,t,pq)...");
	for (int i = 0; i < ide_num; i++) {
		ide[i].center.resize(ide[i].num,2);
		ide[i].exp.resize(ide[i].num, G_nShape);
		ide[i].user.resize(G_iden_num);
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
	iden *ide, Eigen::MatrixXf &bldshps,Eigen::VectorXi &inner_land_corr,
	std :: vector<int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect) {

	puts("calclating focus for each image...");
	for (int i_id = 0; i_id < G_train_pic_id_num; i_id++) {
		if (ide[i_id].num == 0)continue;
		float L = 1, R = 3000, er_L, er_R;
		/*er_L = pre_cal_exp_ide_R_t(L, ide, bldshps, inner_land_corr, slt_line, slt_point_rect, i_id);
		er_R = pre_cal_exp_ide_R_t(R, ide, bldshps, inner_land_corr, slt_line, slt_point_rect, i_id);
		for (int rounds = 0; rounds < 50; rounds++) {
			printf("cal f %.5f %.5f %.5f %.5f\n", L, er_L, R, er_R);
			float mid_l, mid_r, er_mid_l, er_mid_r;
			mid_l = L * 2 / 3 + R / 3;
			mid_r = L / 3 + R * 2 / 3;
			er_mid_l = pre_cal_exp_ide_R_t(mid_l, ide, bldshps, inner_land_corr, slt_line, slt_point_rect, i_id);
			er_mid_r = pre_cal_exp_ide_R_t(mid_r, ide, bldshps, inner_land_corr, slt_line, slt_point_rect, i_id);
			if (er_mid_l < er_mid_r) R = mid_r, er_R = er_mid_r;
			else L = mid_l, er_L = er_mid_l;
		}*/
		/*FILE *fp;
		fopen_s(&fp, "test_f.txt", "w");*/
		for (int i = 400; i < 401; i+=20)
			printf("test cal f %d %.10f\n", i, pre_cal_exp_ide_R_t(i, ide, bldshps, inner_land_corr, slt_line, slt_point_rect, i_id));
		ide[i_id].fcs = L;
	}
}

void init_exp_ide(iden *ide, int train_id_num) {

	puts("initializing coeffients(identitiy,expression) for cal f...");
	for (int i = 0; i < train_id_num; i++) {
		ide[i].exp.array() = 1.0 / G_nShape;
		ide[i].user.array() = 1.0 / G_iden_num;
	}
}

float pre_cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor,
	std::vector <int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect,int id_idx ) {

	puts("preparing expression & other coeffients...");
	init_exp_ide(ide, G_train_pic_id_num);
	float error = 0;
	//for (int rounds = 0; rounds < 5; rounds++) {
		///////////////////////////////////////////////paper's solution
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			cal_rt_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_posit(f, ide, bldshps, inner_land_cor, id_idx, i_shape);
			Eigen::VectorXi out_land_cor(15);
			update_slt(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);
			
			out_land_cor(6) = inner_land_cor(59), out_land_cor(7) = inner_land_cor(60), out_land_cor(8) = inner_land_cor(61);
			/*std::cout << inner_land_cor << '\n';
			std::cout <<"--------------\n"<< out_land_cor << '\n';*/
			Eigen::VectorXi land_cor(G_land_num);
			for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
			for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
			//test_slt(f,ide, bldshps, land_cor, id_idx, i_exp);
			error=cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);


			//cal_3dpaper_ide(f, ide, bldshps, id_idx, i_shape, land_cor);
		}
		//cal_fixed_exp_same_ide();

		//error = cal_err();
		//printf("%d %.10f\n", rounds, error);
	//}
	return error;
}


void cal_rt_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("POSIT...");
	Eigen::MatrixX3f bs_in(G_inner_land_num, 3);
	//std::cout << ide[id_idx].land_2d << '\n';
	Eigen::MatrixX2f land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num + 15, 0, G_land_num - 15, 2);
	//std::cout << land_in << '\n';
	Eigen::RowVector2f m0 = ide[id_idx].land_2d.row(exp_idx*G_land_num + 15);
	/*std::cout << m0 << '\n';
	std::cout << land_in << '\n';*/
	cal_inner_bldshps(ide,bldshps, bs_in,inner_land_cor, id_idx, exp_idx);
	Eigen::RowVector3f M0 = bs_in.row(0);
	bs_in = bs_in.rowwise() - M0;
	//std::cout << bs_in << '\n';
	Eigen::Matrix3Xf B = bs_in.transpose()*bs_in;
	B = B.inverse()*bs_in.transpose();
	//std::cout << B << '\n';
	//puts("D");
	Eigen::VectorXf ep(land_in.rows()), ep_last(land_in.rows());
	ep.setZero();
	ep_last.setZero();
	ep_last(1) = 1;
	Eigen::Vector3f I, J;
	//puts("B");
	//std::cout << B << '\n';
	float s;
	while ((ep - ep_last).norm() > 0.1) {
		//puts("pp");
		ep_last = ep;
		Eigen::MatrixX2f xy(G_inner_land_num,2);
		xy.col(0)= ((1 + ep.array()).array()*land_in.col(0).array()) - m0(0);
		xy.col(1) = ((1 + ep.array()).array()*land_in.col(1).array()) - m0(1);
		//std::cout << xy << '\n';
		//puts("pp");
		Eigen::MatrixX2f ans = B*xy;
		I = ans.col(0), J = ans.col(1);
		//std::cout << I << '\n';
		//std::cout << J << '\n';
		float s1 = I.norm(), s2 = J.norm();
		s = (s1 + s2) / 2;
		//puts("pp");
		//std::cout << s  << ' '<< f << '\n';
		I.normalize(); J.normalize();
		/*std::cout << I << '\n';
		std::cout << J << '\n';*/
		ep = s / f * ((bs_in*(I.cross(J))).array());
		//std::cout << ep << '\n';
	}
	ide[id_idx].tslt.row(exp_idx) = Eigen::RowVector3f(m0(0),m0(1),f);
	ide[id_idx].tslt.row(exp_idx).array() /= s;
	ide[id_idx].rot.row(3 * exp_idx) = I.transpose();
	ide[id_idx].rot.row(3 * exp_idx+1) = J.transpose();
	ide[id_idx].rot.row(3 * exp_idx+2) = (I.cross(J)).transpose();
	ide[id_idx].tslt.row(exp_idx) -= (ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3)*M0.transpose()).transpose();
	//puts("-----------------------------------------");
	//std::cout << I << '\n';
	//std::cout << J << '\n';
	//std::cout << ide[id_idx].rot << '\n';
}
void cal_inner_bldshps(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::MatrixX3f &bs_in,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("calculating inner blandshapes...");
	for (int i = 0; i < G_inner_land_num; i++)
		for (int j = 0; j < 3; j++)
			bs_in(i, j) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), j);
//	std::cout << bs << '\n';
	//std::cout << inner_land_cor.transpose() << '\n';
	//printf("-%d %d\n", bs_in.rows(), bs_in.cols());
}

float cal_3d_vtx(
	iden *ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, int vtx_idx, int axis) {

	//puts("calculating one vertex coordinate...");
	float ans = 0;
	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
			ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
			*bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis);
	return ans;
}

void test_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("testing posit...");
	Eigen::MatrixX3f bs(G_inner_land_num, 3);
	cal_inner_bldshps(ide, bldshps, bs, inner_land_cor, id_idx, exp_idx);
	puts("aa");
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	puts("aabb");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx);
	puts("aacc");
	std::cout << rot << '\n';
	std::cout << tslt << '\n';
	Eigen::MatrixXf temp = (rot * bs.transpose()).colwise()+ tslt;
	temp.row(0).array() /= temp.row(2).array();
	temp.row(1).array() /= temp.row(2).array();
	temp = temp.array()*f;
	std::cout << temp << '\n';
	std::cout << ide[id_idx].land_2d.block(15,0, G_land_num - 15, 2) << '\n';
}


void update_slt(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::VectorXi &out_land_cor) {
	////////////////////////////////project

	puts("updating silhouette...");
	Eigen::Matrix3f R=ide[id_idx].rot.block(3 * exp_idx,0,3,3);
	Eigen::Vector3f T = ide[id_idx].tslt.row(exp_idx).transpose();

	puts("A");
	Eigen::VectorXi slt_cddt(G_line_num);
	Eigen::MatrixX3f slt_cddt_cdnt(G_line_num, 3);
	puts("B");
	//FILE *fp;
	//fopen_s(&fp, "test_slt.txt", "w");
	for (int i = 0; i < G_line_num; i++) {
		//printf("i %d\n", i);
		float min_v_n = 10000;
		int min_idx=0;
		Eigen::Vector3f cdnt;
		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
			//printf("j %d\n", j);
			int x = slt_line[i][j];
			//printf("x %d\n", x);
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++) 
				point[0](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
			point[0] = R * point[0] + T;
			//test															//////////////////////////////////debug
			//puts("A");
			//fprintf(fp, "%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			//printf("%.10f %.10f %.10f \n", point[0](0), point[0](1), point[0](2));
			//puts("B");

			////////////////////////////////////////////////////////////////////////////////////////////////////////

			for (int k = 0, sz = slt_point_rect[x].size(); k < sz; k++) {
				//printf("k %d\n", k);
				for (int axis = 0; axis < 3; axis++) {				
					point[1](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].first, axis);
					point[2](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].second, axis);
				}
				for (int i = 1; i < 3; i++) point[i] = R * point[i] + T;
				V[0] = point[1] - point[0];
				V[1] = point[2] - point[0];
				//puts("C");
				V[0] = V[0].cross(V[1]);
				//puts("D");
				V[0].normalize();
				nor = nor + V[0];
				//printf("__ %.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			}
			//printf("== %.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			//puts("F");
			if (nor.norm() > EPSILON) nor.normalize();
			if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0];// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		}
		//puts("H");
		//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));
		slt_cddt(i) = min_idx;
		cdnt(0) = cdnt(0)*f / cdnt(2);
		cdnt(1) = cdnt(1)*f / cdnt(2);
		slt_cddt_cdnt.row(i) = cdnt.transpose();
	}
	//fclose(fp);
	//puts("C");
	
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

void test_slt(float f,iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_land_num, 3);
	for (int i = 0; i < G_land_num; i++)
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, land_cor(i), axis);
	/*FILE *fp;
	fopen_s(&fp, "test_slt_picked_out.txt", "w");
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);*/

	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	puts("aabb");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx);
	puts("aacc");
	std::cout << bs.transpose() << '\n';
	//std::cout << tslt << '\n';
	Eigen::MatrixXf temp = (rot * bs.transpose()).colwise() + tslt;
	temp.row(0).array() /= temp.row(2).array();
	temp.row(1).array() /= temp.row(2).array();
	temp = temp.array()*f;
	std::cout << temp << '\n';
	std::cout << ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2) << '\n';

}

float cal_3dpaper_exp(
	float f, iden* ide, Eigen::MatrixXf &bldshps, 
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor) {

	puts("calculating expression coeffients by 3dpaper's way");
	float error = 0;
	Eigen::MatrixXf exp_point(G_nShape, 3 * G_land_num);
	
	cal_exp_point_matrix(ide, bldshps, id_idx, exp_idx,land_cor, exp_point);
	Eigen::VectorXf exp = ide[id_idx].exp.row(exp_idx);
	error=bfgs_exp_one(f,ide, id_idx, exp_idx, exp_point,exp);
	ide[id_idx].exp.row(exp_idx) = exp;
	return error;
}
void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx,int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result) {

	puts("prepare exp_point matrix for bfgs...");
	result.resize(G_nShape, 3 * G_land_num);
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		for (int i_v = 0; i_v < G_land_num; i_v++) {
			Eigen::Vector3f V;
			V.setZero();
			for (int j = 0; j < 3; j++)
				for (int i_id = 0; i_id < G_iden_num; i_id++)
					V(j) += ide[id_idx].user(i_id)*bldshps(i_id, i_shape*G_nVerts*3 + land_cor(i_v) * 3 + j);
			V = rot * V;
			for (int j = 0; j < 3; j++)
				result(i_shape, i_v * 3 + j) = V(j);
		}
}

