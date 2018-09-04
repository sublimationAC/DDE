#include "calculate_coeff.h"
#define test_coef
#define test_coef_save_mesh

void init_exp_ide_r_t_pq(iden *ide, int ide_num) {

	puts("initializing coeffients(R,t,pq)...");
	for (int i = 0; i < ide_num; i++) {
		ide[i].center.resize(ide[i].num,2);
		ide[i].exp.resize(ide[i].num, G_nShape);
		ide[i].land_cor.resize(ide[i].num, G_land_num);
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

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name,Eigen::VectorXf &ide_sg_vl,std::string sg_vl_path) {

	puts("loading blendshapes...");
	std::cout << name << std::endl;
	FILE *fp;
	fopen_s(&fp, name.c_str(), "rb");
	for (int i = 0; i < G_iden_num; i++) {
		for (int j = 0; j < G_nShape*G_nVerts * 3; j++) 
			fread(&bldshps(i, j), sizeof(float), 1, fp);
	}
	fclose(fp);
	fopen_s(&fp, sg_vl_path.c_str(), "r");
	for (int i = 0; i < G_iden_num; i++) {
		fscanf_s(fp, "%f",&ide_sg_vl(i));
	}
	fclose(fp);
}

void print_bldshps(Eigen::MatrixXf &bldshps) {
	puts("print blendshapes!");
	FILE *fp;
	fopen_s(&fp, "test_svd_bldshps.txt", "w");
	fprintf(fp, "%d\n", G_nShape);
	int ide = 2;
	for (int i_exp = 0; i_exp< G_nShape; i_exp++) {
		for (int i_v = 0; i_v < G_nVerts; i_v++)
			fprintf(fp, "%.10f %.10f %.10f\n", 
				bldshps(ide, i_exp*G_nVerts*3+ i_v*3), bldshps(ide, i_exp*G_nVerts * 3 + i_v * 3+1), bldshps(ide, i_exp*G_nVerts * 3 + i_v * 3+2));
	}
	fclose(fp);
	puts("over");
}
void cal_f(
	iden *ide, Eigen::MatrixXf &bldshps,Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
	std :: vector<int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect, 
	Eigen::VectorXf &ide_sg_vl) {

	puts("calclating focus for each image...");
	FILE *fp;
#ifdef test_coef
	fopen_s(&fp, "test_coef_f_loss.txt", "w");
#else
	fopen_s(&fp, "test_coef_ide_focus.txt", "w");	
#endif // !test_coef


	
	for (int i_id = 0; i_id < G_train_pic_id_num; i_id++) {
		if (ide[i_id].num == 0)continue;
#ifndef test_coef
		float L = 100, R = 1000, er_L, er_R;
		er_L = pre_cal_exp_ide_R_t(L, ide, bldshps, inner_land_corr, jaw_land_corr,
			slt_line, slt_point_rect, i_id,ide_sg_vl);
		er_R = pre_cal_exp_ide_R_t(R, ide, bldshps, inner_land_corr, jaw_land_corr,
			slt_line, slt_point_rect, i_id, ide_sg_vl);
		fprintf(fp, "-------------------------------------------------\n");
		while (R-L>20) {
			printf("cal f %.5f %.5f %.5f %.5f\n", L, er_L, R, er_R);
			fprintf(fp, "cal f %.5f %.5f %.5f %.5f\n", L, er_L, R, er_R);
			float mid_l, mid_r, er_mid_l, er_mid_r;
			mid_l = L * 2 / 3 + R / 3;
			mid_r = L / 3 + R * 2 / 3;
			er_mid_l = pre_cal_exp_ide_R_t(mid_l, ide, bldshps, inner_land_corr, jaw_land_corr,
				slt_line, slt_point_rect, i_id, ide_sg_vl);
			er_mid_r = pre_cal_exp_ide_R_t(mid_r, ide, bldshps, inner_land_corr, jaw_land_corr, 
				slt_line, slt_point_rect, i_id, ide_sg_vl);
			if (er_mid_l < er_mid_r) R = mid_r, er_R = er_mid_r;
			else L = mid_l, er_L = er_mid_l;
		}
		ide[i_id].fcs = (L+R)/2;
#endif // !test_coef
		
		/*FILE *fp;
		fopen_s(&fp, "test_f.txt", "w");*/
		
#ifdef test_coef
		int st = 300, en = 310, step = 25;
		Eigen::VectorXf temp((en-st)/step+1);
		for (int i = st; i < en; i += step) temp((i-st)/step) = 
			pre_cal_exp_ide_R_t(i, ide, bldshps, inner_land_corr, jaw_land_corr,
				slt_line, slt_point_rect, i_id, ide_sg_vl);
		for (int i = 0; i < (en - st) / step + 1; i++) {
			printf("test cal f %d %.10f\n", st + i * step, temp(i));
			fprintf(fp, "%d %.10f\n", st + i * step, temp(i));
		}
#endif

	}
	fclose(fp);
	//FILE *fp;
	fopen_s(&fp, "test_ide_coeff.txt", "w");
	for (int i = 0; i < G_iden_num; i++)
		fprintf(fp, "%.10f\n", ide[0].user(i));
	fclose(fp);
	fopen_s(&fp, "test_exp_coeff.txt", "w");
	for (int i_exp = 0; i_exp < ide[0].num; i_exp++) {
		fprintf(fp, "------------------------------------------\n");
		for (int i = 0; i < G_nShape; i++)
			fprintf(fp, "%.10f\n", ide[0].exp(i_exp, i));
	}
	fclose(fp);

}

std::string cal_coef_land_name = "test_coef_land_olsgm_25.txt";
std::string cal_coef_mesh_name = "test_coef_mesh_olsgm_25.txt";

float pre_cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor, Eigen::VectorXi &jaw_land_corr,
	std::vector <int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect,int id_idx,
	Eigen::VectorXf &ide_sg_vl) {

	puts("preparing expression & other coeffients...");
	init_exp_ide(ide, G_train_pic_id_num);
	float error = 0;
	
	int tot_r = 4;
	Eigen::VectorXf temp(tot_r);
	//fprintf(fp, "%d\n",tot_r);
	FILE *fp;
#ifdef test_coef_save_mesh
	
	fopen_s(&fp, cal_coef_land_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r+1);
	fclose(fp);

	fopen_s(&fp, cal_coef_mesh_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r+1);
	fclose(fp);
#endif
	//float error_last=0;
	for (int rounds = 0; rounds < tot_r; rounds++) {
		///////////////////////////////////////////////paper's solution
		
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			printf("calculate %d id %d exp:\n", id_idx, i_exp);
			cal_rt_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_posit(f, ide, bldshps, inner_land_cor, id_idx, i_shape);
			Eigen::VectorXi out_land_cor(15);
			update_slt(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor, jaw_land_corr);
			/*std::cout << inner_land_cor << '\n';
			std::cout <<"--------------\n"<< out_land_cor << '\n';*/
			Eigen::VectorXi land_cor(G_land_num);
			for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
			for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
			ide[id_idx].land_cor.row(i_exp) = land_cor.transpose();
			//test_slt(f,ide, bldshps, land_cor, id_idx, i_exp);
#ifdef test_coef_save_mesh
			if (rounds == 0) {
				test_coef_land(ide, bldshps, id_idx, 0);
				test_coef_mesh(ide, bldshps, id_idx, 0);
			}
#endif
			error=cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
			error=cal_3dpaper_ide(f, ide, bldshps, id_idx, i_exp, land_cor, ide_sg_vl);
		}
		error=cal_fixed_exp_same_ide(f, ide, bldshps, id_idx, ide_sg_vl);
		
		printf("+++++++++++++%d %.10f\n", rounds, error);
#ifdef test_coef_save_mesh
		test_coef_land(ide,bldshps,id_idx,0);
		test_coef_mesh(ide, bldshps, id_idx, 0);
#endif
		//if (fabs(error_last - error) < 20) break;
		//error_last = error;
		temp(rounds) = error;
	}
	for (int i = 0; i < tot_r; i++) printf("it %d err %.10f\n",i,temp(i));
	return error;
}

void init_exp_ide(iden *ide, int train_id_num) {

	puts("initializing coeffients(identitiy,expression) for cal f...");
	for (int i = 0; i < train_id_num; i++) {
		ide[i].exp= Eigen::MatrixXf::Constant(ide[i].num, G_nShape, 1.0 / G_nShape);
		ide[i].user= Eigen::MatrixXf::Constant(G_iden_num, 1, 1.0 / G_iden_num);
	}
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
	int cnt = 0;
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
		if (cnt++ > 10) break;
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
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::VectorXi &out_land_cor,
	Eigen::VectorXi &jaw_land_corr) {
	////////////////////////////////project

	puts("updating silhouette...");
	Eigen::Matrix3f R=ide[id_idx].rot.block(3 * exp_idx,0,3,3);
	Eigen::Vector3f T = ide[id_idx].tslt.row(exp_idx).transpose();

	//puts("A");
	Eigen::VectorXi slt_cddt(G_line_num+G_jaw_land_num);
	Eigen::MatrixX3f slt_cddt_cdnt(G_line_num + G_jaw_land_num, 3);
	//puts("B");
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
	for (int i_jaw = 0; i_jaw < G_jaw_land_num; i_jaw++) {
		Eigen::Vector3f point;
		for (int axis = 0; axis < 3; axis++)
			point(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, jaw_land_corr(i_jaw), axis);
		point = R * point + T;
		point(0) = point(0)*f / point(2);
		point(1) = point(1)*f / point(2);
		slt_cddt(i_jaw + G_line_num) = jaw_land_corr(i_jaw);
		slt_cddt_cdnt.row(i_jaw + G_line_num) = point.transpose();
	}
	for (int i = 0; i < 15; i++) {
		float min_dis = 10000;
		int min_idx;
		for (int j = 0; j < G_line_num+G_jaw_land_num; j++) {
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
	error=ceres_exp_one(f,ide, id_idx, exp_idx, exp_point,exp);
	ide[id_idx].exp.row(exp_idx) = exp;
	return error;
}
void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx,int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result) {

	puts("prepare exp_point matrix for bfgs/ceres...");
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

float cal_3dpaper_ide(
	float f, iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::VectorXf &ide_sg_vl) {

	puts("calculating identity coeffients by 3dpaper's way");
	float error = 0;
	Eigen::MatrixXf id_point(G_iden_num, 3 * G_land_num);

	cal_id_point_matrix(ide, bldshps, id_idx, exp_idx, land_cor, id_point);
	Eigen::VectorXf user = ide[id_idx].user;
	error = ceres_user_one(f, ide, id_idx, exp_idx, id_point, user, ide_sg_vl);
	ide[id_idx].user = user;
	return error;
}
void cal_id_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result) {

	puts("prepare user_point matrix for bfgs/ceres...");
	result.resize(G_iden_num, 3 * G_land_num);
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_v = 0; i_v < G_land_num; i_v++) {
			Eigen::Vector3f V;
			V.setZero();
			for (int j = 0; j < 3; j++)
				for (int i_shape = 0; i_shape < G_nShape; i_shape++)
					V(j) += ide[id_idx].exp(id_idx,i_shape)*bldshps(i_id, i_shape*G_nVerts * 3 + land_cor(i_v) * 3 + j);
			V = rot * V;
			for (int j = 0; j < 3; j++)
				result(i_id, i_v * 3 + j) = V(j);
		}
}


float cal_fixed_exp_same_ide(float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx,
	Eigen::VectorXf &ide_sg_vl) {

	puts("calculating identity coeffients by 3dpaper's way while fixing the expression coeffients");
	float error = 0;
	Eigen::MatrixXf id_point_fix_exp(ide[id_idx].num*G_iden_num,G_land_num*3);

	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		Eigen::VectorXi land_cor(G_land_num);
		Eigen::MatrixXf id_point(G_iden_num, 3 * G_land_num);
		land_cor = ide[id_idx].land_cor.row(i_exp);
		cal_id_point_matrix(ide, bldshps, id_idx, i_exp, land_cor, id_point);
		
		id_point_fix_exp.block(i_exp*G_iden_num, 0, G_iden_num, G_land_num * 3)= id_point;
	}
	Eigen::VectorXf user = ide[id_idx].user;
	error = ceres_user_fixed_exp(f, ide, id_idx, id_point_fix_exp, user, ide_sg_vl);
	ide[id_idx].user = user;
	return error;
}

void test_coef_land(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_land_num, 3);
	for (int i = 0; i < G_land_num; i++)
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), axis);

	FILE *fp;
	fopen_s(&fp, cal_coef_land_name.c_str(), "a");
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);
}

void test_coef_mesh(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_nVerts, 3);
	for (int i = 0; i < G_nVerts; i++)
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, i, axis);

	FILE *fp;
	fopen_s(&fp, cal_coef_mesh_name.c_str(), "a");
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);
}

void cal_mesh_land(Eigen::MatrixXf &bldshps) {
	puts("calculating mesh from test_coeff_ide&exp");
	FILE *fpr, *fpw;
	fopen_s(&fpr, "./server/test_ide_coeff.txt", "r");
	Eigen::VectorXf ide(G_iden_num), exp(G_nShape);
	for (int i_id = 0; i_id < G_iden_num; i_id++) fscanf_s(fpr, "%f", &ide(i_id));
	fclose(fpr);
	fopen_s(&fpr, "./server/test_exp_coeff.txt", "r");
	fopen_s(&fpw, "./test_exp_coeff_mesh.txt", "w");
	int num = 4;
	fprintf(fpw, "%d\n", num);
	Eigen::MatrixXf mesh(G_nVerts, 3);
	//char s[500];
	for (int j_no = 0; j_no < num; j_no++) {
		printf("%d \n", j_no);
		fscanf_s(fpr, "------------------------------------------");
		//puts(s);
		for (int i_exp = 0; i_exp < G_nShape; i_exp++)
			fscanf_s(fpr, "%f", &exp(i_exp));
		for (int i = 0; i < G_nVerts; i++)
			for (int axis = 0; axis < 3; axis++) {
				mesh(i, axis) = 0;
				for (int i_id = 0; i_id < G_iden_num; i_id++)
					for (int i_exp=0; i_exp < G_nShape; i_exp++)
						mesh(i, axis) += bldshps(i_id, i_exp*G_nVerts * 3 + i * 3 + axis)*ide(i_id)*exp(i_exp);
			}
		for (int i = 0; i < G_nVerts; i++)
			fprintf(fpw, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));
	}
	fclose(fpw);
}







/*
test cal f 0 879.8588256836
test cal f 1 880.0136108398
test cal f 2 864.6113281250
test cal f 3 875.2151489258
test cal f 4 888.7425537109
test cal f 5 885.1669311523
test cal f 6 906.4459228516
test cal f 7 923.1376342773
test cal f 8 912.3187255859
test cal f 9 940.9351196289
test cal f 10 957.6480712891
test cal f 11 977.1648559570
test cal f 12 999.8181152344
test cal f 13 1015.5399169922
test cal f 14 1030.0798339844
test cal f 15 1048.9896240234
test cal f 16 1103.2413330078
test cal f 17 1165.3564453125
test cal f 18 1178.8156738281
test cal f 19 1174.0402832031
test cal f 20 1190.1907958984
test cal f 21 1191.0943603516
test cal f 22 -431602080.0000000000

test cal f 100 11336.4443359375
test cal f 125 785.0991821289
test cal f 150 588.3906860352
test cal f 175 613.8989868164
test cal f 200 725.7813720703
test cal f 225 734.6677246094
test cal f 250 728.8943481445
test cal f 275 716.5715942383
test cal f 300 750.6123657227
test cal f 325 754.9862670898
test cal f 350 708.9167480469
test cal f 375 695.1560668945
test cal f 400 700.0017700195
test cal f 425 652.4443969727
test cal f 450 612.8259887695
test cal f 475 615.2667236328
test cal f 500 619.5740966797
test cal f 525 620.4445800781
test cal f 550 602.8044433594
test cal f 575 605.1570434570
test cal f 600 607.3097534180
test cal f 625 607.7971801758
test cal f 650 686.6637573242
test cal f 675 691.1859130859
test cal f 700 693.2089843750
test cal f 725 691.2633666992
test cal f 750 690.7508544922
test cal f 775 695.6425170898
test cal f 800 696.4915771484
test cal f 825 697.4221191406
test cal f 850 698.5968627930
test cal f 875 705.6496582031
test cal f 900 706.4727172852
test cal f 925 707.5124511719
test cal f 950 708.5409545898
test cal f 975 709.5115966797
test cal f 1000 715.6920776367
test cal f 1025 716.5400390625
test cal f 1050 717.3754882813
test cal f 1075 726.9896850586

test cal f 100 1519.2838134766
test cal f 125 947.2608642578
test cal f 150 908.4181518555
test cal f 175 867.7921142578
test cal f 200 912.3013916016
test cal f 225 884.7678222656
test cal f 250 879.8588256836
test cal f 275 880.0136108398
test cal f 300 -431602080.0000000000
*/

/*
test cal f 100 1977.3437500000
test cal f 125 1574.7167968750
test cal f 150 1455.8612060547
test cal f 175 1390.5677490234
test cal f 200 1431.9500732422
test cal f 225 1771.1562500000
test cal f 250 1749.6376953125
test cal f 275 1833.3856201172
test cal f 300 1876.6953125000
test cal f 325 1906.9680175781
test cal f 350 1983.8415527344
test cal f 375 1993.3524169922
test cal f 400 1815.2987060547
test cal f 425 1867.6469726563
test cal f 450 1918.1181640625
test cal f 475 1965.1877441406
test cal f 500 1997.8572998047
test cal f 525 2041.1361083984
test cal f 550 2084.7612304688
test cal f 575 2122.1452636719
test cal f 600 2199.0795898438
test cal f 625 2199.3693847656
test cal f 650 2230.9003906250
test cal f 675 2255.6203613281
test cal f 700 2278.8603515625
test cal f 725 2301.1096191406
test cal f 750 2321.9831542969
test cal f 775 2341.6987304688
test cal f 800 2360.1850585938
test cal f 825 2378.5395507813
test cal f 850 2395.2028808594
test cal f 875 2411.1774902344
test cal f 900 2426.2807617188
test cal f 925 2440.7792968750
test cal f 950 2454.6340332031
test cal f 975 2466.4677734375
*/

/*
55 1500
test cal f 100 2313.8242187500
test cal f 125 1838.8928222656
test cal f 150 1466.9392089844
test cal f 175 1257.8214111328
test cal f 200 998.8993530273
test cal f 225 893.9645385742
test cal f 250 832.9778442383
test cal f 275 800.5964965820
test cal f 300 761.2866821289
test cal f 325 758.0947265625
test cal f 350 737.5111694336
test cal f 375 735.3290405273
test cal f 400 734.4934692383
test cal f 425 733.8238525391
test cal f 450 738.4660644531
test cal f 475 735.3060302734
test cal f 500 759.8150024414
test cal f 525 778.3350830078
test cal f 550 797.4025878906
test cal f 575 811.3441162109
test cal f 600 798.0612792969
test cal f 625 872.0380859375
test cal f 650 884.2244262695
test cal f 675 821.6886596680
test cal f 700 907.8507690430
test cal f 725 921.5566406250
test cal f 750 932.9988403320
test cal f 775 944.0285034180
test cal f 800 954.7307128906
test cal f 825 966.6048583984
test cal f 850 975.3027343750
test cal f 875 984.8454589844
test cal f 900 994.2359619141
test cal f 925 1002.8670043945
test cal f 950 1011.5712280273
test cal f 975 1014.2359008789

*/

/*
55 2000
test cal f 100 2309.4392089844
test cal f 125 1889.3881835938
test cal f 150 1566.9396972656
test cal f 175 1334.3238525391
test cal f 200 1034.2542724609
test cal f 225 960.6875000000
test cal f 250 861.1200561523
test cal f 275 842.5196533203
test cal f 300 796.2081298828
test cal f 325 766.7290649414
test cal f 350 755.9409790039
test cal f 375 742.1948242188
test cal f 400 733.4272460938
test cal f 425 729.4384155273
test cal f 450 729.5432739258
test cal f 475 730.4609985352
test cal f 500 724.5762329102
test cal f 525 733.4087524414
test cal f 550 736.9336547852
test cal f 575 742.6765747070
test cal f 600 750.3646240234
test cal f 625 759.3997192383
test cal f 650 767.2517089844
test cal f 675 774.6967773438
test cal f 700 782.2497558594
test cal f 725 806.3529663086
test cal f 750 818.5466918945
test cal f 775 826.3574829102
test cal f 800 833.2935180664
test cal f 825 840.9088134766
test cal f 850 847.7034912109
test cal f 875 854.6730957031
test cal f 900 860.7893676758
test cal f 925 866.4020996094
test cal f 950 871.8389282227
test cal f 975 874.2234497070
test cal f 1000 879.3839721680
test cal f 1025 883.9304199219
test cal f 1050 888.7633056641
test cal f 1075 894.2439575195

*/