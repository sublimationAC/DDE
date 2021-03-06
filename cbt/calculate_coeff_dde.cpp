#include "calculate_coeff_dde.hpp"
//#define test_coef
//#define test_coef_save_mesh
//#define test_posit_by2dland

void fit_solve(
	std::vector<cv::Point2d> &landmarks, Eigen::MatrixXf &bldshps, 
	Eigen::VectorXi &inner_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl,DataPoint &data) {
	puts("fitting for the first frame...!");
	iden ide[1];
	ide[0].num = 1;
	ide[0].land_2d.resize(G_land_num, 2);
	data.land_2d.resize(G_land_num, 2);
	for (int i = 0; i < G_land_num; i++) {
		ide[0].land_2d(i, 0) = landmarks[i].x, ide[0].land_2d(i, 1) = data.image.rows - landmarks[i].y;
		data.land_2d(i, 0) = landmarks[i].x, data.land_2d(i, 1) = data.image.rows - landmarks[i].y;
	}
#ifdef perspective
	ide[0].img_size.resize(1, 2);
	ide[0].img_size(0, 0) = data.image.cols;
	ide[0].img_size(0, 1) = data.image.rows;
#endif // perspective

	init_exp_ide_r_t_pq(ide, 1);
	//data.land_2d = ide[0].land_2d;

#ifdef perspective
	cal_f(ide,bldshps, inner_land_corr, slt_line, slt_point_rect, ide_sg_vl);
#endif // perspective
#ifdef normalization
	solve(ide, bldshps, inner_land_corr, slt_line, slt_point_rect, ide_sg_vl);
#endif // normalization

	puts("A");
	data.center = ide[0].center;
	puts("B");
	data.landmarks = landmarks;
	puts("E");
	data.land_cor = ide[0].land_cor.transpose();
	puts("D");
#ifdef perspective
	data.fcs = ide[0].fcs;
#endif // perspective

#ifdef normalization
	data.s = ide[0].s;
#endif // normalization

	puts("D");
	data.user = ide[0].user;
	puts("D");
	data.shape.exp = ide[0].exp.row(0).transpose();

	//data.shape.rot = ide[0].rot.block(0,0,3,3);
	data.shape.angle = get_uler_angle_zyx(ide[0].rot.block(0, 0, 3, 3));
#ifdef perspective
	data.shape.tslt = ide[0].tslt.row(0).transpose();
#endif // persepctive
#ifdef normalization
	data.shape.tslt = ide[0].tslt.row(0);
#endif // normalization
	puts("F");
	recal_dis_ang(data, bldshps);
}

void init_exp_ide_r_t_pq(iden *ide, int ide_num) {

	puts("initializing coeffients(R,t,pq)...");
	for (int i = 0; i < ide_num; i++) {

		ide[i].exp.resize(ide[i].num, G_nShape);
		ide[i].land_cor.resize(ide[i].num, G_land_num);
		ide[i].land_cor.setZero();
		ide[i].user.resize(G_iden_num);
		ide[i].rot.resize(3 * ide[i].num, 3);
		//ide[i].rot.setZero();
		//ide[i].rot(0, 0) = ide[i].rot(1, 1) = ide[i].rot(2, 2) = 1;
		ide[i].tslt.resize(ide[i].num, 3);
		ide[i].tslt.setZero();
		ide[i].dis.resize(G_land_num*ide[i].num, 2);

		ide[i].center.resize(ide[i].num, 2);
		ide[i].center.setZero();

#ifdef normalization
		ide[i].s.resize(ide[i].num * 2, 3);
#endif // normalization

		//for (int j = 0; j < ide[i].num; j++) 
		//	for (int k = 0; k < G_land_num; k++)
		//		ide[i].center.row(j) += ide[i].land_2d.row(j*G_land_num + k);
		//ide[i].center.array() /= G_land_num;

		ide[i].center.array() = ide[i].img_size.array() / 2;
		//for (int j = 0; j < ide[i].num; j++) 
		//	for (int k = 0; k < G_land_num; k++)
		//		ide[i].land_2d.row(j*G_land_num + k) -= ide[i].center.row(j);
	}
}

void print_bldshps(Eigen::MatrixXf &bldshps) {
	puts("print blendshapes!");
	FILE *fp;
	//fopen_s(&fp, "test_svd_bldshps_test.txt", "w");
	//fprintf(fp, "%d\n", 10);
	//int ide = 1;
	//for (int i_exp = 0; i_exp< 10; i_exp++) {
	//	for (int i_v = 0; i_v < G_nVerts; i_v++)
	//		fprintf(fp, "%.6f %.6f %.6f\n", 
	//			bldshps(ide, i_exp*G_nVerts*3+ i_v*3), bldshps(ide, i_exp*G_nVerts * 3 + i_v * 3+1), bldshps(ide, i_exp*G_nVerts * 3 + i_v * 3+2));
	//}
	//fclose(fp);
	fopen_s(&fp, "test_svd_bldshps_test.txt", "w");
	fprintf(fp, "%d\n", 10);
	int exp = 0;
	for (int i_ide = 0; i_ide < 10; i_ide++) {
		for (int i_v = 0; i_v < G_nVerts; i_v++)
			fprintf(fp, "%.6f %.6f %.6f\n",
				bldshps(i_ide, exp*G_nVerts * 3 + i_v * 3), bldshps(i_ide, exp*G_nVerts * 3 + i_v * 3 + 1), bldshps(i_ide, exp*G_nVerts * 3 + i_v * 3 + 2));
	}
	fclose(fp);



	puts("over");
}

void cal_f(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl) {

	puts("calclating focus for each image...");
	FILE *fp;
#ifdef test_coef
	fopen_s(&fp, "test_coef_f_loss.txt", "w");
#else
	fopen_s(&fp, "test_coef_ide_focus.txt", "w");
#endif // !test_coef



	for (int i_id = 0; i_id < 1; i_id++) {
		if (ide[i_id].num == 0)continue;
#ifndef test_coef
		/*float L = 100, R = 1000, er_L, er_R;
		er_L = pre_cal_exp_ide_R_t(L, ide, bldshps, inner_land_corr,
			slt_line, slt_point_rect, i_id,ide_sg_vl);
		er_R = pre_cal_exp_ide_R_t(R, ide, bldshps, inner_land_corr,
			slt_line, slt_point_rect, i_id, ide_sg_vl);
		fprintf(fp, "-------------------------------------------------\n");
		while (R-L>20) {
			printf("i_id: %d cal f %.5f %.5f %.5f %.5f\n",i_id, L, er_L, R, er_R);
			fprintf(fp, "i_id: %d cal f %.5f %.5f %.5f %.5f\n",i_id, L, er_L, R, er_R);
			float mid_l, mid_r, er_mid_l, er_mid_r;
			mid_l = L * 2 / 3 + R / 3;
			mid_r = L / 3 + R * 2 / 3;
			er_mid_l = pre_cal_exp_ide_R_t(mid_l, ide, bldshps, inner_land_corr,
				slt_line, slt_point_rect, i_id, ide_sg_vl);
			er_mid_r = pre_cal_exp_ide_R_t(mid_r, ide, bldshps, inner_land_corr,
				slt_line, slt_point_rect, i_id, ide_sg_vl);
			if (er_mid_l < er_mid_r) R = mid_r, er_R = er_mid_r;
			else L = mid_l, er_L = er_mid_l;
		}
		ide[i_id].fcs = (L+R)/2;*/
		float f2 = 2500, er_f2;
		er_f2 = pre_cal_exp_ide_R_t(f2, ide, bldshps, inner_land_corr,
			slt_line, slt_point_rect, i_id, ide_sg_vl);
		float f1 = 800, er_f1;
		er_f1 = pre_cal_exp_ide_R_t(f1, ide, bldshps, inner_land_corr,
			slt_line, slt_point_rect, i_id, ide_sg_vl);

		if (er_f1 > er_f2*1.2) {
			ide[i_id].fcs = f2;
			er_f2 = pre_cal_exp_ide_R_t(f2, ide, bldshps, inner_land_corr,
				slt_line, slt_point_rect, i_id, ide_sg_vl);
		}
		else
			ide[i_id].fcs = f1;
#endif // !test_coef

		/*FILE *fp;
		fopen_s(&fp, "test_f.txt", "w");*/

#ifdef test_coef
		int st = 300, en = 310, step = 25;
		Eigen::VectorXf temp((en - st) / step + 1);

		for (int i = st; i < en; i += step) temp((i - st) / step) =
			pre_cal_exp_ide_R_t(i, ide, bldshps, inner_land_corr,
				slt_line, slt_point_rect, i_id, ide_sg_vl);

		for (int i = 0; i < (en - st) / step + 1; i++) {
			printf("test cal f %d %.6f\n", st + i * step, temp(i));
			fprintf(fp, "%d %.6f\n", st + i * step, temp(i));
		}
#endif

	}
	fclose(fp);
	//FILE *fp;
	fopen_s(&fp, "test_ide_coeff.txt", "w");
	for (int i = 0; i < G_iden_num; i++)
		fprintf(fp, "%.6f\n", ide[0].user(i));
	fclose(fp);
	fopen_s(&fp, "test_exp_coeff.txt", "w");
	for (int i_exp = 0; i_exp < ide[0].num; i_exp++) {
		fprintf(fp, "\n");
		for (int i = 0; i < G_nShape; i++)
			fprintf(fp, "%.6f\n", ide[0].exp(i_exp, i));
	}
	fclose(fp);

}

void solve(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXf &ide_sg_vl) {

	puts("calclating coeffients begin...");
	FILE *fp;
	for (int i_id = 0; i_id < G_train_pic_id_num; i_id++) {
		if (ide[i_id].num == 0)continue;
		pre_cal_exp_ide_R_t(0, ide, bldshps, inner_land_corr,
			slt_line, slt_point_rect, i_id, ide_sg_vl);
	}
	fopen_s(&fp, "test_ide_coeff.txt", "w");
	for (int i = 0; i < G_iden_num; i++)
		fprintf(fp, "%.6f\n", ide[0].user(i));
	fclose(fp);
	fopen_s(&fp, "test_exp_coeff.txt", "w");
	for (int i_exp = 0; i_exp < ide[0].num; i_exp++) {
		fprintf(fp, "------------------------------------------\n");
		for (int i = 0; i < G_nShape; i++)
			fprintf(fp, "%.6f\n", ide[0].exp(i_exp, i));
	}
	fclose(fp);

}





std::string cal_coef_land_name = "test_coef_land_olsgm_25.txt";
std::string cal_coef_mesh_name = "test_coef_mesh_olsgm_25.txt";
std::string cal_eoef_2dland_name = "2dland.txt";

float pre_cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor,
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, int id_idx,
	Eigen::VectorXf &ide_sg_vl) {

	puts("preparing expression & other coeffients...");
	init_exp_ide(ide, id_idx);
	float error = 0;

	int tot_r = 4;
	Eigen::VectorXf temp(tot_r);
	//fprintf(fp, "%d\n",tot_r);
	FILE *fp;
	//fopen_s(&fp, cal_coef_land_name.c_str(), "w");
	//fprintf(fp, "%d\n", 1);
	//fclose(fp);

	//fopen_s(&fp, cal_coef_mesh_name.c_str(), "w");
	//fprintf(fp, "%d\n", 1);
	//fclose(fp);
	//test_coef_land(ide, bldshps, id_idx, 0);
	//test_coef_mesh(ide, bldshps, id_idx, 0);
	//exit(-5);
#ifdef test_coef_save_mesh
	fopen_s(&fp, cal_coef_land_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r + 1)*ide[id_idx].num);
	fclose(fp);

	fopen_s(&fp, cal_coef_mesh_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r + 1)*ide[id_idx].num);
	fclose(fp);
#endif

#ifdef test_posit_by2dland
	fopen_s(&fp, cal_eoef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r * 3);
	fclose(fp);
#endif // test_posit_by2dland

	//float error_last=0;
	for (int rounds = 0; rounds < tot_r; rounds++) {
		///////////////////////////////////////////////paper's solution

		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			printf("calculate %d id %d exp:\n", id_idx, i_exp);
#ifdef test_posit_by2dland
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland 
#ifdef perspective
			//cal_rt_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
			cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // perspective
#ifdef normalization
			cal_rt_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // normalization



#ifdef test_posit_by2dland
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland

			Eigen::VectorXi out_land_cor(G_outer_land_num);
			update_slt_me(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);
			//std::cout << inner_land_cor << '\n';
			//std::cout <<"--------------\n"<< out_land_cor << '\n';
			Eigen::VectorXi land_cor(G_land_num);
			for (int i = 0; i < G_outer_land_num; i++) land_cor(i) = out_land_cor(i);
			for (int i = G_outer_land_num; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - G_outer_land_num);
			ide[id_idx].land_cor.row(i_exp) = land_cor.transpose();

			//test_slt(f, ide, bldshps, land_cor, id_idx, i_exp);

#ifdef test_posit_by2dland
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland			
#ifdef test_coef_save_mesh
			if (rounds == 0) {
				for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
					test_coef_land(ide, bldshps, id_idx, i_exp);
					test_coef_mesh(ide, bldshps, id_idx, i_exp);
				}
			}
#endif
			error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
			error = cal_3dpaper_ide(f, ide, bldshps, id_idx, i_exp, land_cor, ide_sg_vl);
		}
		error = cal_fixed_exp_same_ide(f, ide, bldshps, id_idx, ide_sg_vl);

		printf("+++++++++++++%d %.6f\n", rounds, error);
#ifdef test_coef_save_mesh
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			test_coef_land(ide, bldshps, id_idx, i_exp);
			test_coef_mesh(ide, bldshps, id_idx, i_exp);
		}
#endif
		//if (fabs(error_last - error) < 20) break;
		//error_last = error;
		temp(rounds) = error;
	}
	for (int i = 0; i < tot_r; i++) printf("it %d err %.6f\n", i, temp(i));
	return error;
}

void init_exp_ide(iden *ide, int id_idx) {

	puts("initializing coeffients(identitiy,expression) for cal f...");
	ide[id_idx].exp = Eigen::MatrixXf::Constant(ide[id_idx].num, G_nShape, 1.0 / G_nShape);////////////////////////////0.5
	for (int j = 0; j < ide[id_idx].num; j++) ide[id_idx].exp(j, 0) = 1;
	ide[id_idx].user = Eigen::MatrixXf::Constant(G_iden_num, 1, 1.0 / G_iden_num);

}



void cal_rt_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("POSIT...");

	Eigen::MatrixX2f land_in; Eigen::MatrixX3f bs_in;
	int temp_num = 0;
	if (ide[id_idx].land_cor(exp_idx, 20) == inner_land_cor(20 - G_outer_land_num) && ide[id_idx].land_cor(exp_idx, 30) == inner_land_cor(30 - 15)) {
		land_in.resize(G_land_num, 2);
		land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2);
		//std::cout << land_in.transpose() << '\n';
		bs_in.resize(G_land_num, 3);
		for (int i =
			0; i < G_land_num; i++)
			for (int axis = 0; axis < 3; axis++)
				bs_in(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), axis);


		//std::cout << bs_in << '\n';
		//std::cout << land_in << "\n";
		temp_num = G_land_num;
	}
	else
	{
		land_in.resize(G_inner_land_num, 2);
		land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num + 15, 0, G_inner_land_num, 2);
		//std::cout << land_in.transpose() << '\n';
		bs_in.resize(G_inner_land_num, 3);

		cal_inner_bldshps(ide, bldshps, bs_in, inner_land_cor, id_idx, exp_idx);
		//std::cout << bs_in << '\n';
		//std::cout << land_in << "\n";
		temp_num = G_inner_land_num;
	}
	land_in.rowwise() -= ide[id_idx].center.row(exp_idx);
	Eigen::RowVector2f m0 = land_in.row(0);

	/*std::cout << m0 << '\n';
	std::cout << land_in << '\n';*/
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
	ep_last(1) = 1000;
	Eigen::Vector3f I, J;
	//puts("B");
	//std::cout << B << '\n';
	float s;
	int cnt = 0;
	while ((ep - ep_last).norm() > 0.1) {
		//puts("pp");
		ep_last = ep;
		Eigen::MatrixX2f xy(temp_num, 2);
		xy.col(0) = ((1 + ep.array()).array()*land_in.col(0).array()) - m0(0);
		xy.col(1) = ((1 + ep.array()).array()*land_in.col(1).array()) - m0(1);
		//std::cout << xy << '\n';
		//puts("pp");
		Eigen::MatrixX2f ans = B * xy;
		I = ans.col(0), J = ans.col(1);
		printf("I*J %.6f\n", I.dot(J));
		//std::cout << I << '\n';
		//std::cout << J << '\n';
		float s1 = I.norm(), s2 = J.norm();
		s = (s1 + s2) / 2;
		//puts("pp");
		//std::cout << s  << ' '<< f << '\n';		
		I.normalize(); J.normalize();
		std::cout << " dot:\n" << I.dot(J) << "----------------------------------+++++++++\n";
		J = J - (J.dot(I))*I;
		std::cout << " dot:\n" << I.dot(J) << "----------------------------------+++++++++\n";
		J.normalize();
		/*std::cout << I << '\n';
		std::cout << J << '\n';*/
		ep = s / f * ((bs_in*(I.cross(J))).array());

		Eigen::Matrix3f rot;
		rot.row(0) = I.transpose();
		rot.row(1) = J.transpose();
		rot.row(2) = I.cross(J).transpose();
		rot.row(2).normalize();
		std::cout << cnt << " angle:\n" << get_uler_angle_zyx(rot) * 180 / (acos(-1)) << "\n";
		std::cout << " dot:\n" << I.dot(J) << "----------------------------------+++++++++\n";
		if (cnt++ > 10) break;
		std::cout << "dif:  " << (ep - ep_last).norm() << '\n';
	}
	ide[id_idx].tslt.row(exp_idx) = Eigen::RowVector3f(m0(0), m0(1), f);
	ide[id_idx].tslt.row(exp_idx).array() /= s;
	ide[id_idx].rot.row(3 * exp_idx) = I.transpose();
	ide[id_idx].rot.row(3 * exp_idx + 1) = J.transpose();
	ide[id_idx].rot.row(3 * exp_idx + 2) = (I.cross(J)).transpose();
	ide[id_idx].rot.row(3 * exp_idx + 2).normalize();
	ide[id_idx].tslt.row(exp_idx) -= (ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3)*M0.transpose()).transpose();
	puts("-----------------------------------------");
	std::cout << I << '\n';
	std::cout << J << '\n';
	std::cout << ide[id_idx].rot << '\n';
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
	Eigen::MatrixXf temp = (rot * bs.transpose()).colwise() + tslt;
	temp.row(0).array() /= temp.row(2).array();
	temp.row(1).array() /= temp.row(2).array();
	temp = temp.array()*f;
	temp.block(0, 0, 2, G_inner_land_num).colwise() += ide[id_idx].center.row(exp_idx).transpose();
	std::cout << temp << '\n';
	std::cout << ide[id_idx].land_2d.block(G_land_num*exp_idx + 15, 0, G_land_num - 15, 2) << '\n';
	system("pause");
}

void cal_rt_pnp(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("solve pnp...");

	std::vector<cv::Point2f> land_2d; land_2d.clear();
	std::vector<cv::Point3f> land_3d; land_3d.clear();

	int temp_num = 0;
	if (ide[id_idx].land_cor(exp_idx, 20) == inner_land_cor(20 - G_outer_land_num) &&
		ide[id_idx].land_cor(exp_idx, 30) == inner_land_cor(30 - G_outer_land_num)) {
		/*land_in.resize(G_land_num, 2);
		land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2);*/
		for (int i_v = exp_idx * G_land_num; i_v < exp_idx*G_land_num + G_land_num; i_v++)
			land_2d.push_back(cv::Point2f(ide[id_idx].land_2d(i_v, 0), ide[id_idx].land_2d(i_v, 1)));
		//std::cout << land_in.transpose() << '\n';
		//bs_in.resize(G_land_num, 3);
		land_3d.resize(G_land_num);
		for (int i = 0; i < G_land_num; i++) {
			//			for (int axis = 0; axis < 3; axis++)
			land_3d[i].x = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 0);
			land_3d[i].y = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 1);
			land_3d[i].z = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 2);
		}



		//std::cout << bs_in << '\n';
		//std::cout << land_in << "\n";
		temp_num = G_land_num;
	}
	else
	{
		//land_in.resize(G_inner_land_num, 2);
		//land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num + 15, 0, G_inner_land_num, 2);
		for (int i_v = exp_idx * G_land_num + G_outer_land_num; i_v < exp_idx*G_land_num + G_land_num; i_v++)
			land_2d.push_back(cv::Point2f(ide[id_idx].land_2d(i_v, 0), ide[id_idx].land_2d(i_v, 1)));
		//std::cout << land_in.transpose() << '\n';
		//bs_in.resize(G_inner_land_num, 3);
		land_3d.resize(G_inner_land_num);

		//cal_inner_bldshps(ide, bldshps, bs_in, inner_land_cor, id_idx, exp_idx);

		for (int i = 0; i < G_inner_land_num; i++) {
			//			for (int axis = 0; axis < 3; axis++)
			land_3d[i].x = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 0);
			land_3d[i].y = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 1);
			land_3d[i].z = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 2);
			//land_3d[i] *= 10000;
		}
		//std::cout << bs_in << '\n';
		//std::cout << land_in << "\n";
		temp_num = G_inner_land_num;
	}

	// Camera internals

	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
		f, 0, ide[id_idx].center(exp_idx, 0), 0, f, ide[id_idx].center(exp_idx, 1), 0, 0, 1);
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
		ide[id_idx].tslt(exp_idx, axis) = translation_vector.at<double>(axis);

	std::cout << "rotation_vector:\n" << rotation_vector << "\n\n";

	cv::Mat rot;
	cv::Rodrigues(rotation_vector, rot);
	std::cout << "rotation_cv_matrix:\n" << rot << "\n\n";
	//Eigen::Map<Eigen::Matrix3d> R(rot.ptr <double>(), rot.rows, rot.cols);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			ide[id_idx].rot(3 * exp_idx + i, j) = rot.at<double>(i, j);


	puts("-----------------------------------------");
	std::cout << rot << "\n";
	/*std::cout << R << "\n";
	system("pause");*/
	std::cout << ide[id_idx].rot << '\n';

	//#ifdef revise_rot_tslt
	//	Eigen::Vector3f pt;
	//	pt(0) = land_3d[0].x, pt(1) = land_3d[0].y, pt(2) = land_3d[0].z;
	//	pt = ide[id_idx].rot.block(3 * exp_idx,0,3,3) * pt;
	//	if ((pt(2) + ide[id_idx].tslt(exp_idx, 2)) < 0) {
	//		ide[id_idx].rot.row(3 * exp_idx) *= -1;
	//		ide[id_idx].rot.row(3 * exp_idx + 1) *= -1;
	//		ide[id_idx].tslt(exp_idx, 2) = -(pt(2) + ide[id_idx].tslt(exp_idx, 2)) - pt(2);
	//	}
	//#endif // revise_rot_tslt
}

void test_pnp(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("testing pnp...");
	Eigen::MatrixX3f bs(G_inner_land_num, 3);
	cal_inner_bldshps(ide, bldshps, bs, inner_land_cor, id_idx, exp_idx);
	puts("aa");
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	puts("aabb");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx);
	puts("aacc");
	std::cout << rot << '\n';
	std::cout << "rot*rot\n" << rot * rot.transpose() << '\n';
	std::cout << " angle:\n" << get_uler_angle_zyx(rot) * 180 / (acos(-1)) << "\n";
	std::cout << tslt << '\n';
	puts("-------------before rot-------------");
	std::cout << bs << '\n';
	Eigen::MatrixXf temp = (rot * bs.transpose()).colwise() + tslt;
	puts("-------------before persepctive-------------");
	std::cout << temp << '\n';
	puts("-------------------------------");
	temp.row(0).array() /= temp.row(2).array();
	temp.row(1).array() /= temp.row(2).array();
	temp = temp.array()*f;
	temp.block(0, 0, 2, G_inner_land_num).colwise() += ide[id_idx].center.row(exp_idx).transpose();
	std::cout << temp << '\n';
	std::cout << ide[id_idx].land_2d.block(G_land_num*exp_idx + 15, 0, G_land_num - 15, 2) << '\n';
	system("pause");
}

void cal_rt_normalization(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {
#ifdef normalization


	puts("normalization...");
	Eigen::MatrixX2f land_in; Eigen::MatrixX3f bs_in;
	if (ide[id_idx].land_cor(exp_idx, 20) == inner_land_cor(20 - 15) && ide[id_idx].land_cor(exp_idx, 30) == inner_land_cor(30 - 15)) {
		land_in.resize(G_land_num, 2);
		land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2);
		land_in = land_in.rowwise() + ide[id_idx].center.row(exp_idx);
		//std::cout << land_in.transpose() << '\n';
		bs_in.resize(G_land_num, 3);
		for (int i = 0; i < G_land_num; i++)
			for (int axis = 0; axis < 3; axis++)
				bs_in(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), axis);

		//std::cout << bs_in << '\n';
		//std::cout << land_in << "\n";
	}
	else
	{
		land_in.resize(G_inner_land_num, 2);
		land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num + 15, 0, G_inner_land_num, 2);
		land_in = land_in.rowwise() + ide[id_idx].center.row(exp_idx);
		//std::cout << land_in.transpose() << '\n';
		bs_in.resize(G_inner_land_num, 3);

		cal_inner_bldshps(ide, bldshps, bs_in, inner_land_cor, id_idx, exp_idx);
		//std::cout << bs_in << '\n';
		//std::cout << land_in << "\n";

	}

	Eigen::RowVector3f center_3d = bs_in.colwise().mean();
	//std::cout << "\n----ceneter_3d----\n" << center_3d << "-------------\n";
	bs_in = bs_in.rowwise() - center_3d;
	Eigen::RowVector2f center_2d = land_in.colwise().mean();
	land_in = land_in.rowwise() - center_2d;
	//std::cout << "\n----ceneter_2d----\n" << center_2d << "-------------\n";
	//std::cout << bs_in << '\n';
	//puts("A");
	Eigen::MatrixX3f A = land_in.transpose()*bs_in*((bs_in.transpose()*bs_in).inverse());
	//std::cout << A * ((bs_in.rowwise() + center_3d).transpose()) << "+_+_+_+_+_+_+_+\n";
	//std::cout << "\n----A----\n" << A << "-------------\n";
	//printf("%d %d\n", A.rows(), A.cols());
	//std::cout <<
	//	ide[id_idx].center.row(exp_idx).transpose() << "qqqqqqqqqqq-\n";
	ide[id_idx].tslt.block(exp_idx, 0, 1, 2) = (center_2d - (A*center_3d.transpose()).transpose());
	//puts("A");
	Eigen::RowVector3f I = A.row(0), J = A.row(1);
	Eigen::Matrix3f A_;
	//puts("A");
	A_.row(0) = I, A_.row(1) = J, A_.row(2) = I.cross(J);
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A_, Eigen::ComputeFullV | Eigen::ComputeFullU);
	Eigen::Matrix3f R = svd.matrixU()*(svd.matrixV().transpose());
	//puts("A");
	ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = R;
	ide[id_idx].s.block(2 * exp_idx, 0, 2, 3) = A * R.inverse();
	ide[id_idx].s(2 * exp_idx, 1) = ide[id_idx].s(2 * exp_idx, 2) =
		ide[id_idx].s(2 * exp_idx + 1, 0) = ide[id_idx].s(2 * exp_idx + 1, 2) = 0;

	//ide[id_idx].s(exp_idx, 0) = svd.singularValues()(1), ide[id_idx].s(exp_idx, 1) = svd.singularValues()(2);
#endif // normalization
}
void test_normalization(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {
#ifdef normalization


	puts("testing normalization...");
	Eigen::MatrixX3f bs(G_inner_land_num, 3);
	cal_inner_bldshps(ide, bldshps, bs, inner_land_cor, id_idx, exp_idx);
	puts("aa");
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	puts("aabb");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx);
	puts("aacc");
	std::cout << "\n----rotation----\n" << rot << "-------------\n";
	std::cout << "\n----translation----\n" << tslt << "-------------\n";
	std::cout << "\n----scale----\n" << ide[id_idx].s.block(2 * exp_idx, 0, 2, 3) << "-------------\n";
	Eigen::MatrixXf temp = (rot * bs.transpose());
	temp.block(0, 0, 2, G_inner_land_num) = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)*temp;
	/*temp.row(0).array() *= ide[id_idx].s(exp_idx, 0);
	temp.row(1).array() *= ide[id_idx].s(exp_idx, 1);*/
	temp = temp.colwise() + tslt;
	std::cout << temp.block(0, 0, 2, G_inner_land_num) << "+++++++\n";
	std::cout <<
		ide[id_idx].land_2d.block(15 + G_land_num * exp_idx, 0, G_inner_land_num, 2).transpose().colwise()
		+ ide[id_idx].center.row(exp_idx).transpose() << "------\n";
	std::cout <<
		ide[id_idx].center.row(exp_idx).transpose() << "pppppppppppp-\n";
	system("pause");
#endif // normalization
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
			if (i_shape == 0)
				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
				*bldshps(i_id, vtx_idx * 3 + axis);
			else
				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
				*(bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis) - bldshps(i_id, vtx_idx * 3 + axis));
	return ans;
}




void update_slt(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &out_land_cor) {
	////////////////////////////////project
	puts("updating silhouette...");
	Eigen::Matrix3f R = ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3);

	Eigen::VectorXf angle = get_uler_angle_zyx(R);
	Eigen::Vector3f T = ide[id_idx].tslt.row(exp_idx).transpose();

	//puts("A");
	Eigen::VectorXi slt_cddt(G_line_num);
	Eigen::MatrixX3f slt_cddt_cdnt(G_line_num, 3);
	//puts("B");
	//FILE *fp;
	//fopen_s(&fp, "test_slt.txt", "w");
	for (int i = 0; i < G_line_num; i++) {
		//printf("i %d\n", i);
		float min_v_n = 10000;
		int min_idx = 0;
		Eigen::Vector3f cdnt;
		int en = slt_line[i].size();
		if (angle(1) < -0.1 && i < 34) en /= 3;
		if (angle(1) < -0.1 && i >= 34 && i < 41) en /= 2;
		if (angle(1) > 0.1 && i >= 49 && i < 84) en /= 3;
		if (angle(1) > 0.1 && i >= 42 && i < 49) en /= 2;
#ifdef perspective

		for (int j = 0; j < en; j++) {
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
			//printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
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
			//std::cout << "nor++\n\n" << nor << "\n";
			//std::cout << "point--\n\n" << point[0].normalized() << "\n";
			//std::cout << "rltv--\n\n"<<x << ' ' << nor.dot(point[0].normalized()) << "\n";
			if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0];// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));


			/*point[0].normalize();
			if (fabs(point[0](2)) < min_v_n) min_v_n = fabs(point[0](2)), min_idx = x, cdnt = point[0];*/
		}
		//puts("H");
		//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));
		slt_cddt(i) = min_idx;
		cdnt(0) = cdnt(0)*f / cdnt(2) + ide[id_idx].center(exp_idx, 0);
		cdnt(1) = cdnt(1)*f / cdnt(2) + ide[id_idx].center(exp_idx, 1);
		slt_cddt_cdnt.row(i) = cdnt.transpose();
#endif // posit

#ifdef normalization
		for (int j = 0; j < en; j++) {

			//printf("j %d\n", j);
			int x = slt_line[i][j];
			//printf("x %d\n", x);
			//printf("x %d\n", x);
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++)
				point[0](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
			point[0] = R * point[0];
			//test															//////////////////////////////////debug
			//puts("A");
			//fprintf(fp, "%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			//printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
			//puts("B");


			////////////////////////////////////////////////////////////////////////////////////////////////////////

			for (int k = 0, sz = slt_point_rect[x].size(); k < sz; k++) {
				//printf("k %d\n", k);
				for (int axis = 0; axis < 3; axis++) {
					point[1](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].first, axis);
					point[2](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].second, axis);
				}
				for (int i = 1; i < 3; i++) point[i] = R * point[i];
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
			nor.normalize();
			//std::cout << "nor++\n\n" << nor << "\n";
			//std::cout << "point--\n\n" << point[0].normalized() << "\n";
			//std::cout << "rltv--\n\n"<<x << ' ' << nor.dot(point[0].normalized()) << "\n";
			if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0];// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));


			/*point[0].normalize();
			if (fabs(point[0](2)) < min_v_n) min_v_n = fabs(point[0](2)), min_idx = x, cdnt = point[0];*/
		}
		//puts("H");
		//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));
		slt_cddt(i) = min_idx;
		cdnt.block(0, 0, 2, 1) = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)*cdnt + T.block(0, 0, 2, 1);

		slt_cddt_cdnt.row(i) = cdnt.transpose();
		//for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
		//	//printf("j %d\n", j);
		//	int x = slt_line[i][j];
		//	//printf("x %d\n", x);
		//	Eigen::Vector3f point,temp;
		//	for (int axis = 0; axis < 3; axis++)
		//		point(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
		//	point = R * point;
		//	temp = point;
		//	point.normalize();
		//	if (fabs(point(2)) < min_v_n) min_v_n = fabs(point(2)), min_idx = x, cdnt = temp;// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		//}
		//slt_cddt(i) = min_idx;
		//cdnt.block(0, 0, 2, 1) = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)*cdnt+ T.block(0,0,2,1);
		///*cdnt(0) = cdnt(0)*ide[id_idx].s(exp_idx, 0) + T(0);
		//cdnt(1) = cdnt(1)*ide[id_idx].s(exp_idx, 1) + T(1);*/
		//slt_cddt_cdnt.row(i) = cdnt.transpose();
#endif

	}
#ifdef test_updt_slt
	FILE *fp;
	fopen_s(&fp, "test_updt_slt.txt", "a");
	fprintf(fp, "%.5f %.5f %.5f  ", angle(0) * 180 / acos(-1), angle(1) * 180 / acos(-1), angle(2) * 180 / acos(-1));
	for (int j = 0; j < G_line_num; j++)
		fprintf(fp, " %d", slt_cddt(j));
	fprintf(fp, "\n");
	fclose(fp);
	fopen_s(&fp, "test_updt_slt_2d_point.txt", "a");
	for (int j = 0; j < G_line_num; j++)
		fprintf(fp, "%.5f %.5f\n", slt_cddt_cdnt(j, 0), slt_cddt_cdnt(j, 1));
	fprintf(fp, "\n");
	fclose(fp);


#endif // test_updt_slt
	//fclose(fp);
	//puts("C");
//	for (int i_jaw = 0; i_jaw < G_jaw_land_num; i_jaw++) {
//		Eigen::Vector3f point;
//		for (int axis = 0; axis < 3; axis++)
//			point(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, jaw_land_corr(i_jaw), axis);
//#ifdef perspective
//		point = R * point + T;
//		point(0) = point(0)*f / point(2) + ide[id_idx].center(exp_idx, 0);
//		point(1) = point(1)*f / point(2) + ide[id_idx].center(exp_idx, 0);
//#endif // posit
//#ifdef normalization
//		point.block(0,0,2,1) = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)*R * point;
//		//point(0) *= ide[id_idx].s(exp_idx, 0), point(1) *= ide[id_idx].s(exp_idx, 1);
//		point = point + T;
//#endif // normalization		
//		slt_cddt(i_jaw + G_line_num) = jaw_land_corr(i_jaw);
//		slt_cddt_cdnt.row(i_jaw + G_line_num) = point.transpose();
//	}
	for (int i = 0; i < 15; i++) {
		float min_dis = 10000;
		int min_idx = 0;
		int be = 0, en = G_land_num;
		if (i < 7) be = 41, en = 84;
		if (i > 7) be = 0, en = 42;
		for (int j = be; j < en; j++) {
#ifdef perspective
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 1));
#endif // posit
#ifdef normalization
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 0) - ide[id_idx].center(exp_idx, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 1) - ide[id_idx].center(exp_idx, 1));
#endif // normalization


			if (temp < min_dis) min_dis = temp, min_idx = j;
		}
		//printf("%d %d %d\n", i, min_idx, slt_cddt(min_idx));
		out_land_cor(i) = slt_cddt(min_idx);

	}
	std::cout << "slt_cddt_cdnt\n" << slt_cddt_cdnt.block(0, 0, slt_cddt_cdnt.rows(), 2).rowwise() + ide[id_idx].center.row(exp_idx) << "\n";
	std::cout << "out land\n" << ide[id_idx].land_2d.block(G_land_num*exp_idx, 0, 15, 2).rowwise() + ide[id_idx].center.row(exp_idx) << "\n";
	std::cout << "out land correlation\n" << out_land_cor.transpose() << "\n";
	system("pause");
}

void update_slt_me(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &out_land_cor) {
	////////////////////////////////project
	puts("updating silhouette...");
	Eigen::Matrix3f R = ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3);
	Eigen::VectorXf angle = get_uler_angle_zyx(R);
	Eigen::Vector3f T = ide[id_idx].tslt.row(exp_idx).transpose();

	//puts("A");

	Eigen::VectorXf land_cor_mi(15);
	for (int i = 0; i < 15; i++) land_cor_mi(i) = 1e8;

	//puts("B");
	//FILE *fp;
	//fopen_s(&fp, "test_slt.txt", "w");
	if (fabs(angle(2)) < 0.2) {
		/*std::vector<cv::Point2f> test_slt_2dpt;
		test_slt_2dpt.clear();*/
		for (int i_line = 0; i_line < G_line_num; i_line++) {
#ifdef perspective
			for (int j = 0; j < slt_line[i_line].size(); j++) {
				//printf("j %d\n", j);
				int x = slt_line[i_line][j];

				Eigen::Vector3f point;
				for (int axis = 0; axis < 3; axis++)
					point(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
				point = R * point + T;
				point(0) = point(0)*f / point(2) + ide[id_idx].center(exp_idx, 0);
				point(1) = point(1)*f / point(2) + ide[id_idx].center(exp_idx, 1);
				//test_slt_2dpt.push_back(cv::Point2f(point(0), point(1)));
				for (int p = 0; p < 15; p++) {
					float temp = (point.block(0, 0, 2, 1).transpose() - ide[id_idx].land_2d.row(G_land_num*exp_idx + p)).squaredNorm();
					if (temp < land_cor_mi(p)) {
						land_cor_mi(p) = temp;
						out_land_cor(p) = x;
					}
				}
			}

#endif // perspective

		}
		//		for (int i_line = 49; i_line < G_line_num; i_line++) {
		//#ifdef perspective
		//			for (int j = 0; j < slt_line[i_line].size(); j++) {
		//				//printf("j %d\n", j);
		//				int x = slt_line[i_line][j];
		//
		//				Eigen::Vector3f point;
		//				for (int axis = 0; axis < 3; axis++)
		//					point(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
		//				point = R * point + T;
		//				point(0) = point(0)*f / point(2) + ide[id_idx].center(exp_idx, 0);
		//				point(1) = point(1)*f / point(2) + ide[id_idx].center(exp_idx, 1);
		//				//test_slt_2dpt.push_back(cv::Point2f(point(0), point(1)));
		//				for (int p = 0; p < 8; p++) {
		//					float temp = (point.block(0, 0, 2, 1).transpose() - ide[id_idx].land_2d.row(G_land_num*exp_idx + p)).squaredNorm();
		//					if (temp < land_cor_mi(p)) {
		//						land_cor_mi(p) = temp;
		//						out_land_cor(p) = x;
		//					}
		//				}
		//			}
		//
		//#endif // perspective
		//
		//		}
		////#ifdef test_updt_slt
		////		FILE *fp;
		////		fopen_s(&fp, "test_updt_slt_me_2d_point.txt", "w");
		////		fprintf(fp, "%d\n", test_slt_2dpt.size());
		////		for (int t = 0; t < test_slt_2dpt.size(); t++)
		////			fprintf(fp, "%.5f %.5f\n", test_slt_2dpt[t].x, test_slt_2dpt[t].y);
		////		fprintf(fp, "\n");
		////		fclose(fp);
		////#endif // test_updt_slt
		//		for (int i_line = 34; i_line < 49; i_line++) {
		//			float min_v_n = 10000;
		//			int min_idx = 0;
		//			Eigen::Vector3f cdnt;
		//			int en = slt_line[i_line].size(), be = 0;
		//			if (angle(1) < 0 && i_line < 41) en /= 2;
		//			if (angle(1) > 0 && i_line >= 41) en /= 2;
		//#ifdef perspective
		//			for (int j = be; j < en; j++) {
		//				//printf("j %d\n", j);
		//				int x = slt_line[i_line][j];
		//				//printf("x %d\n", x);
		//				Eigen::Vector3f nor;
		//				nor.setZero();
		//				Eigen::Vector3f V[2], point[3];
		//				for (int axis = 0; axis < 3; axis++)
		//					point[0](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
		//				point[0] = R * point[0] + T;
		//				//test															//////////////////////////////////debug
		//				//puts("A");
		//				//fprintf(fp, "%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		//				//printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		//				//puts("B");
		//
		//
		//				////////////////////////////////////////////////////////////////////////////////////////////////////////
		//
		//				for (int k = 0, sz = slt_point_rect[x].size(); k < sz; k++) {
		//					//printf("k %d\n", k);
		//					for (int axis = 0; axis < 3; axis++) {
		//						point[1](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].first, axis);
		//						point[2](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].second, axis);
		//					}
		//					for (int i = 1; i < 3; i++) point[i] = R * point[i] + T;
		//					V[0] = point[1] - point[0];
		//					V[1] = point[2] - point[0];
		//					//puts("C");
		//					V[0] = V[0].cross(V[1]);
		//					//puts("D");
		//					V[0].normalize();
		//					nor = nor + V[0];
		//					//printf("__ %.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		//				}
		//				//printf("== %.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		//				//puts("F");
		//				if (nor.norm() > EPSILON) nor.normalize();
		//				//std::cout << "nor++\n\n" << nor << "\n";
		//				//std::cout << "point--\n\n" << point[0].normalized() << "\n";
		//				//std::cout << "rltv--\n\n"<<x << ' ' << nor.dot(point[0].normalized()) << "\n";
		//				if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0];// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
		//
		//
		//				/*point[0].normalize();
		//				if (fabs(point[0](2)) < min_v_n) min_v_n = fabs(point[0](2)), min_idx = x, cdnt = point[0];*/
		//			}
		//			//puts("H");
		//			//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));
		//
		//			cdnt(0) = cdnt(0)*f / cdnt(2) + ide[id_idx].center(exp_idx, 0);
		//			cdnt(1) = cdnt(1)*f / cdnt(2) + ide[id_idx].center(exp_idx, 1);
		//			for (int p = 0; p < 12; p++) {
		//				float temp = (cdnt.block(0, 0, 2, 1).transpose() - ide[id_idx].land_2d.row(G_land_num*exp_idx + p)).squaredNorm();
		//
		//				if (temp < land_cor_mi(p)) {
		//					land_cor_mi(p) = temp;
		//					out_land_cor(p) = min_idx;
		//				}
		//			}
		//#endif // perspective
		//
		//		}

		std::cout << "land_cor_mi:\n" << land_cor_mi.transpose() << "\n";

		std::cout << "out land correlation\n" << out_land_cor.transpose() << "\n";
		return;
	}


	if (angle(2) < 0) {
		std::vector<cv::Point2f> test_slt_2dpt;
		test_slt_2dpt.clear();
		for (int i_line = 0; i_line < 34; i_line++) {
#ifdef perspective
			for (int j = 0; j < slt_line[i_line].size(); j++) {
				//printf("j %d\n", j);
				int x = slt_line[i_line][j];

				Eigen::Vector3f point;
				for (int axis = 0; axis < 3; axis++)
					point(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
				point = R * point + T;
				point(0) = point(0)*f / point(2) + ide[id_idx].center(exp_idx, 0);
				point(1) = point(1)*f / point(2) + ide[id_idx].center(exp_idx, 1);
				test_slt_2dpt.push_back(cv::Point2f(point(0), point(1)));
				for (int p = 8; p < 15; p++) {
					float temp = (point.block(0, 0, 2, 1).transpose() - ide[id_idx].land_2d.row(G_land_num*exp_idx + p)).squaredNorm();
					if (temp < land_cor_mi(p)) {
						land_cor_mi(p) = temp;
						out_land_cor(p) = x;
					}
				}
			}

#endif // perspective

		}
#ifdef test_updt_slt
		FILE *fp;
		fopen_s(&fp, "test_updt_slt_me_2d_point.txt", "w");
		fprintf(fp, "%d\n", test_slt_2dpt.size());
		for (int t = 0; t < test_slt_2dpt.size(); t++)
			fprintf(fp, "%.5f %.5f\n", test_slt_2dpt[t].x, test_slt_2dpt[t].y);
		fprintf(fp, "\n");
		fclose(fp);
#endif // test_updt_slt
		for (int i_line = 34; i_line < G_line_num; i_line++) {
			float min_v_n = 10000;
			int min_idx = 0;
			Eigen::Vector3f cdnt;
			int en = slt_line[i_line].size(), be = 0;
			if (angle(1) < -0.1 && i_line < 41) en /= 2;
#ifdef perspective
			for (int j = be; j < en; j++) {
				//printf("j %d\n", j);
				int x = slt_line[i_line][j];
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
				//printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
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
				//std::cout << "nor++\n\n" << nor << "\n";
				//std::cout << "point--\n\n" << point[0].normalized() << "\n";
				//std::cout << "rltv--\n\n"<<x << ' ' << nor.dot(point[0].normalized()) << "\n";
				if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0];// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));


				/*point[0].normalize();
				if (fabs(point[0](2)) < min_v_n) min_v_n = fabs(point[0](2)), min_idx = x, cdnt = point[0];*/
			}
			//puts("H");
			//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));

			cdnt(0) = cdnt(0)*f / cdnt(2) + ide[id_idx].center(exp_idx, 0);
			cdnt(1) = cdnt(1)*f / cdnt(2) + ide[id_idx].center(exp_idx, 1);
			for (int p = 0; p < 12; p++) {
				float temp = (cdnt.block(0, 0, 2, 1).transpose() - ide[id_idx].land_2d.row(G_land_num*exp_idx + p)).squaredNorm();

				if (temp < land_cor_mi(p)) {
					land_cor_mi(p) = temp;
					out_land_cor(p) = min_idx;
				}
			}
#endif // perspective

		}

	}
	else {
		for (int i_line = 49; i_line < G_line_num; i_line++) {
#ifdef perspective
			for (int j = 0; j < slt_line[i_line].size(); j++) {
				//printf("j %d\n", j);
				int x = slt_line[i_line][j];

				Eigen::Vector3f point;
				for (int axis = 0; axis < 3; axis++)
					point(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
				point = R * point + T;
				point(0) = point(0)*f / point(2) + ide[id_idx].center(exp_idx, 0);
				point(1) = point(1)*f / point(2) + ide[id_idx].center(exp_idx, 1);
				for (int p = 0; p < 7; p++) {
					float temp = (point.block(0, 0, 2, 1).transpose() - ide[id_idx].land_2d.row(G_land_num*exp_idx + p)).squaredNorm();


					if (temp < land_cor_mi(p)) {
						land_cor_mi(p) = temp;
						out_land_cor(p) = x;
					}
				}
			}

#endif // perspective

		}
		for (int i_line = 0; i_line < 49; i_line++) {
			float min_v_n = 10000;
			int min_idx = 0;
			Eigen::Vector3f cdnt;
			int en = slt_line[i_line].size(), be = 0;
			if (angle(1) > 0.1 && i_line >= 42) en /= 2;
#ifdef perspective
			for (int j = be; j < en; j++) {
				//printf("j %d\n", j);
				int x = slt_line[i_line][j];
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
				//printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
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
				//std::cout << "nor++\n\n" << nor << "\n";
				//std::cout << "point--\n\n" << point[0].normalized() << "\n";
				//std::cout << "rltv--\n\n"<<x << ' ' << nor.dot(point[0].normalized()) << "\n";
				if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0];// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));


				/*point[0].normalize();
				if (fabs(point[0](2)) < min_v_n) min_v_n = fabs(point[0](2)), min_idx = x, cdnt = point[0];*/
			}
			//puts("H");
			//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));

			cdnt(0) = cdnt(0)*f / cdnt(2) + ide[id_idx].center(exp_idx, 0);
			cdnt(1) = cdnt(1)*f / cdnt(2) + ide[id_idx].center(exp_idx, 1);
			for (int p = 4; p < 15; p++) {
				float temp = (cdnt.block(0, 0, 2, 1).transpose() - ide[id_idx].land_2d.row(G_land_num*exp_idx + p)).squaredNorm();
				if (temp < land_cor_mi(p)) {
					land_cor_mi(p) = temp;
					out_land_cor(p) = min_idx;
				}
			}
#endif // perspective

		}
	}

	std::cout << "land_cor_mi:\n" << land_cor_mi.transpose() << "\n";

	std::cout << "out land correlation\n" << out_land_cor.transpose() << "\n";
	system("pause");
}

void test_slt(float f, iden *ide, Eigen::MatrixXf &bldshps,
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
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
	puts("aacc");
	std::cout << bs << '\n';
	//std::cout << tslt << '\n';
#ifdef perspective
	Eigen::MatrixXf temp = (rot * bs.transpose()).colwise() + tslt;
	temp.row(0).array() /= temp.row(2).array();
	temp.row(1).array() /= temp.row(2).array();
	temp = temp.array()*f;
	temp.block(0, 0, 2, G_land_num).colwise() += ide[id_idx].center.row(exp_idx).transpose();
#endif // perspective

#ifdef normalization
	Eigen::MatrixXf temp = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)* rot * bs.transpose();
	/*temp.row(0).array() *= ide[id_idx].s(exp_idx, 0);
	temp.row(1).array() *= ide[id_idx].s(exp_idx, 1);*/
	temp = temp.colwise() + tslt.block(0, 0, 2, 1);
#endif // normalization


	std::cout << temp.transpose() << '\n';
	std::cout << ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2) << '\n';
	system("pause");
}

float cal_3dpaper_exp(
	float f, iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor) {

	puts("calculating expression coeffients by 3dpaper's way");
	float error = 0;
	Eigen::MatrixXf exp_point(G_nShape, 3 * G_land_num);

	cal_exp_point_matrix(ide, bldshps, id_idx, exp_idx, land_cor, exp_point);
	Eigen::RowVectorXf exp = ide[id_idx].exp.row(exp_idx);
	error = ceres_exp_one(f, ide, id_idx, exp_idx, exp_point, exp);
	ide[id_idx].exp.row(exp_idx) = exp;
	return error;
}
void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result) {

	puts("prepare exp_point matrix for bfgs/ceres...");
	result.resize(G_nShape, 3 * G_land_num);
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
#ifdef normalization
	Eigen::MatrixX3f S = ide[id_idx].s.block(exp_idx * 2, 0, 2, 3);
#endif // normalization

	for (int i_shape = 0; i_shape < G_nShape; i_shape++)
		for (int i_v = 0; i_v < G_land_num; i_v++) {
			Eigen::Vector3f V;
			V.setZero();
			for (int j = 0; j < 3; j++)
				for (int i_id = 0; i_id < G_iden_num; i_id++)
					if (i_shape == 0)
						V(j) += ide[id_idx].user(i_id)*bldshps(i_id, land_cor(i_v) * 3 + j);
					else
						V(j) += ide[id_idx].user(i_id)*
						(bldshps(i_id, i_shape*G_nVerts * 3 + land_cor(i_v) * 3 + j) - bldshps(i_id, land_cor(i_v) * 3 + j));
			V = rot * V;
#ifdef normalization

			V.block(0, 0, 2, 1) = S * V;
#endif // normalization

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
#ifdef normalization
	Eigen::MatrixX3f S = ide[id_idx].s.block(exp_idx * 2, 0, 2, 3);
#endif // normalization
	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_v = 0; i_v < G_land_num; i_v++) {
			Eigen::Vector3f V;
			V.setZero();
			for (int j = 0; j < 3; j++)
				for (int i_shape = 0; i_shape < G_nShape; i_shape++)
					if (i_shape == 0)
						V(j) += ide[id_idx].exp(exp_idx, i_shape)*bldshps(i_id, land_cor(i_v) * 3 + j);
					else
						V(j) += ide[id_idx].exp(exp_idx, i_shape)
						*(bldshps(i_id, i_shape*G_nVerts * 3 + land_cor(i_v) * 3 + j) - bldshps(i_id, land_cor(i_v) * 3 + j));
			V = rot * V;
#ifdef normalization
			V.block(0, 0, 2, 1) = S * V;
#endif // normalization
			for (int j = 0; j < 3; j++)
				result(i_id, i_v * 3 + j) = V(j);
		}

}


float cal_fixed_exp_same_ide(float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx,
	Eigen::VectorXf &ide_sg_vl) {

	puts("calculating identity coeffients by 3dpaper's way while fixing the expression coeffients");
	float error = 0;
	Eigen::MatrixXf id_point_fix_exp(ide[id_idx].num*G_iden_num, G_land_num * 3);

	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		Eigen::VectorXi land_cor(G_land_num);
		Eigen::MatrixXf id_point(G_iden_num, 3 * G_land_num);
		land_cor = ide[id_idx].land_cor.row(i_exp);
		cal_id_point_matrix(ide, bldshps, id_idx, i_exp, land_cor, id_point);

		id_point_fix_exp.block(i_exp*G_iden_num, 0, G_iden_num, G_land_num * 3) = id_point;
	}
	Eigen::VectorXf user = ide[id_idx].user;
	error = ceres_user_fixed_exp(f, ide, id_idx, id_point_fix_exp, user, ide_sg_vl);
	ide[id_idx].user = user;
	return error;
}

void test_coef_land(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_land_num, 3);
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
#ifdef solve_cvpnp
	Eigen::Vector3f hm_angle = pnpR2humanA(R);
	R = get_r_from_angle_zyx(hm_angle);
#endif // solve_cvpnp
	for (int i = 0; i < G_land_num; i++) {
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), axis);
		bs.row(i) = (R*(bs.row(i).transpose())).transpose();
	}

	FILE *fp;
	fopen_s(&fp, cal_coef_land_name.c_str(), "a");
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);
}

void test_coef_mesh(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_nVerts, 3);
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
#ifdef solve_cvpnp
	Eigen::Vector3f hm_angle = pnpR2humanA(R);
	R = get_r_from_angle_zyx(hm_angle);
#endif // solve_cvpnp
	std::cout << R << "\n";
	//system("pause");
	for (int i = 0; i < G_nVerts; i++) {
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, i, axis);
		bs.row(i) = (R*(bs.row(i).transpose())).transpose();
	}

	FILE *fp;
	fopen_s(&fp, cal_coef_mesh_name.c_str(), "a");
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);
}

void test_2dland(float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {
	Eigen::MatrixX3f land3d(G_land_num, 3);
	puts("A");
	for (int i = 0; i < G_land_num; i++)
		for (int axis = 0; axis < 3; axis++)
			land3d(i, axis) =
			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), axis);
	puts("B");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx, 0, 3, 3);

	puts("C");
	FILE *fp;
	fopen_s(&fp, cal_eoef_2dland_name.c_str(), "a");
	for (int i = 0; i < G_land_num; i++) {
		Eigen::Vector3f X = land3d.row(i).transpose();
#ifdef perspective
		X = R * X + tslt;
		fprintf(fp, "%.6f %.6f\n", X(0)*f / X(2) + ide[id_idx].center(exp_idx, 0), X(1)*f / X(2) + ide[id_idx].center(exp_idx, 1));
#endif // perspective
#ifdef normalization
		X.block(0, 0, 2, 1) = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)*R * X + tslt.block(0, 0, 2, 1);
		fprintf(fp, "%.6f %.6f\n", X(0), X(1));
#endif // normalization
	}

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
	int num = 3;
	fprintf(fpw, "%d\n", num);
	Eigen::MatrixXf mesh(G_nVerts, 3);
	//char s[500];
	for (int j_no = 0; j_no < num; j_no++) {
		printf("%d \n", j_no);
		//fscanf_s(fpr, "------------------------------------------");
		//puts(s);
		for (int i_exp = 0; i_exp < G_nShape; i_exp++)
			fscanf_s(fpr, " %f", &exp(i_exp)), printf("%d %.6f\n", i_exp, exp(i_exp));
		std::cout << exp << "-----------------\n";
		system("pause");
		for (int i = 0; i < G_nVerts; i++)
			for (int axis = 0; axis < 3; axis++) {
				mesh(i, axis) = 0;
				for (int i_id = 0; i_id < G_iden_num; i_id++)
					for (int i_exp = 0; i_exp < G_nShape; i_exp++)
						if (i_exp == 0)
							mesh(i, axis) += bldshps(i_id, i * 3 + axis)*ide(i_id)*exp(i_exp);
						else
							mesh(i, axis) += (bldshps(i_id, i_exp*G_nVerts * 3 + i * 3 + axis) - bldshps(i_id, i * 3 + axis))
							*ide(i_id)*exp(i_exp);
			}
		for (int i = 0; i < G_nVerts; i++)
			fprintf(fpw, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));
	}
	fclose(fpw);
}

void cal_mesh_land_exp_only(Eigen::MatrixXf &bldshps) {
	puts("calculating mesh from test_coeff_exp");
	FILE *fpr, *fpw;
	Eigen::VectorXf exp(G_nShape);
	fopen_s(&fpr, "./server/test_exp_coeff_t23.txt", "r");
	fopen_s(&fpw, "./test_exp_coeff_mesh.txt", "w");
	int num = 1;
	fprintf(fpw, "%d\n", num);
	Eigen::MatrixXf mesh(G_nVerts, 3);
	//char s[500];
	for (int j_no = 0; j_no < num; j_no++) {
		printf("%d \n", j_no);
		//fscanf_s(fpr, "------------------------------------------");
		//puts(s);
		for (int i_exp = 0; i_exp < G_nShape; i_exp++)
			fscanf_s(fpr, "%f", &exp(i_exp));
		for (int i = 0; i < G_nVerts; i++)
			for (int axis = 0; axis < 3; axis++) {
				mesh(i, axis) = 0;
				int i_id = 0;
				for (int i_exp = 0; i_exp < G_nShape; i_exp++)
					if (i_exp == 0)
						mesh(i, axis) += bldshps(i_id, i * 3 + axis)*exp(i_exp);
					else
						mesh(i, axis) +=
						(bldshps(i_id, i_exp*G_nVerts * 3 + i * 3 + axis) - bldshps(i_id, i * 3 + axis))*exp(i_exp);
			}
		for (int i = 0; i < G_nVerts; i++)
			fprintf(fpw, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));
	}
	fclose(fpw);
}


Eigen::Vector3f pnpR2humanA(Eigen::Matrix3f R) {
	Eigen::Vector3f angle = get_uler_angle_zyx(R);
	//(pi-angle)-pi   pi-angle~pi/2--pi*3/2
	Eigen::Vector3f ans(-angle(0), -angle(1), angle(2));
	return ans;
	//system("pause");
}