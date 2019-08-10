#include "calculate_coeff.h"
#define test_coef
#define test_coef_save_mesh
//#define test_inner_land
//#define upt_inner_cor

#define px_err_def
//#define pca_ide
//#define test_ide_cf_def

#define test_posit_by2dland
//#define revise_rot_tslt
#define test_updt_slt

using std::min;
using std::max;

std::string test_updt_slt_path = "test_updt_slt_dde.txt";
std::string test_updt_slt_2d_path = "test_updt_slt_dde_2d_point.txt";


void init_exp_ide_r_t_pq(iden *ide, int ide_num) {

	puts("initializing coeffients(R,t,pq)...");
	for (int i = 0; i < ide_num; i++) {
		
		ide[i].exp.resize(ide[i].num, G_nShape);
		ide[i].land_cor.resize(ide[i].num, G_land_num);
		ide[i].land_cor.setZero();
		ide[i].user.resize(G_iden_num);
		ide[i].rot.resize(3 * ide[i].num,3);
		//ide[i].rot.setZero();
		//ide[i].rot(0, 0) = ide[i].rot(1, 1) = ide[i].rot(2, 2) = 1;
		ide[i].tslt.resize(ide[i].num,3);
		ide[i].tslt.setZero();
		ide[i].dis.resize(G_land_num*ide[i].num, 2);

		ide[i].center.resize(ide[i].num, 2);
		ide[i].center.setZero();
		
#ifdef normalization
		ide[i].s.resize(ide[i].num*2, 3);
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

void load_bldshps(Eigen::MatrixXf &bldshps, std::string &name,Eigen::VectorXf &ide_sg_vl,std::string sg_vl_path) {

	puts("loading blendshapes...");
	std::cout << name << std::endl;
	FILE *fp;
	fp=fopen( name.c_str(), "rb");
	for (int i = 0; i < G_iden_num; i++) {
		for (int j = 0; j < G_nShape*G_nVerts * 3; j++) 
			fread(&bldshps(i, j), sizeof(float), 1, fp);
	}
	fclose(fp);
	fp=fopen( sg_vl_path.c_str(), "r");
	for (int i = 0; i < G_iden_num; i++) {
		fscanf(fp, "%f",&ide_sg_vl(i));
	}
	fclose(fp);
}

void print_bldshps(Eigen::MatrixXf &bldshps) {
	puts("print blendshapes!");
	FILE *fp;
	//fp=fopen( "test_svd_bldshps_test.txt", "w");
	//fprintf(fp, "%d\n", 10);
	//int ide = 1;
	//for (int i_exp = 0; i_exp< 10; i_exp++) {
	//	for (int i_v = 0; i_v < G_nVerts; i_v++)
	//		fprintf(fp, "%.6f %.6f %.6f\n", 
	//			bldshps(ide, i_exp*G_nVerts*3+ i_v*3), bldshps(ide, i_exp*G_nVerts * 3 + i_v * 3+1), bldshps(ide, i_exp*G_nVerts * 3 + i_v * 3+2));
	//}
	//fclose(fp);
	fp=fopen( "test_svd_bldshps_test.txt", "w");
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
	iden *ide, Eigen::MatrixXf &bldshps,Eigen::VectorXi &inner_land_corr,
	std :: vector<int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect, 
	Eigen::VectorXf &ide_sg_vl) {

	puts("calclating focus for each image...");
	FILE *fp;
#ifdef test_coef
	fp=fopen( "test_coef_f_loss.txt", "w");
#else
	fp=fopen( "test_coef_ide_focus.txt", "w");	
#endif // !test_coef


	
	for (int i_id = 0; i_id < G_train_pic_id_num; i_id++) {
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
		//float f2 = 2500, er_f2;
		//er_f2 = pre_cal_exp_ide_R_t(f2, ide, bldshps, inner_land_corr,
		//	slt_line, slt_point_rect, i_id, ide_sg_vl);
		//float f1 = 800, er_f1;
		//er_f1 = pre_cal_exp_ide_R_t(f1, ide, bldshps, inner_land_corr,
		//	slt_line, slt_point_rect, i_id, ide_sg_vl);

		//if (er_f1 > er_f2*1.2) {
		//	ide[i_id].fcs = f2;
		//	er_f2 = pre_cal_exp_ide_R_t(f2, ide, bldshps, inner_land_corr,
		//		slt_line, slt_point_rect, i_id, ide_sg_vl);
		//}
		//else
		//	ide[i_id].fcs = f1;
		//float mi_er = 1e9, mi_f = 500, mi_itexp = 0;
		//for (float f = 500; f < 810; f += 100) {
		//	for (float init_exp = 0; init_exp < 0.9; init_exp += 0.4) {
		//		float er =
		//			pre_cal_exp_ide_R_t(f, ide, bldshps, inner_land_corr,
		//				slt_line, slt_point_rect, i_id, ide_sg_vl, init_exp);
		//		if (er < mi_er) mi_er = er, mi_f = f, mi_itexp = init_exp;
		//	}
		//}

		//ide[i_id].fcs = mi_f;
		//mi_er = pre_cal_exp_ide_R_t(mi_f, ide, bldshps, inner_land_corr,
		//	slt_line, slt_point_rect, i_id, ide_sg_vl, mi_itexp);

		float mi_er = 1e9, mi_f = 800;
		
		ide[i_id].fcs = mi_f;
		mi_er = pre_cal_exp_ide_R_t_dvd(mi_f, ide, bldshps, inner_land_corr,
			slt_line, slt_point_rect, i_id, ide_sg_vl);

		if (mi_er > 3) {
			for (mi_f = 500; mi_f < 3000; mi_f += 500) {
				ide[i_id].fcs = mi_f;
				mi_er = pre_cal_exp_ide_R_t_dvd(mi_f, ide, bldshps, inner_land_corr,
					slt_line, slt_point_rect, i_id, ide_sg_vl);
				if (mi_er <4) break;
			}
		}

#endif // !test_coef
		
		/*FILE *fp;
		fp=fopen( "test_f.txt", "w");*/
		
#ifdef test_coef
		int st = 500, en = 510, step = 25;
		Eigen::VectorXf temp((en-st)/step+1);
		Eigen::MatrixX3f temp_tslt((en - st) / step + 1,3);
		Eigen::MatrixX3f temp_angle((en - st) / step + 1, 3);

		for (int i = st; i < en; i += step) {
			temp((i - st) / step) =
				pre_cal_exp_ide_R_t(i, ide, bldshps, inner_land_corr,
					slt_line, slt_point_rect, i_id, ide_sg_vl,0);
			temp_tslt.row((i - st) / step) = ide[0].tslt.row(0);
			Eigen::Matrix3f R = ide[0].rot.block(0, 0, 3, 3);
			temp_angle.row((i - st) / step) = get_uler_angle_zyx(R).transpose();
			ide[i_id].fcs = i;

			//float px_err = 0;
			//for (int i_exp = 0; i_exp < ide[i_id].num; i_exp++)
			//	px_err += print_error(i, ide, bldshps, i_id, i_exp);
			//temp((i - st) / step) = px_err / ide[i_id].num;
		}
		for (int i = 0; i < (en - st) / step + 1; i++) {
			printf("test cal f %d %.6f\n", st + i * step, temp(i));
			fprintf(fp, "%d %.6f %.6f %.6f %.6f", st + i * step, temp(i), temp_tslt(i,0), temp_tslt(i, 1), temp_tslt(i, 2));
			fprintf(fp, " %.6f %.6f %.6f \n", temp_angle(i, 0), temp_angle(i, 1), temp_angle(i, 2));
		}
		//float er=pre_cal_exp_ide_R_t(500, ide, bldshps, inner_land_corr,
		//	slt_line, slt_point_rect, i_id, ide_sg_vl,0);
		//for (int i = st; i < en; i += step) {
		//	temp((i - st) / step) =
		//		print_error(i, ide, bldshps, i_id, 0);

		//	//float px_err = 0;
		//	//for (int i_exp = 0; i_exp < ide[i_id].num; i_exp++)
		//	//	px_err += print_error(i, ide, bldshps, i_id, i_exp);
		//	//temp((i - st) / step) = px_err / ide[i_id].num;
		//}
		//printf("f:%d er:%.5f\n", 500, er);
		//for (int i = 0; i < (en - st) / step + 1; i++) {
		//	printf("test cal f %d %.6f\n", st + i * step, temp(i));
		//	fprintf(fp, "%d %.6f \n", st + i * step, temp(i));
		//	
		//}
#endif

	}
	fclose(fp);
	//FILE *fp;
	fp=fopen( "test_ide_coeff.txt", "w");
	for (int i = 0; i < G_iden_num; i++)
		fprintf(fp, "%.6f\n", ide[0].user(i));
	fprintf(fp, "sum: %.6f\n", ide[0].user.sum());
	fclose(fp);
	fp=fopen( "test_exp_coeff.txt", "w");
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
			slt_line, slt_point_rect, i_id, ide_sg_vl,0.5);
	}
	fp=fopen( "test_ide_coeff.txt", "w");
	for (int i = 0; i < G_iden_num; i++)
		fprintf(fp, "%.6f\n", ide[0].user(i));
	fclose(fp);
	fp=fopen( "test_exp_coeff.txt", "w");
	for (int i_exp = 0; i_exp < ide[0].num; i_exp++) {
		fprintf(fp, "------------------------------------------\n");
		for (int i = 0; i < G_nShape; i++)
			fprintf(fp, "%.6f\n", ide[0].exp(i_exp, i));
	}
	fclose(fp);
	
}





std::string cal_coef_land_name = "test_coef_land_olsgm_25.txt";
std::string cal_coef_mesh_name = "test_coef_mesh_olsgm_25.txt";
std::string cal_coef_2dland_name = "2dland.txt";

float pre_cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor,
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, int id_idx,
	Eigen::VectorXf &ide_sg_vl, float init_exp) {



	puts("preparing expression & other coeffients...");
	init_exp_ide(ide, id_idx, init_exp);
	float error = 0;

	int tot_r_all = 1;
	int tot_r_one = 1;
	int tot_r_pose = 1;
	Eigen::VectorXf temp(tot_r_all);
	//fprintf(fp, "%d\n",tot_r);
	FILE *fp;
	//fp=fopen( cal_coef_land_name.c_str(), "w");
	//fprintf(fp, "%d\n", 1);
	//fclose(fp);

	//fp=fopen( cal_coef_mesh_name.c_str(), "w");
	//fprintf(fp, "%d\n", 1);
	//fclose(fp);
	//test_coef_land(ide, bldshps, id_idx, 0);
	//test_coef_mesh(ide, bldshps, id_idx, 0);
	//exit(-5);
#ifdef test_coef_save_mesh
	fp=fopen( cal_coef_land_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r_all*tot_r_one + 1)*ide[id_idx].num);
	fclose(fp);

	fp=fopen( cal_coef_mesh_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r_all*tot_r_one + 1)*ide[id_idx].num);
	fclose(fp);
#endif

#ifdef test_posit_by2dland
	fp=fopen( cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r_all * tot_r_one*(tot_r_pose+2));
	fclose(fp);
#endif // test_posit_by2dland
#ifdef test_updt_slt
	fp=fopen(test_updt_slt_path.c_str(), "w");
	fprintf(fp, "%d\n", tot_r_all*tot_r_one);
	fclose(fp);
	fp=fopen(test_updt_slt_2d_path.c_str(), "w");
	fprintf(fp, "%d\n", tot_r_all*tot_r_one*tot_r_pose);
	fclose(fp);
#endif // test_updt_slt
	//float error_last=0;
#ifdef test_inner_land
	fp=fopen( cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r_all*tot_r_one + 1);
	fclose(fp);
#endif // test_inner_land

#ifdef test_ide_cf_def
	fp=fopen( "test_ide_cf.txt", "w");
	fprintf(fp, "%d\n", tot_r_all*tot_r_one + 1);
	fclose(fp);
#endif

	for (int rounds = 0; rounds < tot_r_all; rounds++) {
		///////////////////////////////////////////////paper's solution
		Eigen::VectorXf user_ini = ide[id_idx].user;

		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			ide[id_idx].user = user_ini;
			for (int oneexp_rounds = 0; oneexp_rounds < tot_r_one; oneexp_rounds++) {
				printf("calculate %d id %d exp:\n", id_idx, i_exp);
#ifdef test_posit_by2dland
				test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland 
#ifdef posit

				Eigen::VectorXi land_cor(G_land_num);
				for (int pose_rounds = 0; pose_rounds < tot_r_pose; pose_rounds++) {
					//cal_rt_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
					//test_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
					cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
					//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // posit
#ifdef normalization
					cal_rt_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
					//test_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // normalization
#ifdef test_posit_by2dland
					test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland

					Eigen::VectorXi out_land_cor(15);
					update_slt_dde(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);
#ifdef test_updt_slt
					save_result_one(ide, 0, 0, "./slt_test_obj/Tester_88_pose1_" + std::to_string(rounds) + ".psp_f");
#endif // test_updt_slt

					//std::cout << inner_land_cor << '\n';
					//std::cout <<"--------------\n"<< out_land_cor << '\n';					
					for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
					for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
					ide[id_idx].land_cor.row(i_exp) = land_cor.transpose();

					//test_slt(f, ide, bldshps, land_cor, id_idx, i_exp);
					/*test_slt_vtx_angle(f, ide, bldshps, id_idx, i_exp, 
						slt_line, slt_point_rect, inner_land_cor,0);*/
					test_upt_slt_angle(f, ide, bldshps, id_idx, i_exp,
						slt_line, slt_point_rect, inner_land_cor);

					updt_angle_slt_more(f, ide, bldshps, id_idx, i_exp,
						slt_line, slt_point_rect, inner_land_cor);
				}

#ifdef test_posit_by2dland
				test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland			
#ifdef test_coef_save_mesh
				if (rounds == 0 && oneexp_rounds==0) {
					for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
						test_coef_land(ide, bldshps, id_idx, i_exp);
						test_coef_mesh(ide, bldshps, id_idx, i_exp);
#ifdef test_inner_land
						test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_inner_land
					}

				}
#endif
				error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
				error = cal_3dpaper_ide(f, ide, bldshps, id_idx, i_exp, land_cor, ide_sg_vl);
#ifdef test_coef_save_mesh
				for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
					test_coef_land(ide, bldshps, id_idx, i_exp);
					test_coef_mesh(ide, bldshps, id_idx, i_exp);
#ifdef test_inner_land
					test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_inner_land
				}
#endif
			}
		}
		ide[id_idx].user = user_ini;
		error = cal_fixed_exp_same_ide(f, ide, bldshps, id_idx, ide_sg_vl);

#ifdef test_ide_cf_def
		fp=fopen( "test_ide_cf.txt", "a");
		fprintf(fp, "%d:  ", rounds);
		for (int i_d = 0; i_d < G_iden_num; i_d++)
			fprintf(fp, "%.5f ", ide[id_idx].user(i_d));
		fprintf(fp, "\n\n");
		fclose(fp);
#endif // test_ide_cf_def

		//ide[id_idx].user << 
		//	1,
/*
	0.1381161362,
		0.0222614892,
		0.0128200175,
		0.1203504056,
		0.0936280265,
		0.0555039272,
		-0.1181689948,
		-0.0961747840,
		-0.0250166059,
		0.0107572880,
		-0.1414988190,
		0.0958925188,
		-0.0447807871,
		-0.0162459109,
		0.0070259790,
		-0.0973661989,
		0.1625646651,
		-0.1153525487,
		0.1856404245,
		0.0264545660,
		0.0938270763,
		-0.0678874254,
		-0.1080164239,
		0.0847992525,
		0.1446871161,
		-0.0713867918,
		0.0396330655,
		-0.0704018027,
		0.0161220282,
		-0.0664369911,
	*/


		printf("+++++++++++++%d %.6f\n", rounds, error);
#ifdef test_coef_save_mesh
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			test_coef_land(ide, bldshps, id_idx, i_exp);
			test_coef_mesh(ide, bldshps, id_idx, i_exp);
#ifdef test_inner_land
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_inner_land
		}
#endif

		//if (fabs(error_last - error) < 20) break;
		//error_last = error;
		/*error = print_error(f, ide, bldshps, id_idx, 0);*/
		temp(rounds) = error;

#ifdef upt_inner_cor
		update_inner_land_cor(f, ide, id_idx, 0, inner_land_cor, bldshps);
		fp=fopen( "inner_cor_upt.txt", "a");
		fprintf(fp, "%d", rounds);
		for (int i_v = 0; i_v < G_inner_land_num; i_v++)
			fprintf(fp, " %d", inner_land_cor(i_v));
		fprintf(fp, "\n");
		fclose(fp);
#endif // upt_inner_cor

	}
	for (int i = 0; i < tot_r_all; i++) printf("it %d err %.6f\n", i, temp(i));

	fp=fopen( "convergence_test.txt", "w");
	for (int i = 0; i < tot_r_all; i++) fprintf(fp, "it %d err %.6f\n", i, temp(i));
	fclose(fp);

#ifdef px_err_def
	float px_err = 0;
	fp=fopen( "oneide_difexp_err_px.txt", "w");
	fprintf(fp, "tot err:%.5f\n", error);
	fprintf(fp, "f:%.5f initexp:%.5f\n", f, init_exp);


	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		float this_err = print_error(f, ide, bldshps, id_idx, i_exp);
		px_err += this_err;
		fprintf(fp, "%d %.5f\n", i_exp, this_err);
	}
	fclose(fp);
	return px_err / ide[id_idx].num;

#endif //px_err_def

	return error;
}

void init_exp_ide(iden *ide,int id_idx,float init_exp) {

	puts("initializing coeffients(identitiy,expression) for cal f...");
	ide[id_idx].exp= Eigen::MatrixXf::Constant(ide[id_idx].num, G_nShape, init_exp);////////////////////////////0.5
	for (int j = 0; j < ide[id_idx].num; j++) ide[id_idx].exp(j, 0) = 1;
	ide[id_idx].user= Eigen::MatrixXf::Constant(G_iden_num, 1, 1.0/ G_iden_num);
	//ide[id_idx].user(0) = 1;
	/*
	0.1381161362,
		0.0222614892,
		0.0128200175,
		0.1203504056,
		0.0936280265,
		0.0555039272,
		-0.1181689948,
		-0.0961747840,
		-0.0250166059,
		0.0107572880,
		-0.1414988190,
		0.0958925188,
		-0.0447807871,
		-0.0162459109,
		0.0070259790,
		-0.0973661989,
		0.1625646651,
		-0.1153525487,
		0.1856404245,
		0.0264545660,
		0.0938270763,
		-0.0678874254,
		-0.1080164239,
		0.0847992525,
		0.1446871161,
		-0.0713867918,
		0.0396330655,
		-0.0704018027,
		0.0161220282,
		-0.0664369911,
	*/
}


float pre_cal_exp_ide_R_t_dvd(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor,
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, int id_idx,
	Eigen::VectorXf &ide_sg_vl) {



	puts("preparing expression & other coeffients...");
	init_exp_ide(ide, id_idx, 0.5);
	float error = 0;

	int tot_r = 30, tot_exp_r=6;
	Eigen::VectorXf temp(tot_r);
	//fprintf(fp, "%d\n",tot_r);
	FILE *fp;
	//fp=fopen( cal_coef_land_name.c_str(), "w");
	//fprintf(fp, "%d\n", 1);
	//fclose(fp);

	//fp=fopen( cal_coef_mesh_name.c_str(), "w");
	//fprintf(fp, "%d\n", 1);
	//fclose(fp);
	//test_coef_land(ide, bldshps, id_idx, 0);
	//test_coef_mesh(ide, bldshps, id_idx, 0);
	//exit(-5);
#ifdef test_coef_save_mesh
	fp=fopen( cal_coef_land_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r + 1)*ide[id_idx].num);
	fclose(fp);

	fp=fopen( cal_coef_mesh_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r + 1)*ide[id_idx].num);
	fclose(fp);
#endif

#ifdef test_posit_by2dland
	fp=fopen( cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r * 3);
	fclose(fp);
#endif // test_posit_by2dland
#ifdef test_updt_slt
	fp=fopen( "test_updt_slt.txt", "w");
	fprintf(fp, "%d\n", tot_r);
	fclose(fp);
	fp=fopen( "test_updt_slt_2d_point.txt", "w");
	fprintf(fp, "%d\n", tot_r);
	fclose(fp);
#endif // test_updt_slt
	//float error_last=0;
#ifdef test_inner_land
	fp=fopen( cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r + 1);
	fclose(fp);
#endif // test_inner_land

	for (int rounds = 0; rounds < tot_r; rounds++) {
		///////////////////////////////////////////////paper's solution
		Eigen::VectorXf user_ini = ide[id_idx].user;
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {

			ide[id_idx].user = user_ini;
			printf("calculate %d id %d exp:\n", id_idx, i_exp);
			for (int exp_r = 0; exp_r < tot_exp_r; exp_r++) {
#ifdef test_posit_by2dland
				test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland 
#ifdef posit
				//cal_rt_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
				//test_posit(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
				cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
				//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // posit
#ifdef normalization
				cal_rt_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
				//test_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // normalization



#ifdef test_posit_by2dland
				test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland

				Eigen::VectorXi out_land_cor(15);
				update_slt_dde(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);
#ifdef test_updt_slt
				save_result_one(ide, 0, 0, "./slt_test_obj/Tester_88_pose1_" + std::to_string(rounds) + ".psp_f");
#endif // test_updt_slt

				//std::cout << inner_land_cor << '\n';
				//std::cout <<"--------------\n"<< out_land_cor << '\n';
				Eigen::VectorXi land_cor(G_land_num);
				for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
				for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
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
#ifdef test_inner_land
						test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_inner_land
					}

				}
#endif
				error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
				error = cal_3dpaper_ide(f, ide, bldshps, id_idx, i_exp, land_cor, ide_sg_vl);
			}
		}
		ide[id_idx].user = user_ini;
		error = cal_fixed_exp_same_ide(f, ide, bldshps, id_idx, ide_sg_vl);

		printf("+++++++++++++%d %.6f\n", rounds, error);
#ifdef test_coef_save_mesh
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			test_coef_land(ide, bldshps, id_idx, i_exp);
			test_coef_mesh(ide, bldshps, id_idx, i_exp);
#ifdef test_inner_land
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_inner_land
		}
#endif

		//if (fabs(error_last - error) < 20) break;
		//error_last = error;
		/*error = print_error(f, ide, bldshps, id_idx, 0);*/
		temp(rounds) = error;

#ifdef upt_inner_cor
		update_inner_land_cor(f, ide, id_idx, 0, inner_land_cor, bldshps);
		fp=fopen( "inner_cor_upt.txt", "a");
		fprintf(fp, "%d", rounds);
		for (int i_v = 0; i_v < G_inner_land_num; i_v++)
			fprintf(fp, " %d", inner_land_cor(i_v));
		fprintf(fp, "\n");
		fclose(fp);
#endif // upt_inner_cor
		bool fl = 0;
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			float this_err = print_error(f, ide, bldshps, id_idx, i_exp);
			if (this_err > 3) fl = 1;
		}
		if (fl == 0) break;
	}
	for (int i = 0; i < tot_r; i++) printf("it %d err %.6f\n", i, temp(i));

#ifdef px_err_def
	float px_err = 0;
	fp=fopen( "oneide_difexp_err_px_bf.txt", "w");
	fprintf(fp, "tot err:%.5f\n", error);
	fprintf(fp, "f:%.5f initexp:%.5f\n", f, 0.5);


	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		float this_err = print_error(f, ide, bldshps, id_idx, i_exp);
		px_err += this_err;
		fprintf(fp, "%d %.5f\n", i_exp, this_err);
	}
	fclose(fp);
	//return px_err / ide[id_idx].num;

#endif //px_err_def

	fp=fopen( "optmz_exp.txt", "w");
	int exp_tot_r = 4;
	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		float mi_er = 1e8, mi_exp = 0;
		for (float init_exp = 0; init_exp < 0.91; init_exp += 0.3) {
			ide[id_idx].exp.row(i_exp) = Eigen::MatrixXf::Constant(1, G_nShape, init_exp);
			ide[id_idx].exp(i_exp, 0) = 1;
			float error = 0, error_px = 0;
			fprintf(fp, "-------------------------------\n");
			fprintf(fp, "id:%d exp:%d init_exp:%.5f \n", id_idx, i_exp, init_exp);
			for (int rounds = 0; rounds < exp_tot_r; rounds++) {

#ifdef posit
				cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
				//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // posit

				Eigen::VectorXi out_land_cor(15);
				update_slt_dde(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);

				Eigen::VectorXi land_cor(G_land_num);
				for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
				for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
				ide[id_idx].land_cor.row(i_exp) = land_cor.transpose();

				error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
				error_px = print_error(f, ide, bldshps, id_idx, i_exp);
				fprintf(fp, "rounds:%d error:%.5f error_px:%.5f \n", rounds, error, error_px);
			}
			if (error_px < mi_er) mi_er = error_px, mi_exp = init_exp;
		}
		if (mi_er > 3) {
			for (float init_exp = 0.1; init_exp < 0.91; init_exp += 0.3) {
				ide[id_idx].exp.row(i_exp) = Eigen::MatrixXf::Constant(1, G_nShape, init_exp);
				ide[id_idx].exp(i_exp, 0) = 1;
				float error = 0, error_px = 0;
				fprintf(fp, "-------------------------------\n");
				fprintf(fp, "id:%d exp:%d init_exp:%.5f \n", id_idx, i_exp, init_exp);
				for (int rounds = 0; rounds < exp_tot_r; rounds++) {

#ifdef posit
					cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
					//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // posit

					Eigen::VectorXi out_land_cor(15);
					update_slt_dde(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);

					Eigen::VectorXi land_cor(G_land_num);
					for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
					for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
					ide[id_idx].land_cor.row(i_exp) = land_cor.transpose();

					error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
					error_px = print_error(f, ide, bldshps, id_idx, i_exp);
					fprintf(fp, "rounds:%d error:%.5f error_px:%.5f \n", rounds, error, error_px);
				}
				if (error_px < mi_er) mi_er = error_px, mi_exp = init_exp;
			}
			if (mi_er > 3) {
				for (float init_exp = 0.2; init_exp < 0.91; init_exp += 0.3) {
					ide[id_idx].exp.row(i_exp) = Eigen::MatrixXf::Constant(1, G_nShape, init_exp);
					ide[id_idx].exp(i_exp, 0) = 1;
					float error = 0, error_px = 0;
					fprintf(fp, "-------------------------------\n");
					fprintf(fp, "id:%d exp:%d init_exp:%.5f \n", id_idx, i_exp, init_exp);
					for (int rounds = 0; rounds < exp_tot_r; rounds++) {

#ifdef posit
						cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
						//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // posit

						Eigen::VectorXi out_land_cor(15);
						update_slt_dde(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);

						Eigen::VectorXi land_cor(G_land_num);
						for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
						for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
						ide[id_idx].land_cor.row(i_exp) = land_cor.transpose();

						error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
						error_px = print_error(f, ide, bldshps, id_idx, i_exp);
						fprintf(fp, "rounds:%d error:%.5f error_px:%.5f \n", rounds, error, error_px);
					}
					if (error_px < mi_er) mi_er = error_px, mi_exp = init_exp;
				}
			}
		}
		ide[id_idx].exp.row(i_exp) = Eigen::MatrixXf::Constant(1, G_nShape, mi_exp);
		ide[id_idx].exp(i_exp, 0) = 1;
		float error = 0, error_px = 0;
		for (int rounds = 0; rounds < exp_tot_r; rounds++) {

#ifdef posit
			cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // posit

			Eigen::VectorXi out_land_cor(15);
			update_slt_dde(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);

			Eigen::VectorXi land_cor(G_land_num);
			for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
			for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
			ide[id_idx].land_cor.row(i_exp) = land_cor.transpose();

			error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
			error_px = print_error(f, ide, bldshps, id_idx, i_exp);
		}
		fprintf(fp, "final : init_exp:%.2f error:%.5f error_px:%.5f \n", mi_exp, error, error_px);
	}
	fclose(fp);

#ifdef px_err_def
	px_err = 0;
	fp=fopen( "oneide_difexp_err_px_aft.txt", "w");	
	fprintf(fp, "f:%.5f \n", f);

	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		float this_err = print_error(f, ide, bldshps, id_idx, i_exp);
		px_err += this_err;
		fprintf(fp, "%d %.5f\n", i_exp, this_err);
	}
	fclose(fp);
	return px_err / ide[id_idx].num;

#endif //px_err_def

	return error;
}


float admm_cal_exp_ide_R_t(
	float f, iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &inner_land_cor,
	std::vector <int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, int id_idx,
	Eigen::VectorXf &ide_sg_vl) {



	puts("preparing expression & other coeffients...");
	init_exp_ide(ide, id_idx, 0.5);
	float error = 0;

	int tot_r = 6;
	Eigen::VectorXf temp(tot_r), temp_lmd(tot_r), temp_sumu(tot_r);
	//fprintf(fp, "%d\n",tot_r);
	FILE *fp;

#ifdef test_coef_save_mesh
	fp=fopen( cal_coef_land_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r + 1)*ide[id_idx].num);
	fclose(fp);

	fp=fopen( cal_coef_mesh_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r + 1)*ide[id_idx].num);
	fclose(fp);
#endif

#ifdef test_posit_by2dland
	fp=fopen( cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r * 3);
	fclose(fp);
#endif // test_posit_by2dland
#ifdef test_updt_slt
	fp=fopen( "test_updt_slt.txt", "w");
	fprintf(fp, "%d\n", tot_r);
	fclose(fp);
	fp=fopen( "test_updt_slt_2d_point.txt", "w");
	fprintf(fp, "%d\n", tot_r);
	fclose(fp);
#endif // test_updt_slt
	//float error_last=0;
#ifdef test_inner_land
	fp=fopen( cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", tot_r + 1);
	fclose(fp);
#endif // test_inner_land
	float lmd = 1;
	for (int rounds = 0; rounds < tot_r; rounds++) {
		///////////////////////////////////////////////paper's solution

		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			printf("calculate %d id %d exp:\n", id_idx, i_exp);
#ifdef test_posit_by2dland
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland 
#ifdef posit

			cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_pnp(f, ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // posit
#ifdef normalization
			cal_rt_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
			//test_normalization(ide, bldshps, inner_land_cor, id_idx, i_exp);
#endif // normalization

#ifdef test_posit_by2dland
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_posit_by2dland

			Eigen::VectorXi out_land_cor(15);
			update_slt_me(f, ide, bldshps, id_idx, i_exp, slt_line, slt_point_rect, out_land_cor);
#ifdef test_updt_slt
			save_result_one(ide, 0, 0, "./slt_test_obj/Tester_88_pose1_" + std::to_string(rounds) + ".psp_f");
#endif // test_updt_slt

			//std::cout << inner_land_cor << '\n';
			//std::cout <<"--------------\n"<< out_land_cor << '\n';
			Eigen::VectorXi land_cor(G_land_num);
			for (int i = 0; i < 15; i++) land_cor(i) = out_land_cor(i);
			for (int i = 15; i < G_land_num; i++) land_cor(i) = inner_land_cor(i - 15);
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
#ifdef test_inner_land
					test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_inner_land
				}

			}
#endif
			error = cal_3dpaper_exp(f, ide, bldshps, id_idx, i_exp, land_cor);
			//error = cal_3dpaper_ide_admm(f, ide, bldshps, id_idx, i_exp, land_cor, ide_sg_vl, lmd);
			lmd = lmd + 0.1*(ide[id_idx].user.sum() - 1);
		}
		

		printf("+++++++++++++%d %.6f\n", rounds, error);
#ifdef test_coef_save_mesh
		for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
			test_coef_land(ide, bldshps, id_idx, i_exp);
			test_coef_mesh(ide, bldshps, id_idx, i_exp);
#ifdef test_inner_land
			test_2dland(f, ide, bldshps, id_idx, i_exp);
#endif // test_inner_land
		}
#endif

		//if (fabs(error_last - error) < 20) break;
		//error_last = error;
		/*error = print_error(f, ide, bldshps, id_idx, 0);*/
		temp(rounds) = error;
		temp_lmd(rounds) = lmd;
		temp_sumu(rounds) = ide[id_idx].user.sum();
		

	}
	for (int i = 0; i < tot_r; i++) printf("it %d err %.6f   lmd: %.6f   sumu: %.6f\n", i, temp(i),temp_lmd(i),temp_sumu(i));

#ifdef px_err_def
	float px_err = 0;
	fp=fopen( "oneide_difexp_err_px_dmm.txt", "w");
	fprintf(fp, "tot err:%.5f\n", error);
	fprintf(fp, "f:%.5f initexp:%.5f\n", f, 0.5);


	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
		float this_err = print_error(f, ide, bldshps, id_idx, i_exp);
		px_err += this_err;
		fprintf(fp, "%d %.5f\n", i_exp, this_err);
	}
	fclose(fp);
	return px_err / ide[id_idx].num;

#endif //px_err_def

	return error;
}



void cal_rt_posit(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("POSIT...");

	Eigen::MatrixX2f land_in; Eigen::MatrixX3f bs_in;
	int temp_num = 0;
	if (ide[id_idx].land_cor(exp_idx,20) == inner_land_cor(20 - 15) && ide[id_idx].land_cor(exp_idx, 30) == inner_land_cor(30 - 15)) {
		land_in.resize(G_land_num, 2);
		land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2);
		//std::cout << land_in.transpose() << '\n';
		bs_in.resize(G_land_num, 3);
		for (int i = 0; i < G_land_num; i++)
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
		std::cout<< cnt << " angle:\n" <<get_uler_angle_zyx(rot)*180/(acos(-1)) << "\n";
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
	std::cout << ide[id_idx].land_2d.block(G_land_num*exp_idx+15, 0, G_land_num - 15, 2) << '\n';
	system("pause");
}

const int eye_corner_center[5] = { 27,29,31,33,64};

void cal_rt_pnp(
	float f, iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &inner_land_cor, int id_idx, int exp_idx) {

	puts("solve pnp...");

	std::vector<cv::Point2f> land_2d; land_2d.clear();
	std::vector<cv::Point3f> land_3d; land_3d.clear();

	//if (ide[id_idx].land_cor(exp_idx, 20) == inner_land_cor(20 - 15) && ide[id_idx].land_cor(exp_idx, 30) == inner_land_cor(30 - 15)) {
	//	/*land_in.resize(G_land_num, 2);
	//	land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2);*/
	//	for (int i_v = exp_idx * G_land_num; i_v < exp_idx*G_land_num + G_land_num; i_v++)
	//		land_2d.push_back(cv::Point2f(ide[id_idx].land_2d(i_v, 0), ide[id_idx].land_2d(i_v, 1)));
	//	//std::cout << land_in.transpose() << '\n';
	//	//bs_in.resize(G_land_num, 3);
	//	land_3d.resize(G_land_num);
	//	for (int i = 0; i < G_land_num; i++) {
	//		//			for (int axis = 0; axis < 3; axis++)
	//		land_3d[i].x = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 0);
	//		land_3d[i].y = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 1);
	//		land_3d[i].z = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 2);
	//	}



	//	//std::cout << bs_in << '\n';
	//	//std::cout << land_in << "\n";
	//	temp_num = G_land_num;
	//}
	//else
	//{
	//	//land_in.resize(G_inner_land_num, 2);
	//	//land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num + 15, 0, G_inner_land_num, 2);
	//	for (int i_v = exp_idx * G_land_num + 15; i_v < exp_idx*G_land_num + G_land_num; i_v++)
	//		land_2d.push_back(cv::Point2f(ide[id_idx].land_2d(i_v, 0), ide[id_idx].land_2d(i_v, 1)));
	//	//std::cout << land_in.transpose() << '\n';
	//	//bs_in.resize(G_inner_land_num, 3);
	//	land_3d.resize(G_inner_land_num);

	//	//cal_inner_bldshps(ide, bldshps, bs_in, inner_land_cor, id_idx, exp_idx);

	//	for (int i = 0; i < G_inner_land_num; i++) {
	//		//			for (int axis = 0; axis < 3; axis++)
	//		land_3d[i].x = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 0);
	//		land_3d[i].y = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 1);
	//		land_3d[i].z = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 2);
	//		//land_3d[i] *= 10000;
	//	}
	//	//std::cout << bs_in << '\n';
	//	//std::cout << land_in << "\n";
	//	temp_num = G_inner_land_num;
	//}

	if (ide[id_idx].land_cor(exp_idx, 20) == inner_land_cor(20 - 15) && ide[id_idx].land_cor(exp_idx, 30) == inner_land_cor(30 - 15)) {
		puts("eye_corner_center");
		for (int i_v = 0; i_v < 15; i_v++) {			
			land_2d.push_back(
				cv::Point2f(
					ide[id_idx].land_2d(exp_idx * G_land_num + i_v, 0),
					ide[id_idx].land_2d(exp_idx * G_land_num + i_v, 1)));

			land_3d.push_back(
				cv::Point3f(
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 0),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 1),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 2)));

		}
		for (int i_v = 35; i_v < 46; i_v++) {
			land_2d.push_back(
				cv::Point2f(
					ide[id_idx].land_2d(exp_idx * G_land_num + i_v, 0),
					ide[id_idx].land_2d(exp_idx * G_land_num + i_v, 1)));

			land_3d.push_back(
				cv::Point3f(
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 0),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 1),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 2)));

		}

		for (int i = 0; i < 5; i++) {

			int i_v = eye_corner_center[i];
			land_2d.push_back(
				cv::Point2f(
					ide[id_idx].land_2d(exp_idx * G_land_num + i_v, 0),
					ide[id_idx].land_2d(exp_idx * G_land_num + i_v, 1)));

			land_3d.push_back(
				cv::Point3f(
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 0),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 1),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i_v), 2)));

		}

		//for (int i = 0; i < G_land_num; i++) {
		//	land_2d.push_back(
		//		cv::Point2f(
		//			ide[id_idx].land_2d(exp_idx * G_land_num + i, 0),
		//			ide[id_idx].land_2d(exp_idx * G_land_num + i, 1)));

		//	land_3d.push_back(
		//		cv::Point3f(
		//			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 0),
		//			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 1),
		//			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), 2)));

		//}
		//for (int i = 0; i < G_inner_land_num; i++) {
		//	land_2d.push_back(
		//		cv::Point2f(
		//			ide[id_idx].land_2d(exp_idx * G_land_num + i + 15, 0),
		//			ide[id_idx].land_2d(exp_idx * G_land_num + i + 15, 1)));

		//	land_3d.push_back(
		//		cv::Point3f(
		//			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 0),
		//			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 1),
		//			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 2)));

		//}

	}
	else
	{
		//land_in.resize(G_inner_land_num, 2);
		//land_in = ide[id_idx].land_2d.block(exp_idx*G_land_num + 15, 0, G_inner_land_num, 2);
		for (int i = 0; i < G_inner_land_num; i++) {			
			land_2d.push_back(
				cv::Point2f(
					ide[id_idx].land_2d(exp_idx * G_land_num + i + 15, 0),
					ide[id_idx].land_2d(exp_idx * G_land_num + i + 15, 1)));

			land_3d.push_back(
				cv::Point3f(
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 0),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 1),
					cal_3d_vtx(ide, bldshps, id_idx, exp_idx, inner_land_cor(i), 2)));

		}
	}




	// Camera internals

	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 
		f, 0, ide[id_idx].center(exp_idx,0), 0, f, ide[id_idx].center(exp_idx, 1), 0, 0, 1);
	cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

	//std:: cout << "Camera Matrix \n" << camera_matrix << "\n";
	// Output rotation and translation
	cv::Mat rotation_vector; // Rotation in axis-angle form
	cv::Mat translation_vector;

	// Solve for pose
/*	puts("ini 3d:");
	for (int i = 0; i < 6; i++)
		std::cout << i << ' ' << land_3d[i] << "\n";
	puts("ini 2d:");
	for (int i = 0; i < 6; i++)
		std::cout << i << ' ' << land_2d[i] << "\n";
*/
	cv::solvePnP(land_3d, land_2d, camera_matrix, dist_coeffs, rotation_vector, translation_vector,
		0, CV_EPNP);

	for (int axis = 0; axis < 3; axis++)
		ide[id_idx].tslt(exp_idx, axis) = translation_vector.at<double>(axis);

//	std::cout << "rotation_vector:\n" << rotation_vector << "\n\n";

	cv::Mat rot;
	cv::Rodrigues(rotation_vector, rot);
	//Eigen::Map<Eigen::Matrix3d> R(rot.ptr <double>(), rot.rows, rot.cols);
	for (int i =0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			ide[id_idx].rot(3 * exp_idx+i, j) = rot.at<double>(i, j);
	

	//puts("-----------------------------------------");
	//std::cout << rot << "\n";
	///*std::cout << R << "\n";
	//system("pause");*/
	//std::cout << ide[id_idx].rot << '\n';



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
	std::cout <<"rot*rot\n" << rot*rot.transpose() << '\n';
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
	if (ide[id_idx].land_cor(exp_idx,20)==inner_land_cor(20-15) && ide[id_idx].land_cor(exp_idx,30) == inner_land_cor(30 - 15)) {
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
	ide[id_idx].tslt.block(exp_idx,0,1,2) = (center_2d - (A*center_3d.transpose()).transpose());
	//puts("A");
	Eigen::RowVector3f I=A.row(0), J = A.row(1);
	Eigen::Matrix3f A_;
	//puts("A");
	A_.row(0) = I, A_.row(1) = J, A_.row(2) = I.cross(J);
	Eigen :: JacobiSVD<Eigen::MatrixXf> svd(A_, Eigen::ComputeFullV | Eigen:: ComputeFullU);
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
	std::cout << temp.block(0,0,2,G_inner_land_num) << "+++++++\n";
	std::cout << 
		ide[id_idx].land_2d.block(15+G_land_num*exp_idx, 0, G_inner_land_num, 2).transpose().colwise()
		+ ide[id_idx].center.row(exp_idx).transpose()<< "------\n";
	std::cout <<
		ide[id_idx].center.row(exp_idx).transpose() << "pppppppppppp-\n";
	system("pause");
#endif // normalization
}

void test_slt_vtx_angle(float f,iden* ide, Eigen::MatrixXf &bldshps,int id_idx,int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, 
	Eigen::VectorXi &inner_land_cor, int which_angle) {


	Eigen::Matrix3f R = ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3);
	Eigen::VectorXf angle = get_uler_angle_zyx(R);
	std::vector<std::pair<float, float> > ans;
	ans.push_back(std::make_pair(angle(which_angle), print_error(f, ide, bldshps, id_idx, exp_idx)));

	float be = -1.2, en = 1.55, step = 0.1;
	FILE *fp;
	fp = fopen(cal_coef_2dland_name.c_str(), "w");	
	fprintf(fp, "%d\n",(int)((en-be)/step)+1 );
	fclose(fp);
#ifdef test_posit_by2dland
	test_2dland(f, ide, bldshps, id_idx, exp_idx);
#endif // test_posit_by2dland

	Eigen::MatrixXf aet((int)((en - be) / step) + 1 + 1, 8);
	aet.block(0, 0, 1, 3) = angle.transpose();
	aet.block(0, 3, 1, 3) = ide[id_idx].tslt.row(exp_idx);
	aet(0, 6) = print_error(f, ide, bldshps, id_idx, exp_idx);
	aet(0, 7) = angle_err(f, ide, bldshps, id_idx, exp_idx);

	int i = 1;
	for (float ag = be; ag < en; ag += step,i++) {
		angle(which_angle) = ag;
		ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = get_r_from_angle_zyx(angle);

		Eigen::VectorXi out_land_cor(15);
		update_slt_dde(f, ide, bldshps, id_idx, exp_idx, slt_line, slt_point_rect, out_land_cor);
		ide[id_idx].land_cor.block(exp_idx, 0, 1, 15) = out_land_cor.transpose();
		cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
		
		printf("-----------------------------\naxis y angle: %.10f\n", ag);
		float err=print_error(f, ide, bldshps, id_idx, exp_idx);
		Eigen::VectorXf ang_aft = get_uler_angle_zyx(ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3));
		ans.push_back(std::make_pair(ang_aft(which_angle), err));
		aet.block(i, 0, 1, 3) = ang_aft.transpose();
		aet.block(i, 3, 1, 3) = ide[id_idx].tslt.row(exp_idx);
		aet(i, 6) = err;
		aet(i,7)= angle_err(f, ide, bldshps, id_idx, exp_idx);
#ifdef test_posit_by2dland
		test_2dland(f, ide, bldshps, id_idx, exp_idx);
#endif // test_posit_by2dland
	}
	for (int i = 0; i < ans.size(); i++)
		printf("%d %.2f %.2f %.5f\n", i,step*i+be, ans[i].first, ans[i].second);
	
	std::cout << aet << "\n";


	exit(99);
}

float angle_err(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {
	Eigen::MatrixX2f land;
	cal_2dland_fidexp(f, ide, bldshps, land, id_idx, exp_idx);
	Eigen::MatrixX2f dis = ide[id_idx].land_2d.block(G_land_num*exp_idx, 0, G_land_num, 2) - land;

	float ag_err =
		dis.block(0, 0, G_land_num - G_inner_land_num, 2).squaredNorm()*2 +
		dis.block(G_land_num - G_inner_land_num, 0, G_inner_land_num, 2).squaredNorm();

	return sqrt(ag_err / G_land_num / 2);				
}

float deal_one_angle(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &inner_land_cor, int which_angle,float be,float en,float step) {

#ifdef test_posit_by2dland
	FILE *fp;
	fp = fopen(cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", (int)((en - be) / step+1)*(2*2)+1);
	fclose(fp);
#endif // test_posit_by2dland
#ifdef test_updt_slt
	fp = fopen(test_updt_slt_path.c_str(), "w");
	fprintf(fp, "%d\n", (int)((en - be) / step + 1)*(2) + 1);
	fclose(fp);
	fp = fopen(test_updt_slt_2d_path.c_str(), "w");
	fprintf(fp, "%d\n", (int)((en - be) / step + 1)*(2) + 1);
	fclose(fp);
#endif // test_updt_slt
	Eigen::Matrix3f R = ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3);
	Eigen::VectorXf angle = get_uler_angle_zyx(R), mi_angle = angle;
	float mi_ag_err = 10000;
	for (float ag = be; ag < en+1e-6; ag += step) {
		angle(which_angle) = ag;
		ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = get_r_from_angle_zyx(angle);
		for (int rd = 0; rd < 2; rd++) {
			Eigen::VectorXi out_land_cor(15);

			update_slt_dde(f, ide, bldshps, id_idx, exp_idx, slt_line, slt_point_rect, out_land_cor);
			ide[id_idx].land_cor.block(exp_idx, 0, 1, 15) = out_land_cor.transpose();
#ifdef test_posit_by2dland
			test_2dland(f, ide, bldshps, id_idx, exp_idx);
#endif // test_posit_by2dland
			cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
#ifdef test_posit_by2dland
			test_2dland(f, ide, bldshps, id_idx, exp_idx);
#endif // test_posit_by2dland
			printf("%.2f %d ag error: %.5f----------\n",ag, rd, angle_err(f, ide, bldshps, id_idx, exp_idx));
			Eigen::VectorXf ang_aft = get_uler_angle_zyx(ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3));
			std::cout << "now angle: " << ang_aft.transpose() << "\n";
			//ang_aft(0) = angle(0), ang_aft(2) = angle(2);
			//ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = get_r_from_angle_zyx(ang_aft);
		}
		float error = angle_err(f, ide, bldshps, id_idx, exp_idx);

		printf("%d which %.2f %.2f\n",which_angle, ag, error);
		Eigen::VectorXf ang_aft = get_uler_angle_zyx(ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3));
		std::cout << "aft angle: " << ang_aft.transpose() << "\n";
		if (error < mi_ag_err) {
			mi_ag_err= error;
			mi_angle = ang_aft;
		}
	}	
	std::cout << " final smallest angle"<< mi_angle << "\n";

	ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = get_r_from_angle_zyx(mi_angle);
	Eigen::VectorXi out_land_cor(15);
	update_slt_dde(f, ide, bldshps, id_idx, exp_idx, slt_line, slt_point_rect, out_land_cor);
	ide[id_idx].land_cor.block(exp_idx, 0, 1, 15) = out_land_cor.transpose();
	cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
	return mi_ag_err;
}

float updt_angle_slt_more(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &inner_land_cor) {

	if (angle_err(f, ide, bldshps, id_idx, exp_idx) > 3.5) {
		
		std::cout << "bf error: " << angle_err(f, ide, bldshps, id_idx, exp_idx) << "\n";
		deal_one_angle(
			f, ide, bldshps, id_idx, exp_idx,
			slt_line, slt_point_rect, inner_land_cor, 1, 0, 1.51,0.1);
		std::cout << "aft error: " << angle_err(f, ide, bldshps, id_idx, exp_idx) << "\n";
		std::cout << "aft angle :" 
			<< get_uler_angle_zyx(ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3)).transpose() << "\n";
		//exit(99);
		/*deal_one_angle(
			f, ide, bldshps, id_idx, exp_idx,
			slt_line, slt_point_rect, inner_land_cor, 2, -1, 1.1, 0.5);
			*/
	}
	return angle_err(f, ide, bldshps, id_idx, exp_idx);
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




void update_slt_me(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, 
	Eigen::VectorXi &out_land_cor) {
	////////////////////////////////project
	puts("updating silhouette...");
	Eigen::Matrix3f R=ide[id_idx].rot.block(3 * exp_idx,0,3,3);
	Eigen::VectorXf angle = get_uler_angle_zyx(R);
	Eigen::Vector3f T = ide[id_idx].tslt.row(exp_idx).transpose();

	//puts("A");

	Eigen::VectorXf land_cor_mi(15);
	for (int i = 0; i < 15; i++) land_cor_mi(i) = 1e8;

	//puts("B");
	//FILE *fp;
	//fp=fopen( "test_slt.txt", "w");
	if (fabs(angle(2)) < 0.2) {
		/*std::vector<cv::Point2f> test_slt_2dpt;
		test_slt_2dpt.clear();*/
		for (int i_line = 0; i_line < G_line_num; i_line++) {
#ifdef posit
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

#endif // posit

		}
//		for (int i_line = 49; i_line < G_line_num; i_line++) {
//#ifdef posit
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
//#endif // posit
//
//		}
////#ifdef test_updt_slt
////		FILE *fp;
////		fp=fopen( "test_updt_slt_me_2d_point.txt", "w");
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
//#ifdef posit
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
//#endif // posit
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
#ifdef posit
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

#endif // posit

		}
#ifdef test_updt_slt
		FILE *fp;
		fp=fopen( "test_updt_slt_me_2d_point.txt", "w");
		fprintf(fp, "%d\n", test_slt_2dpt.size());
		for (int t=0;t< test_slt_2dpt.size();t++)
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
#ifdef posit
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
#endif // posit
		
		}

	}
	else {
		for (int i_line = 49; i_line < G_line_num; i_line++) {
#ifdef posit
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

#endif // posit

		}
		for (int i_line = 0; i_line < 49; i_line++) {
			float min_v_n = 10000;
			int min_idx = 0;
			Eigen::Vector3f cdnt;
			int en = slt_line[i_line].size(), be = 0;
			if (angle(1) > 0.1 && i_line >= 42) en /= 2;
#ifdef posit
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
#endif // posit

		}
	}

	std::cout << "land_cor_mi:\n" << land_cor_mi.transpose() << "\n";
	
	std::cout << "out land correlation\n" << out_land_cor.transpose() << "\n";
	system("pause");
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
	//fp=fopen( "test_slt.txt", "w");
	for (int i = 0; i < G_line_num; i++) {
		//printf("i %d\n", i);
		float min_v_n = 10000;
		int min_idx = 0;
		Eigen::Vector3f cdnt;
		int en = slt_line[i].size(), be = 0;
		if (angle(1) < -0.1 && i < 34) en /= 3;
		if (angle(1) < -0.1 && i >= 34 && i < 41) en /= 2;
		if (angle(1) > 0.1 && i >= 49 && i < 84) en /= 3;
		if (angle(1) > 0.1 && i >= 42 && i < 49) en /= 2;

		if ((fabs(angle(1)) < 0.5) && ((i < 26) || (i >= 57 && i < 84))) be = 4, en = slt_line[i].size() / 2;
		if ((fabs(angle(1)) < 0.5) && ((i >= 26 && i < 31) || (0))) be = 2, en = slt_line[i].size() / 3;
#ifdef posit

		for (int j = be; j < en; j++) {
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
		for (int j = be; j < en; j++) {

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
	fp=fopen( "test_updt_slt.txt", "a");
	fprintf(fp, "%.5f %.5f %.5f  ", angle(0) * 180 / acos(-1), angle(1) * 180 / acos(-1), angle(2) * 180 / acos(-1));
	for (int j = 0; j < G_line_num; j++)
		fprintf(fp, " %d", slt_cddt(j));
	fprintf(fp, "\n");
	fclose(fp);
	fp=fopen( "test_updt_slt_2d_point.txt", "a");
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
//#ifdef posit
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
#ifdef posit
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

void update_slt_dde(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &out_land_cor) {
	////////////////////////////////project
	puts("updating silhouette dde...");
	Eigen::Matrix3f R = ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3);
//	puts("A");
	Eigen::VectorXf angle = get_uler_angle_zyx(R);

	//std::cout << "angle : " << angle << "\n";
	Eigen::Vector3f T = ide[id_idx].tslt.row(exp_idx).transpose();

//	puts("A");

	Eigen::VectorXi slt_cddt(G_line_num);
	Eigen::MatrixX3f slt_cddt_cdnt(G_line_num, 3);

	for (int i = 0; i < G_line_num; i++) {
		//printf("i %d\n", i);
		float min_v_n = 10000;
		int min_idx = 0;
		Eigen::Vector3f cdnt;

		int en = slt_line[i].size(), be = 0;
		if ( angle(1) < -0.1 && i < 34) en /= 4;
		if ( angle(1) < -0.1 && i >= 34 && i < 41) en /= 2;
		if ( angle(1) > 0.1 && i >= 49 && i < 84) en /= 4;
		if ( angle(1) > 0.1 && i >= 42 && i < 49) en /= 2;
		//if ((fabs( angle(1)) < 0.5) && ((i < 26) || (i >= 57 && i < 84))) be = 3, en = slt_line[i].size() / 2;
		//if ((fabs( angle(1)) < 0.5) && ((i >= 26 && i < 31) || (0))) be = 2, en = slt_line[i].size() / 3;

		for (int j = be; j < en; j++) {
			int x = slt_line[i][j];

			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++)
				point[0](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, x, axis);
				
			point[0] = R * point[0];

			//for (int k = 0, sz = slt_point_rect[x].size(); k < sz; k++) {
			//	//printf("k %d\n", k);
			//	for (int axis = 0; axis < 3; axis++) {
			//		point[1](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].first, axis);						
			//		point[2](axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, slt_point_rect[x][k].second, axis);						
			//	}
			//	for (int i = 1; i < 3; i++) point[i] = R * point[i];
			//	V[0] = point[1] - point[0];
			//	V[1] = point[2] - point[0];

			//	V[0] = V[0].cross(V[1]);

			//	V[0].normalize();
			//	nor = nor + V[0];

			//}

			//nor.normalize();

			//if (fabs(nor(2)) < min_v_n) min_v_n = fabs(nor(2)), min_idx = x, cdnt = point[0]+T;

			point[1] = point[0];
			point[0].normalize();
			
			if (fabs(point[0](2)) < min_v_n) min_v_n = fabs(point[0](2)), min_idx = x, cdnt = point[1]+T;
		}
		//puts("H");
		//fprintf(fp, "%.6f %.6f %.6f \n", cdnt(0), cdnt(1), cdnt(2));
		slt_cddt(i) = min_idx;
		cdnt(0) = cdnt(0)*f / cdnt(2) + ide[id_idx].center(exp_idx, 0);
		cdnt(1) = cdnt(1)*f / cdnt(2) + ide[id_idx].center(exp_idx, 1);
		slt_cddt_cdnt.row(i) = cdnt.transpose();

	}
#ifdef test_updt_slt
	FILE *fp;	
	fp = fopen("test_updt_slt_dde.txt", "a");
	fprintf(fp, "%.5f %.5f %.5f  ", angle(0) * 180 / acos(-1), angle(1) * 180 / acos(-1), angle(2) * 180 / acos(-1));
	for (int j = 0; j < G_line_num; j++)
		fprintf(fp, " %d", slt_cddt(j));
	fprintf(fp, "\n");
	fclose(fp);
	fp=fopen( "test_updt_slt_dde_2d_point.txt", "a");
	fprintf(fp, "%d\n", slt_cddt_cdnt.rows());
	for (int t = 0; t < slt_cddt_cdnt.rows(); t++)
		fprintf(fp, "%.5f %.5f\n", slt_cddt_cdnt(t,0), slt_cddt_cdnt(t,1));
	fprintf(fp, "\n");
	fclose(fp);
#endif // test_updt_slt

	int mid_jaw_idx = 41;
	out_land_cor(7) = slt_cddt(mid_jaw_idx);

	Eigen::VectorXf length_sum(G_line_num);
	length_sum(mid_jaw_idx) = 0;
	for (int i = mid_jaw_idx - 1; i >= 0; i--) {
		//printf("right slt %d\n", i);
		Eigen::Vector2f pt0 = slt_cddt_cdnt.block(i + 1, 0, 1, 2).transpose();
		Eigen::Vector2f pt1 = slt_cddt_cdnt.block(i, 0, 1, 2).transpose();
		
		length_sum(i) = length_sum(i + 1) + (pt0 - pt1).norm();
//		std::cout << "pt0:" << pt0 << " pt1:" << pt1 << "\n";
//		printf("%d %.2f\n", i, length_sum(i));
	}
	float intv = length_sum(8) / 7;// in case of no index for the last one due to the float error.
	//printf(" %.2f\n", intv);
	int now_idx = 8;
	for (int i = mid_jaw_idx - 1; i >= 0 && now_idx < 15; i--) {
//		printf("%d %d %.2f %.2f\n", i, now_idx, length_sum(i), (now_idx - 7)*intv);
		if (length_sum(i) > (now_idx - 7)*intv) {
			out_land_cor(now_idx) = slt_cddt(i);
			now_idx++;
		}
	}
	if (now_idx == 14) out_land_cor(now_idx) = slt_cddt(0);// in case of no index for the last one due to the float error.

	length_sum(mid_jaw_idx) = 0;
	for (int i = mid_jaw_idx + 1; i < G_line_num; i++) {
//		printf("left slt %d\n", i);
		Eigen::Vector2f pt0 = slt_cddt_cdnt.block(i - 1, 0, 1, 2).transpose();
		Eigen::Vector2f pt1 = slt_cddt_cdnt.block(i, 0, 1, 2).transpose();
		length_sum(i) = length_sum(i - 1) + (pt0 - pt1).norm();
	}
	intv = length_sum(G_line_num - 10) / 7;// in case of no index for the last one due to the float error.

	now_idx = 6;
	for (int i = mid_jaw_idx + 1; i < G_line_num && now_idx >= 0; i++)
		if (length_sum(i) > (7 - now_idx)*intv) {
			out_land_cor(now_idx) = slt_cddt(i);
			now_idx--;
		}
	if (now_idx == 0) out_land_cor(now_idx) = slt_cddt(G_line_num - 1);


	//std::cout << "length_sum:\n" << length_sum.transpose() << "\n";

	//std::cout << "out land correlation\n" << out_land_cor.transpose() << "\n";
	//system("pause");
}

void test_slt(float f,iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_land_num, 3);
	for (int i = 0; i < G_land_num; i++)
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, land_cor(i), axis);
	/*FILE *fp;
	fp=fopen( "test_slt_picked_out.txt", "w");
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);*/


	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	puts("aabb");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
	puts("aacc");
	std::cout << bs << '\n';
	//std::cout << tslt << '\n';
#ifdef posit
	Eigen::MatrixXf temp = (rot * bs.transpose()).colwise() + tslt;
	temp.row(0).array() /= temp.row(2).array();
	temp.row(1).array() /= temp.row(2).array();
	temp = temp.array()*f;
	temp.block(0, 0,2, G_land_num).colwise() += ide[id_idx].center.row(exp_idx).transpose();
#endif // posit

#ifdef normalization
	Eigen::MatrixXf temp = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)* rot * bs.transpose();
	/*temp.row(0).array() *= ide[id_idx].s(exp_idx, 0);
	temp.row(1).array() *= ide[id_idx].s(exp_idx, 1);*/
	temp = temp.colwise() + tslt.block(0,0,2,1);
#endif // normalization

	
	std::cout << temp.transpose() << '\n';
	std::cout << ide[id_idx].land_2d.block(exp_idx*G_land_num, 0, G_land_num, 2) << '\n';
	system("pause");
}

void test_upt_slt_angle(
	float f, iden* ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &inner_land_cor) {
	float pi = acos(-1);
	float be = -pi / 2, en = pi / 2+0.1,step=pi/20;

#ifdef test_posit_by2dland
	FILE *fp;
	fp = fopen(cal_coef_2dland_name.c_str(), "w");
	fprintf(fp, "%d\n", (int)((en - be) / step + 1)*3);
	fclose(fp);
#endif // test_posit_by2dland
#ifdef test_updt_slt
	fp = fopen(test_updt_slt_path.c_str(), "w");
	fprintf(fp, "%d\n", (int)((en - be) / step + 1) * 3);
	fclose(fp);
	fp = fopen(test_updt_slt_2d_path.c_str(), "w");
	fprintf(fp, "%d\n", (int)((en - be) / step + 1) * 3);
	fclose(fp);
#endif // test_updt_slt



	Eigen::Vector3f angle;
	angle.setZero();


	int i = 1,which=0;
	for (float ag = be; ag < en; ag += step, i++) {
		angle(which) = ag;
		ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = get_r_from_angle_zyx(angle);

		Eigen::VectorXi out_land_cor(15);
		update_slt_dde(f, ide, bldshps, id_idx, exp_idx, slt_line, slt_point_rect, out_land_cor);
		ide[id_idx].land_cor.block(exp_idx, 0, 1, 15) = out_land_cor.transpose();
		//cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
#ifdef test_posit_by2dland
		test_2dland(f, ide, bldshps, id_idx, exp_idx);
#endif // test_posit_by2dland
	}
	printf("i %d\n", i);
	angle(which) = 0;
	which = 1;
	for (float ag = be; ag < en; ag += step, i++) {
		angle(which) = ag;
		ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = get_r_from_angle_zyx(angle);

		Eigen::VectorXi out_land_cor(15);
		update_slt_dde(f, ide, bldshps, id_idx, exp_idx, slt_line, slt_point_rect, out_land_cor);
		ide[id_idx].land_cor.block(exp_idx, 0, 1, 15) = out_land_cor.transpose();
		//cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
#ifdef test_posit_by2dland
		test_2dland(f, ide, bldshps, id_idx, exp_idx);
#endif // test_posit_by2dland
	}
	printf("i %d\n", i);
	angle(which) = 0;
	which = 2;
	for (float ag = be; ag < en; ag += step, i++) {
		angle(which) = ag;
		ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = get_r_from_angle_zyx(angle);

		Eigen::VectorXi out_land_cor(15);
		update_slt_dde(f, ide, bldshps, id_idx, exp_idx, slt_line, slt_point_rect, out_land_cor);
		ide[id_idx].land_cor.block(exp_idx, 0, 1, 15) = out_land_cor.transpose();
		//cal_rt_pnp(f, ide, bldshps, inner_land_cor, id_idx, exp_idx);
#ifdef test_posit_by2dland
		test_2dland(f, ide, bldshps, id_idx, exp_idx);
#endif // test_posit_by2dland
	}

	exit(99);
}


float cal_3dpaper_exp(
	float f, iden* ide, Eigen::MatrixXf &bldshps, 
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor) {

	puts("calculating expression coeffients by 3dpaper's way");
	float error = 0;
	Eigen::MatrixXf exp_point(G_nShape, 3 * G_land_num);
	
	cal_exp_point_matrix(ide, bldshps, id_idx, exp_idx,land_cor, exp_point);
	Eigen::RowVectorXf exp = ide[id_idx].exp.row(exp_idx);
	error=ceres_exp_one(f,ide, id_idx, exp_idx, exp_point, exp);
	ide[id_idx].exp.row(exp_idx) = exp;
	return error;
}
void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx,int exp_idx, Eigen::VectorXi &land_cor,
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
					if (i_shape==0)
						V(j) += ide[id_idx].user(i_id)*bldshps(i_id, land_cor(i_v) * 3 + j);
					else
						V(j) += ide[id_idx].user(i_id)*
						(bldshps(i_id, i_shape*G_nVerts * 3 + land_cor(i_v) * 3 + j)- bldshps(i_id, land_cor(i_v) * 3 + j));
			V = rot * V;
#ifdef normalization

			V.block(0,0,2,1) = S * V;
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
					if (i_shape==0) 
						V(j) += ide[id_idx].exp(exp_idx,i_shape)*bldshps(i_id,land_cor(i_v) * 3 + j);
					else
						V(j) += ide[id_idx].exp(exp_idx, i_shape)
						*(bldshps(i_id, i_shape*G_nVerts * 3 + land_cor(i_v) * 3 + j)- bldshps(i_id,land_cor(i_v) * 3 + j));
			V = rot * V;
#ifdef normalization
			V.block(0, 0, 2, 1) = S * V;
#endif // normalization
			for (int j = 0; j < 3; j++)
				result(i_id, i_v * 3 + j) = V(j);
		}

}

float cal_3dpaper_ide_admm(
	float f, iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::VectorXf &ide_sg_vl,float lmd) {

	puts("calculating identity coeffients by 3dpaper's way");
	float error = 0;
	Eigen::MatrixXf id_point(G_iden_num, 3 * G_land_num);

	cal_id_point_matrix(ide, bldshps, id_idx, exp_idx, land_cor, id_point);
	Eigen::VectorXf user = ide[id_idx].user;
	//error = admm_user_one(f, ide, id_idx, exp_idx, id_point, user, lmd);
	ide[id_idx].user = user;
	return error;
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
	fp=fopen( cal_coef_land_name.c_str(), "a");
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);
}

void test_coef_mesh(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_nVerts, 3);
	Eigen::Matrix3f R=ide[id_idx].rot.block(exp_idx*3,0,3,3);
#ifdef solve_cvpnp
	Eigen::Vector3f hm_angle=pnpR2humanA(R);
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
	fp=fopen( cal_coef_mesh_name.c_str(), "a");
	for (int i = 0; i < G_nVerts; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);
}

void test_2dland(float f, iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {
	Eigen::MatrixX3f land3d(G_land_num, 3);
//	puts("A");
	for (int i = 0; i < G_land_num; i++)
		for (int axis = 0; axis < 3; axis++)
			land3d(i, axis) =
			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx,i), axis);
//	puts("B");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx, 0, 3, 3);

//	puts("C");
	FILE *fp;
	fp=fopen( cal_coef_2dland_name.c_str(), "a");
	for (int i = 0; i < G_land_num; i++) {
		Eigen::Vector3f X = land3d.row(i).transpose();
#ifdef posit
		X = R * X + tslt;
		fprintf(fp, "%.6f %.6f\n", X(0)*f / X(2)+ide[id_idx].center(exp_idx,0), X(1)*f / X(2) + ide[id_idx].center(exp_idx, 1));
#endif // posit
#ifdef normalization
		X.block(0,0,2,1) = ide[id_idx].s.block(2 * exp_idx, 0, 2, 3)*R * X+ tslt.block(0,0,2,1);
		fprintf(fp, "%.6f %.6f\n",X(0),X(1));
#endif // normalization
	}

	fclose(fp);
}

void cal_mesh_land(Eigen::MatrixXf &bldshps) {
	puts("calculating mesh from test_coeff_ide&exp");
	FILE *fpr, *fpw;
	fpr=fopen( "./server/test_ide_coeff.txt", "r");
	Eigen::VectorXf ide(G_iden_num), exp(G_nShape);
	for (int i_id = 0; i_id < G_iden_num; i_id++) fscanf(fpr, "%f", &ide(i_id));
	fclose(fpr);
	fpr=fopen( "./server/test_exp_coeff.txt", "r");
	fpw=fopen( "./test_exp_coeff_mesh.txt", "w");
	int num = 3;
	fprintf(fpw, "%d\n", num);
	Eigen::MatrixXf mesh(G_nVerts, 3);
	//char s[500];
	for (int j_no = 0; j_no < num; j_no++) {
		printf("%d \n", j_no);
		//fscanf(fpr, "------------------------------------------");
		//puts(s);
		for (int i_exp = 0; i_exp < G_nShape; i_exp++)
			fscanf(fpr, " %f", &exp(i_exp)) , printf("%d %.6f\n", i_exp, exp(i_exp));
		std::cout << exp << "-----------------\n";
		system("pause");
		for (int i = 0; i < G_nVerts; i++)
			for (int axis = 0; axis < 3; axis++) {
				mesh(i, axis) = 0;
				for (int i_id = 0; i_id < G_iden_num; i_id++)
					for (int i_exp=0; i_exp < G_nShape; i_exp++)
						if (i_exp==0)
							mesh(i, axis) += bldshps(i_id, i * 3 + axis)*ide(i_id)*exp(i_exp);
						else
							mesh(i, axis) += (bldshps(i_id, i_exp*G_nVerts * 3 + i * 3 + axis)- bldshps(i_id, i * 3 + axis))
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
	fpr=fopen( "./server/test_exp_coeff_t23.txt", "r");
	fpw=fopen( "./test_exp_coeff_mesh.txt", "w");
	int num = 1;
	fprintf(fpw, "%d\n", num);
	Eigen::MatrixXf mesh(G_nVerts, 3);
	//char s[500];
	for (int j_no = 0; j_no < num; j_no++) {
		printf("%d \n", j_no);
		//fscanf(fpr, "------------------------------------------");
		//puts(s);
		for (int i_exp = 0; i_exp < G_nShape; i_exp++)
			fscanf(fpr, "%f", &exp(i_exp));
		for (int i = 0; i < G_nVerts; i++)
			for (int axis = 0; axis < 3; axis++) {
				mesh(i, axis) = 0;
				int i_id = 0;
				for (int i_exp = 0; i_exp < G_nShape; i_exp++)
					if (i_exp==0) 
						mesh(i, axis) += bldshps(i_id, i * 3 + axis)*exp(i_exp);
					else 
						mesh(i, axis) += 
						(bldshps(i_id, i_exp*G_nVerts * 3 + i * 3 + axis)- bldshps(i_id,i * 3 + axis))*exp(i_exp);
			}
		for (int i = 0; i < G_nVerts; i++)
			fprintf(fpw, "%.6f %.6f %.6f \n", mesh(i, 0), mesh(i, 1), mesh(i, 2));
	}
	fclose(fpw);
}
Eigen::Vector3f get_uler_angle_zyx(Eigen::Matrix3f R) {
	Eigen::Vector3f x, y, z, t;
	x = R.row(0).transpose();
	y = R.row(1).transpose();
	z = R.row(2).transpose();
	float al, be, ga;
	if (fabs(1 - x(2)*x(2)) < 1e-3) {
		be = asin(x(2));
		al = ga = 0;
		//exit(1);
	}
	else {

		be = asin(max(min(1.0, double(x(2))), -1.0));
		al = asin(max(min(1.0, double(-x(1) / sqrt(1 - x(2)*x(2)))), -1.0));
		ga = asin(max(min(1.0, double(-y(2) / sqrt(1 - x(2)*x(2)))), -1.0));

	}
	//std::cout << R << "\n----------------------\n";
	//printf("%.10f %.10f %.10f %.10f\n", x(2), al / pi * 180, be / pi * 180, ga / pi * 180);
	Eigen::Vector3f ans;
	ans << al, be, ga;
	return ans;
	//system("pause");
}

Eigen::Matrix3f get_r_from_angle_zyx(const Eigen::Vector3f &angle) {
	Eigen::Matrix3f ans;
	float Sa = sin(angle(0)), Ca = cos(angle(0)), Sb = sin(angle(1)),
		Cb = cos(angle(1)), Sc = sin(angle(2)), Cc = cos(angle(2));

	ans(0, 0) = Ca * Cb;
	ans(0, 1) = -Sa * Cb;
	ans(0, 2) = Sb;
	ans(1, 0) = Sa * Cc + Ca * Sb*Sc;
	ans(1, 1) = Ca * Cc - Sa * Sb*Sc;
	ans(1, 2) = -Cb * Sc;
	ans(2, 0) = Sa * Sc - Ca * Sb*Cc;
	ans(2, 1) = Ca * Sc + Sa * Sb*Cc;
	ans(2, 2) = Cb * Cc;
	return ans;
}

Eigen::Vector3f pnpR2humanA(Eigen::Matrix3f R) {
	Eigen::Vector3f angle = get_uler_angle_zyx(R);
	//(pi-angle)-pi   pi-angle~pi/2--pi*3/2
	Eigen::Vector3f ans(-angle(0),-angle(1),angle(2));	
	return ans;
	//system("pause");
}
void update_inner_land_cor(float f,iden *ide,int id_idx,int exp_idx,Eigen::VectorXi &inner_cor,Eigen::MatrixXf &bldshps) {
	inner_cor.resize(G_inner_land_num);
	Eigen::VectorXf mi_inner(G_inner_land_num);
	for (int i = 0; i < G_inner_land_num; i++) mi_inner(i) = 1e8;
	//Eigen::MatrixX3f bs(G_nVerts, 3);
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
#ifdef solve_cvpnp
	Eigen::Vector3f hm_angle = pnpR2humanA(R);
	R = get_r_from_angle_zyx(hm_angle);
#endif // solve_cvpnp
	std::cout << R << "\n";
	Eigen::Vector3f angle = get_uler_angle_zyx(R);
	std::cout << "angle: "<<angle.transpose() << "\n";
	//system("pause");
	for (int i = 0; i < G_nVerts; i++) {
		Eigen::Vector3f v;
		v.setZero();
		for (int axis = 0; axis < 3; axis++)
			v(axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, i, axis);
		if (v(2) < 0.1) continue;
		v = R*v+tslt;
		Eigen::RowVector2f ld;
		ld(0) = v(0)*f / v(2) + ide[id_idx].center(exp_idx, 0);
		ld(1) = v(1)*f / v(2) + ide[id_idx].center(exp_idx, 1);
		for (int j=0;j<G_inner_land_num;j++)
			if ((ld - ide[id_idx].land_2d.row(exp_idx*G_land_num + 15 + j)).norm() < mi_inner(j)) {
				mi_inner(j) = (ld - ide[id_idx].land_2d.row(exp_idx*G_land_num + 15 + j)).norm();
				inner_cor(j) = i;
			}
	}

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