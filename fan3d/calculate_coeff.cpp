#include "calculate_coeff.h"

#define test_coef_save_mesh
//#define test_posit_by2dland


void init_exp_ide_r_t_pq(iden *ide, int ide_num,Eigen::VectorXi &land_cor) {

	puts("initializing coeffients(R,t,pq)...");
	for (int i = 0; i < ide_num; i++) {
#ifdef fan3d
		ide[i].center_3d.resize(ide[i].num, 3);
		ide[i].center_3d.setZero();
		for (int j = 0; j < ide[i].num; j++)
			for (int k = 0; k < G_land_num; k++)
				ide[i].center_3d.row(j) += ide[i].land_3d.row(j*G_land_num + k);
		ide[i].center_3d.array() /= G_land_num;
		for (int j = 0; j < ide[i].num; j++) {
			float eye_dis = (ide[i].land_3d.row(j*G_land_num + G_left_idx)
				- ide[i].land_3d.row(j*G_land_num + G_right_idx)).norm();
			for (int k = 0; k < G_land_num; k++) {
				ide[i].land_3d.row(j*G_land_num + k) -= ide[i].center_3d.row(j);
				ide[i].land_3d.row(j*G_land_num + k).array() /= eye_dis;
			}
		}
#else
		ide[i].center.resize(ide[i].num,2);
		ide[i].center.setZero();
		for (int j = 0; j < ide[i].num; j++)
			for (int k = 0; k < G_land_num; k++)
				ide[i].center.row(j) += ide[i].land_2d.row(j*G_land_num + k);
		ide[i].center.array() /= G_land_num;
		for (int j = 0; j < ide[i].num; j++)
			for (int k = 0; k < G_land_num; k++)
				ide[i].land_2d.row(j*G_land_num + k) -= ide[i].center.row(j);
#endif // fan3d

		ide[i].exp.resize(ide[i].num, G_nShape);
		ide[i].land_cor.resize(ide[i].num, G_land_num);
		ide[i].land_cor.rowwise()=land_cor.transpose();
		ide[i].user.resize(G_iden_num);
		ide[i].rot.resize(3 * ide[i].num,3);
		ide[i].tslt.resize(ide[i].num,3);
		
		ide[i].dis.resize(G_land_num*ide[i].num, 2);
#ifdef normalization
		ide[i].s.resize(ide[i].num*2, 3);
#endif // normalization

		
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

//void cal_f(
//	iden *ide, Eigen::MatrixXf &bldshps,Eigen::VectorXi &inner_land_corr, Eigen::VectorXi &jaw_land_corr,
//	std :: vector<int> *slt_line, std::vector<std::pair<int,int> > *slt_point_rect, 
//	Eigen::VectorXf &ide_sg_vl) {
//
//	puts("calclating focus for each image...");
//	FILE *fp;
//#ifdef test_coef
//	fopen_s(&fp, "test_coef_f_loss.txt", "w");
//#else
//	fopen_s(&fp, "test_coef_ide_focus.txt", "w");	
//#endif // !test_coef
//
//
//	
//	for (int i_id = 0; i_id < G_train_pic_id_num; i_id++) {
//		if (ide[i_id].num == 0)continue;
//#ifndef test_coef
//		float L = 100, R = 1000, er_L, er_R;
//		er_L = pre_cal_exp_ide_R_t(L, ide, bldshps, inner_land_corr, jaw_land_corr,
//			slt_line, slt_point_rect, i_id,ide_sg_vl);
//		er_R = pre_cal_exp_ide_R_t(R, ide, bldshps, inner_land_corr, jaw_land_corr,
//			slt_line, slt_point_rect, i_id, ide_sg_vl);
//		fprintf(fp, "-------------------------------------------------\n");
//		while (R-L>20) {
//			printf("cal f %.5f %.5f %.5f %.5f\n", L, er_L, R, er_R);
//			fprintf(fp, "cal f %.5f %.5f %.5f %.5f\n", L, er_L, R, er_R);
//			float mid_l, mid_r, er_mid_l, er_mid_r;
//			mid_l = L * 2 / 3 + R / 3;
//			mid_r = L / 3 + R * 2 / 3;
//			er_mid_l = pre_cal_exp_ide_R_t(mid_l, ide, bldshps, inner_land_corr, jaw_land_corr,
//				slt_line, slt_point_rect, i_id, ide_sg_vl);
//			er_mid_r = pre_cal_exp_ide_R_t(mid_r, ide, bldshps, inner_land_corr, jaw_land_corr, 
//				slt_line, slt_point_rect, i_id, ide_sg_vl);
//			if (er_mid_l < er_mid_r) R = mid_r, er_R = er_mid_r;
//			else L = mid_l, er_L = er_mid_l;
//		}
//		ide[i_id].fcs = (L+R)/2;
//#endif // !test_coef
//		
//		/*FILE *fp;
//		fopen_s(&fp, "test_f.txt", "w");*/
//		
//#ifdef test_coef
//		int st = 400, en = 410, step = 25;
//		Eigen::VectorXf temp((en-st)/step+1);
//		for (int i = st; i < en; i += step) temp((i-st)/step) = 
//			pre_cal_exp_ide_R_t(i, ide, bldshps, inner_land_corr, jaw_land_corr,
//				slt_line, slt_point_rect, i_id, ide_sg_vl);
//		for (int i = 0; i < (en - st) / step + 1; i++) {
//			printf("test cal f %d %.6f\n", st + i * step, temp(i));
//			fprintf(fp, "%d %.6f\n", st + i * step, temp(i));
//		}
//#endif
//
//	}
//	fclose(fp);
//	//FILE *fp;
//	fopen_s(&fp, "test_ide_coeff.txt", "w");
//	for (int i = 0; i < G_iden_num; i++)
//		fprintf(fp, "%.6f\n", ide[0].user(i));
//	fclose(fp);
//	fopen_s(&fp, "test_exp_coeff.txt", "w");
//	for (int i_exp = 0; i_exp < ide[0].num; i_exp++) {
//		fprintf(fp, "\n");
//		for (int i = 0; i < G_nShape; i++)
//			fprintf(fp, "%.6f\n", ide[0].exp(i_exp, i));
//	}
//	fclose(fp);
//
//}



void solve_3d(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &land_corr, Eigen::VectorXf &ide_sg_vl) {

	puts("calclating 3d coeffients begin...");
	FILE *fp;
	for (int i_id = 0; i_id < G_train_pic_id_num; i_id++) {
		if (ide[i_id].num == 0)continue;
		cal_exp_ide_R_3d(ide, bldshps, land_corr, i_id, ide_sg_vl);
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

float cal_exp_ide_R_3d(
	iden *ide, Eigen::MatrixXf &bldshps, Eigen::VectorXi &land_cor, int id_idx,
	Eigen::VectorXf &ide_sg_vl) {

	puts("preparing 3d expression & other coeffients...");
	init_exp_ide(ide, G_train_pic_id_num, id_idx);
	float error = 0;

	int tot_r = 4;
	Eigen::VectorXf temp(tot_r);
	//fprintf(fp, "%d\n",tot_r);
	FILE *fp;
#ifdef test_coef_save_mesh
	fopen_s(&fp, cal_coef_land_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r)*ide[id_idx].num);
	fclose(fp);

	fopen_s(&fp, cal_coef_mesh_name.c_str(), "w");
	fprintf(fp, "%d\n", (tot_r)*ide[id_idx].num);
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
			cal_3d_R_t(ide, bldshps, land_cor, id_idx, i_exp);
			test_cal_3d_R_t(ide, bldshps, land_cor, id_idx, i_exp);
			error = cal_3d_exp(ide, bldshps, id_idx, i_exp, land_cor);
			error = cal_3d_ide(ide, bldshps, id_idx, i_exp, land_cor, ide_sg_vl);
		}
		//error = cal_fixed_exp_same_ide(ide, bldshps, id_idx, ide_sg_vl);

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

void init_exp_ide(iden *ide, int train_id_num,int id_idx) {

	puts("initializing coeffients(identitiy,expression) for cal f...");
	ide[id_idx].exp= Eigen::MatrixXf::Constant(ide[id_idx].num, G_nShape, 1.0 / G_nShape);
	for (int j = 0; j < ide[id_idx].num; j++) ide[id_idx].exp(j, 0) = 1;
	ide[id_idx].user= Eigen::MatrixXf::Constant(G_iden_num, 1, 1.0 / G_iden_num);
	
}

void cal_3d_R_t(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx) {

	puts("normalization...");
	Eigen::MatrixX3f land_3d; Eigen::MatrixX3f land_bs;
	land_3d.resize(G_land_num, 3);
	land_3d = ide[id_idx].land_3d.block(exp_idx*G_land_num, 0, G_land_num, 3);
	
	
	land_bs.resize(G_land_num, 3);
	for (int i = 0; i < G_land_num; i++)
		for (int axis = 0; axis < 3; axis++)
			land_bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, land_cor(i), axis);


	Eigen::RowVector3f center = land_bs.colwise().mean();
	//std::cout << "\n----ceneter_3d----\n" << center_3d << "-------------\n";
	land_bs = land_bs.rowwise() - center;
	float eye_dis = (land_bs.row(G_left_idx) - land_bs.row(G_right_idx)).norm();
	land_bs.array() /= eye_dis;

	//std::cout << "\n----ceneter_2d----\n" << center_2d << "-------------\n";
	//std::cout << bs_in << '\n';
	//puts("A");
	Eigen::Matrix3f A_= land_3d.transpose()*land_bs;

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A_, Eigen::ComputeFullV | Eigen::ComputeFullU);
	Eigen::Matrix3f R = svd.matrixU()*(svd.matrixV().transpose());
	//puts("A");
	ide[id_idx].rot.block(3 * exp_idx, 0, 3, 3) = R;

}
void test_cal_3d_R_t(
	iden *ide, Eigen::MatrixXf &bldshps,
	Eigen::VectorXi &land_cor, int id_idx, int exp_idx) {

	puts("testing normalization...");
	Eigen::MatrixX3f land_bs(G_land_num, 3);
	land_bs.resize(G_land_num, 3);
	for (int i = 0; i < G_land_num; i++)
		for (int axis = 0; axis < 3; axis++)
			land_bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, land_cor(i), axis);
	Eigen::RowVector3f center = land_bs.colwise().mean();
	//std::cout << "\n----ceneter_3d----\n" << center_3d << "-------------\n";
	land_bs = land_bs.rowwise() - center;
	float eye_dis = (land_bs.row(G_left_idx) - land_bs.row(G_right_idx)).norm();
	land_bs.array() /= eye_dis;
	puts("aa");
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	puts("aabb");
	std::cout << "\n----rotation----\n" << rot << "-------------\n";
	Eigen::Vector3f angle = get_uler_angle_zyx(rot);
	std::cout << "\n----angle----\n" << angle << "-------------\n";
	Eigen::MatrixXf temp = (rot * land_bs.transpose());

	std::cout << temp << "+++++++\n";
	std::cout <<
		ide[id_idx].land_3d.block(exp_idx*G_land_num,0,G_land_num,3).transpose() << "pppppppppppp-\n";
	system("pause");
}

float cal_3d_vtx(
	iden *ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, int vtx_idx, int axis) {

	//puts("calculating one vertex coordinate...");
	float ans = 0;

	for (int i_id = 0; i_id < G_iden_num; i_id++)
		for (int i_shape = 0; i_shape < G_nShape; i_shape++)
			if (i_shape==0)
				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
				*bldshps(i_id, vtx_idx * 3 + axis);
			else
				ans += ide[id_idx].exp(exp_idx, i_shape)*ide[id_idx].user(i_id)
				*(bldshps(i_id, 3 * G_nVerts*i_shape + vtx_idx * 3 + axis) - bldshps(i_id, vtx_idx * 3 + axis));
	return ans;
}


float cal_3d_exp(
	iden* ide, Eigen::MatrixXf &bldshps, 
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor) {

	puts("calculating expression coeffients by 3dpaper's way");
	float error = 0;
	Eigen::MatrixXf exp_point(G_nShape, 3 * G_land_num);
	
	cal_exp_point_matrix(ide, bldshps, id_idx, exp_idx,land_cor, exp_point);
	Eigen::RowVectorXf exp = ide[id_idx].exp.row(exp_idx);
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	error=ceres_exp_one(ide, id_idx, exp_idx, exp_point, exp,rot);
	ide[id_idx].exp.row(exp_idx) = exp;
	return error;
}
void cal_exp_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx,int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result) {

	puts("prepare exp_point matrix for bfgs/ceres...");
	result.resize(G_nShape, 3 * G_land_num);

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

			for (int j = 0; j < 3; j++)
				result(i_shape, i_v * 3 + j) = V(j);
		}

}

float cal_3d_ide(
	iden* ide, Eigen::MatrixXf &bldshps,
	int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::VectorXf &ide_sg_vl) {

	puts("calculating identity coeffients by 3dpaper's way");
	float error = 0;
	Eigen::MatrixXf id_point(G_iden_num, 3 * G_land_num);

	cal_id_point_matrix(ide, bldshps, id_idx, exp_idx, land_cor, id_point);
	Eigen::VectorXf user = ide[id_idx].user;
	Eigen::Matrix3f rot = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	error = ceres_user_one(ide, id_idx, exp_idx, id_point, user, rot, ide_sg_vl);
	ide[id_idx].user = user;
	return error;
}
void cal_id_point_matrix(
	iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx, Eigen::VectorXi &land_cor,
	Eigen::MatrixXf &result) {

	puts("prepare user_point matrix for bfgs/ceres...");
	result.resize(G_iden_num, 3 * G_land_num);

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

			for (int j = 0; j < 3; j++)
				result(i_id, i_v * 3 + j) = V(j);
		}

}


//float cal_fixed_exp_same_ide(iden *ide, Eigen::MatrixXf &bldshps, int id_idx,
//	Eigen::VectorXf &ide_sg_vl) {
//
//	puts("calculating identity coeffients by 3dpaper's way while fixing the expression coeffients");
//	float error = 0;
//	Eigen::MatrixXf id_point_fix_exp(ide[id_idx].num*G_iden_num,G_land_num*3);
//
//	for (int i_exp = 0; i_exp < ide[id_idx].num; i_exp++) {
//		Eigen::VectorXi land_cor(G_land_num);
//		Eigen::MatrixXf id_point(G_iden_num, 3 * G_land_num);
//		land_cor = ide[id_idx].land_cor.row(i_exp);
//		cal_id_point_matrix(ide, bldshps, id_idx, i_exp, land_cor, id_point);
//		
//		id_point_fix_exp.block(i_exp*G_iden_num, 0, G_iden_num, G_land_num * 3)= id_point;
//	}
//	Eigen::VectorXf user = ide[id_idx].user;
//	error = ceres_user_fixed_exp(ide, id_idx, id_point_fix_exp, user, ide_sg_vl);
//	ide[id_idx].user = user;
//	return error;
//}

void test_coef_land(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {

	Eigen::MatrixX3f bs(G_land_num, 3);
	
	for (int i = 0; i < G_land_num; i++) {
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), axis);		
	}
	Eigen::RowVector3f center = bs.colwise().mean();
	//std::cout << "\n----ceneter_3d----\n" << center_3d << "-------------\n";
	
	bs = bs.rowwise() - center;
	float eye_dis = (bs.row(G_left_idx) - bs.row(G_right_idx)).norm();
	bs.array() /= eye_dis;
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	bs = (R*(bs.transpose())).transpose();
	FILE *fp;
	fopen_s(&fp, cal_coef_land_name.c_str(), "a");
	for (int i = 0; i < G_land_num; i++)
		fprintf(fp, "%.6f %.6f %.6f \n", bs(i, 0), bs(i, 1), bs(i, 2));
	fclose(fp);
}

void test_coef_mesh(iden *ide, Eigen::MatrixXf &bldshps, int id_idx, int exp_idx) {
	Eigen::MatrixX3f bs_land(G_land_num, 3);

	for (int i = 0; i < G_land_num; i++) {
		for (int axis = 0; axis < 3; axis++)
			bs_land(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx, i), axis);
	}
	Eigen::RowVector3f center = bs_land.colwise().mean();
	//std::cout << "\n----ceneter_3d----\n" << center_3d << "-------------\n";
	bs_land = bs_land.rowwise() - center;
	float eye_dis = (bs_land.row(G_left_idx) - bs_land.row(G_right_idx)).norm();
	Eigen::MatrixX3f bs(G_nVerts, 3);
	
	for (int i = 0; i < G_nVerts; i++) {
		for (int axis = 0; axis < 3; axis++)
			bs(i, axis) = cal_3d_vtx(ide, bldshps, id_idx, exp_idx, i, axis);
	}
	bs.rowwise() -= center;
	bs.array() /= eye_dis;
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx * 3, 0, 3, 3);
	bs = (R*bs.transpose()).transpose();
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
			cal_3d_vtx(ide, bldshps, id_idx, exp_idx, ide[id_idx].land_cor(exp_idx,i), axis);
	puts("B");
	Eigen::Vector3f tslt = ide[id_idx].tslt.row(exp_idx).transpose();
	Eigen::Matrix3f R = ide[id_idx].rot.block(exp_idx, 0, 3, 3);

	puts("C");
	FILE *fp;
	fopen_s(&fp, cal_eoef_2dland_name.c_str(), "a");
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
			fscanf_s(fpr, " %f", &exp(i_exp)) , printf("%d %.6f\n", i_exp, exp(i_exp));
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
		exit(1);
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