#include "dde_x.hpp"
//#define intersection

#include <algorithm>
#include <stdexcept>
//#define update_slt_def
//#define no_nearest_me
#define debug_init
#ifdef debug_init
	cv::VideoWriter debug_init_output_video("debug_init.avi", CV_FOURCC_DEFAULT, 25.0, cv::Size(640, 480));
#endif // debug_init



using namespace std;

DDEX::DDEX(const string & filename)
{
	cv::FileStorage model_file;
	model_file.open(filename, cv::FileStorage::READ);
	if (!model_file.isOpened())
		throw runtime_error("Cannot open model file \"" + filename + "\".");
	model_file["ref_shape"] >> ref_shape_;
	model_file["tri_idx_save"] >> tri_idx_;
	cv::FileNode fn;/*= model_file["test_init_shapes"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		vector<cv::Point2d> shape;
		*it >> shape;
		test_init_shapes_.push_back(shape);
	}*/
	fn = model_file["stage_regressors"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		regressor_dde r;
		*it >> r;
		stage_regressors_dde_.push_back(r);
	}
}


void change_nearest(DataPoint &data, std::vector<DataPoint> &train_data) {
	puts("change nearest");
	float mi_land = 100000000;
	int idx = 0;
	data.centeroid = data.land_2d.colwise().mean();
	for (int i = 0; i < train_data.size(); i++) {
		//float distance = (data.center - train_data[i].center).norm();
		float distance_land = 
			((data.land_2d.rowwise()-data.centeroid) - (train_data[i].land_2d.rowwise()-train_data[i].centeroid)).squaredNorm();
		if (distance_land < mi_land) {
			idx = i;
			mi_land = distance_land;
			printf("idx %d dis%.10f center :%.10f %.10f train_center :%.10f %.10f\n",
				idx,mi_land,data.centeroid(0), data.centeroid(1), train_data[i].centeroid(0), train_data[i].centeroid(1));
		}
	}
/*	printf("nearest vertex: %d train dataset size: %d\n", idx,train_data.size());
	printf("mi_land: %.5f\n",mi_land);
	Eigen::MatrixX2f dis=(data.land_2d.rowwise()-data.centeroid) - (train_data[idx].land_2d.rowwise()-train_data[idx].centeroid);
	show_dis_part(dis);
	exit(9);
	if (mi_land>2000) return;*/
	//data.shape.rot = train_data[idx].shape.rot;
	Eigen::Vector3f yuan_angle = data.shape.angle;
	data.shape.angle = train_data[idx].shape.angle;
	data.land_cor = train_data[idx].land_cor;
	data.shape.land_cor = train_data[idx].land_cor;
	FILE *fp;
	fp=fopen( "debug_ite_nearest_idx.txt", "a");
	fprintf(fp, "%d %.10f\n", idx,mi_land);
	fprintf(fp, "%.3f %.3f\n", yuan_angle(1), data.shape.angle(1));
	fclose(fp);

}

void change_nearest_slt(DataPoint &data, std::vector<DataPoint> &train_data) {
	puts("change nearest");
	float mi_land = 100000000;
	int idx = 0;
	data.centeroid = data.land_2d.colwise().mean();
	for (int i = 0; i < train_data.size(); i++) {
		//float distance = (data.center - train_data[i].center).norm();
		/*Eigen::MatrixXf tr_d(G_outer_land_num + 1, 2), da_d(G_outer_land_num + 1, 2);
		//Eigen::MatrixXf tr_d(G_outer_land_num + 11+1, 2), da_d(G_outer_land_num + 11+1, 2);

		tr_d.block(0, 0, G_outer_land_num, 2) = train_data[i].land_2d.block(0, 0, G_outer_land_num, 2);
		da_d.block(0, 0, G_outer_land_num, 2) = data.land_2d.block(0, 0, G_outer_land_num, 2);

		tr_d.row(G_outer_land_num) = train_data[i].land_2d.row(64);
		da_d.row(G_outer_land_num) = data.land_2d.row(64);

		
		tr_d.block(G_outer_land_num + 1, 0, 11, 2) = train_data[i].land_2d.block(35, 0, 11, 2);
		da_d.block(G_outer_land_num + 1, 0, 11, 2) = data.land_2d.block(35, 0, 11, 2);
		*/
		Eigen::MatrixXf tr_d(11+1+12, 2), da_d(11+1+12, 2);

		tr_d.row(0) = train_data[i].land_2d.row(64);
		da_d.row(0) = data.land_2d.row(64);

		
		tr_d.block(1, 0, 11, 2) = train_data[i].land_2d.block(35, 0, 11, 2);
		da_d.block(1, 0, 11, 2) = data.land_2d.block(35, 0, 11, 2);
		
		tr_d.block(12,0,12,2)= train_data[i].land_2d.block(15, 0, 12, 2);
		da_d.block(12, 0, 12, 2) = train_data[i].land_2d.block(15, 0, 12, 2);
		
		tr_d.rowwise() -= tr_d.colwise().mean();
		da_d.rowwise() -= da_d.colwise().mean();

		float distance_land = sqrt((tr_d - da_d).squaredNorm() / (G_outer_land_num + 1) / 2);
		//float distance_land = sqrt((tr_d - da_d).squaredNorm() / (G_outer_land_num + 11 + 1) / 2);

		if (distance_land < mi_land) {
			idx = i;
			mi_land = distance_land;
			printf("idx %d dis%.10f\n",idx, mi_land);
		}
	}
	printf("nearest vertex: %d train dataset size: %d\n", idx, train_data.size());
	//data.shape.rot = train_data[idx].shape.rot;
	if (mi_land > 3) return;
	Eigen::Vector3f yuan_angle = data.shape.angle;
	data.shape.angle = train_data[idx].shape.angle;
	data.land_cor = train_data[idx].land_cor;
	data.shape.land_cor= train_data[idx].land_cor;
	FILE *fp;
	fp=fopen("debug_ite_nearest_idx_slt.txt", "a");
	fprintf(fp, "%d %.10f\n", idx, mi_land);
	fprintf(fp, "%.3f %.3f\n", yuan_angle(1), data.shape.angle(1));
	fclose(fp);

}

std::vector<pair<int,float> > init_shape_la(G_dde_K);
int flag_init_shape = 0;
int cmp_init_shape(pair<int, float> x, pair<int, float> y) {
	return x.second < y.second;
}
void get_init_shape(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {

	puts("calculating initial shape");
	float mi_land[G_dde_K]; int idx[G_dde_K];
	for (int i = 0; i < G_dde_K; i++) mi_land[i] = 100000000, idx[i] = -1;
	if (flag_init_shape) {
		for (int i = 0; i < G_dde_K; i++) init_shape_la[i].second =
			((data.land_2d.rowwise() - data.centeroid) -
			(train_data[init_shape_la[i].first].land_2d.rowwise() - train_data[init_shape_la[i].first].centeroid)).norm();
		std::sort(init_shape_la.begin(), init_shape_la.end(), cmp_init_shape);
		for (int i = 0; i < G_dde_K; i++) mi_land[i] = init_shape_la[i].second, idx[i] = init_shape_la[i].first;
	}
	for (int t = 0; t < G_dde_K; t++)
		printf("%d ", idx[t]);
	puts("");
	for (int t = 0; t < G_dde_K; t++)
		printf("%.5f ", mi_land[t]);
	puts("");
	for (int i = 0; i < train_data.size(); i++) {
		bool fl = 0;
		for (int j = 0; j < G_dde_K; j++)
			if (i == idx[j]) fl = 1;
		if (fl) continue;
		float distance_land = ((data.land_2d - train_data[i].land_2d).rowwise()-(data.centeroid -train_data[i].centeroid)).norm();
		//for (int j=0;j<G_dde_K;j++)
		//	if (di+data.shape.dis(i_v,0);stance_land < mi_land[j]) {
		//		for (int k = G_dde_K - 1; k > j; k--) {
		//			idx[k] = idx[k - 1];
		//			mi_land[k] = mi_land[k - 1];
		//		}
		//		idx[j] = i;
		//		mi_land[j] = distance_land;
		//		for (int t = 0; t < G_dde_K; t++)
		//			printf("%d ", idx[t]);
		//		puts("");
		//		break;
		//	}
		for (int j = G_dde_K - 1; j >= 0; j--) {
			if (distance_land > mi_land[j]) {
				j++;
				if (j == G_dde_K) break;
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("+++");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("++++");
				break;
			}
			if (j == 0) {
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("");
				break;
			}
		}
	}
	for (int i = 0; i < G_dde_K; i++) {
		ans[i] = train_data[idx[i]].shape;
		init_shape_la[i].first = idx[i];
		init_shape_la[i].second = mi_land[i];
		
/*		printf("%d ",idx[i]);
		if (mi_land[i]>50){
			ans[i] = data.shape;
			continue;
		}*/
#ifdef normalization
//align the center by tslt
		ans[i].tslt.block(0, 0, 1, 2) -= train_data[idx[i]].centeroid;
		ans[i].tslt.block(0, 0, 1, 2) += data.centeroid;
#endif // normalization
#ifdef perspective
//		ans[i].tslt = data.shape.tslt;
		ans[i].tslt.block(0, 0, 2, 1) -=
			((train_data[idx[i]].centeroid - data.centeroid) / train_data[idx[i]].fcs*ans[i].tslt(2)).transpose();
		ans[i].tslt(2) = data.shape.tslt(2);
#endif

	}
	puts("");
	flag_init_shape = 1;
	//system("pause");
}

void get_init_shape_inner(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {

	puts("calculating initial shape");
	float mi_land[G_dde_K]; int idx[G_dde_K];
	for (int i = 0; i < G_dde_K; i++) mi_land[i] = 100000000, idx[i] = -1;
	if (flag_init_shape) {
		for (int i = 0; i < G_dde_K; i++) init_shape_la[i].second =
			((data.land_2d.block(15,0,G_inner_land_num,2).rowwise() - data.centeroid) -
			(train_data[init_shape_la[i].first].land_2d.block(15, 0, G_inner_land_num, 2).rowwise() - train_data[init_shape_la[i].first].centeroid)).norm();
		std::sort(init_shape_la.begin(), init_shape_la.end(), cmp_init_shape);
		for (int i = 0; i < G_dde_K; i++) mi_land[i] = init_shape_la[i].second, idx[i] = init_shape_la[i].first;
	}
	for (int t = 0; t < G_dde_K; t++)
		printf("%d ", idx[t]);
	puts("");
	for (int t = 0; t < G_dde_K; t++)
		printf("%.5f ", mi_land[t]);
	puts("");
	for (int i = 0; i < train_data.size(); i++) {
		bool fl = 0;
		for (int j = 0; j < G_dde_K; j++)
			if (i == idx[j]) fl = 1;
		if (fl) continue;
		float distance_land = 
			((data.land_2d.block(15, 0, G_inner_land_num, 2) - train_data[i].land_2d.block(15, 0, G_inner_land_num, 2)).rowwise()
			- (data.centeroid - train_data[i].centeroid)).norm();
		//for (int j=0;j<G_dde_K;j++)
		//	if (distance_land < mi_land[j]) {
		//		for (int k = G_dde_K - 1; k > j; k--) {
		//			idx[k] = idx[k - 1];
		//			mi_land[k] = mi_land[k - 1];
		//		}
		//		idx[j] = i;
		//		mi_land[j] = distance_land;
		//		for (int t = 0; t < G_dde_K; t++)
		//			printf("%d ", idx[t]);
		//		puts("");
		//		break;
		//	}
		for (int j = G_dde_K - 1; j >= 0; j--) {
			if (distance_land > mi_land[j]) {
				j++;
				if (j == G_dde_K) break;
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("+++");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("++++");
				break;
			}
			if (j == 0) {
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("");
				break;
			}
		}
	}
	for (int i = 0; i < G_dde_K; i++) {
		ans[i] = train_data[idx[i]].shape;

#ifdef normalization
//align the center by tslt
		ans[i].tslt.block(0, 0, 1, 2) -= train_data[idx[i]].centeroid;
		ans[i].tslt.block(0, 0, 1, 2) += data.centeroid;
#endif // normalization
#ifdef perspective
//		ans[i].tslt = data.shape.tslt;
/*		ans[i].tslt(2) = data.shape.tslt(2);
		ans[i].tslt.block(0, 0, 2, 1) -=
			((train_data[idx[i]].centeroid - data.centeroid) / data.fcs*ans[i].tslt(2)).transpose();*/
		ans[i].tslt=data.shape.tslt;
		ans[i].angle=data.shape.angle;
		
#endif


		init_shape_la[i].first = idx[i];
		printf("%d ", idx[i]);
	}
	puts("");
	flag_init_shape = 1;
	//system("pause");
}



void get_init_shape_rand(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {

	puts("calculating initial shape");
	
	for (int i = 0; i < G_dde_K; i++) {
		int rnd= cv::theRNG().uniform(0, train_data.size());
		ans[i] = data.shape;
		ans[i].exp = train_data[rnd].shape.exp;

		//align the center by tslt
		ans[i].tslt.block(0, 0, 1, 2) -= train_data[rnd].center;
		ans[i].tslt.block(0, 0, 1, 2) += data.center;

		printf("%d ", rnd);
	}
	puts("");
	flag_init_shape = 1;
	//system("pause");
}

void get_init_shape_exp(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {

	puts("calculating initial shape");
	float mi_land[G_dde_K]; int idx[G_dde_K];
	for (int i = 0; i < G_dde_K; i++) mi_land[i] = 100000000, idx[i] = -1;
	if (flag_init_shape) {
		for (int i = 0; i < G_dde_K; i++) {
			init_shape_la[i].second = 0;
			for (int p=1;p<G_nShape;p++)
				init_shape_la[i].second += (data.shape.exp(p) - train_data[init_shape_la[i].first].shape.exp(p))*
										   (data.shape.exp(p) - train_data[init_shape_la[i].first].shape.exp(p));
		}
		std::sort(init_shape_la.begin(), init_shape_la.end(), cmp_init_shape);
		for (int i = 0; i < G_dde_K; i++) mi_land[i] = init_shape_la[i].second, idx[i] = init_shape_la[i].first;
	}
	for (int t = 0; t < G_dde_K; t++)
		printf("%d ", idx[t]);
	puts("");
	for (int t = 0; t < G_dde_K; t++)
		printf("%.5f ", mi_land[t]);
	puts("");
	for (int i = 0; i < train_data.size(); i++) {
		bool fl = 0;
		for (int j = 0; j < G_dde_K; j++)
			if (i == idx[j]) fl = 1;
		if (fl) continue;
		float distance_land = 0;
		for (int p = 1; p < G_nShape; p++)
			distance_land += (data.shape.exp(p) - train_data[i].shape.exp(p))*
							 (data.shape.exp(p) - train_data[i].shape.exp(p));
		//for (int j=0;j<G_dde_K;j++)
		//	if (distance_land < mi_land[j]) {
		//		for (int k = G_dde_K - 1; k > j; k--) {
		//			idx[k] = idx[k - 1];
		//			mi_land[k] = mi_land[k - 1];
		//		}
		//		idx[j] = i;
		//		mi_land[j] = distance_land;
		//		for (int t = 0; t < G_dde_K; t++)
		//			printf("%d ", idx[t]);
		//		puts("");
		//		break;
		//	}
		for (int j = G_dde_K - 1; j >= 0; j--) {
			if (distance_land > mi_land[j]) {
				j++;
				if (j == G_dde_K) break;
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("+++");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("++++");
				break;
			}
			if (j == 0) {
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("");
				break;
			}
		}
	}
	for (int i = 0; i < G_dde_K; i++) {
		//ans[i] = data.shape;
		ans[i] = train_data[idx[i]].shape;

		init_shape_la[i].first = idx[i];
		printf("%d ", idx[i]);
	}
	puts("");
	flag_init_shape = 1;
	//system("pause");
}

void update_slt_ddex(
	Eigen::MatrixXf &exp_r_t_all_matrix, std::vector<int> *slt_line,
	std::vector<std::pair<int, int> > *slt_point_rect, DataPoint &data) {
	////////////////////////////////project

	puts("updating silhouette...");
	//Eigen::Matrix3f R = data.shape.rot;
	Eigen::Matrix3f R = get_r_from_angle_zyx(data.shape.angle);
#ifdef perspective
	Eigen::Vector3f T = data.shape.tslt;
#endif // perspective
#ifdef normalization
	Eigen::Vector3f T = data.shape.tslt.transpose();
#endif // normalization


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
		int en = slt_line[i].size();
		if (data.shape.angle(1) < -0.1 && i < 34) en /= 3;
		if (data.shape.angle(1) < -0.1 && i >= 34 && i < 41) en /= 2;
		if (data.shape.angle(1) > 0.1 && i >= 49 && i < 84) en /= 3;
		if (data.shape.angle(1) > 0.1 && i >= 42 && i < 49) en /= 2;
#ifdef perspective
		for (int j = 0, sz = en; j < sz; j++) {
			//printf("j %d\n", j);
			int x = slt_line[i][j];
			//printf("x %d\n", x);
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++)
				point[0](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);

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
					point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].first, axis);
					point[2](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].second, axis);
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
		cdnt(0) = cdnt(0)*data.fcs / cdnt(2) + data.center(0);
		cdnt(1) = cdnt(1)*data.fcs / cdnt(2) + data.center(1);
		slt_cddt_cdnt.row(i) = cdnt.transpose();
#endif // perspective
#ifdef normalization
		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
			//			printf("j %d\n", j);
			int x = slt_line[i][j];
			//			printf("j %d x %d ",j, x);
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++)
				point[0](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);
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
					point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].first, axis);
					point[2](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].second, axis);
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
		cdnt.block(0, 0, 2, 1) = data.s*cdnt + T.block(0, 0, 2, 1);
		slt_cddt_cdnt.row(i) = cdnt.transpose();
		//		printf("\n=-= + %d %d %.5f %.5f\n",i,min_idx,cdnt(0),cdnt(1));
		/*		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
					//printf("j %d\n", j);
					int x = slt_line[i][j];
					//printf("x %d\n", x);
					Eigen::Vector3f point, temp;
					for (int axis = 0; axis < 3; axis++)
						point(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix,data.shape.exp, x, axis);
					point = R * point;
					temp = point;
					point.normalize();
					if (fabs(point(2)) < min_v_n) min_v_n = fabs(point(2)), min_idx = x, cdnt = temp;// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
				}
				slt_cddt(i) = min_idx;
				cdnt.block(0, 0, 2, 1) = data.s*cdnt + T.block(0, 0, 2, 1);
				/*cdnt(0) = cdnt(0)*ide[id_idx].s(exp_idx, 0) + T(0);
				cdnt(1) = cdnt(1)*ide[id_idx].s(exp_idx, 1) + T(1);
				slt_cddt_cdnt.row(i) = cdnt.transpose();*/
#endif

	}
	//fclose(fp);
	//puts("C");
	for (int i = 0; i < 15; i++) {
		//printf("C i %d\n", i);
		float min_dis = 10000;
		int min_idx = 0;
		int be = 0, en = G_land_num;
		if (i < 7) be = 41, en = 84;
		if (i > 7) be = 0, en = 42;
		for (int j = be; j < en; j++) {
#ifdef perspective
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - data.land_2d(i, 0) + data.shape.dis(i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - data.land_2d(i, 1) + data.shape.dis(i, 1));
#endif // perspective
#ifdef normalization
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - data.land_2d(i, 0) + data.shape.dis(i, 0)) + //data.shape.dis(i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - data.land_2d(i, 1) + data.shape.dis(i, 1));// +data.shape.dis(i, 1));
#endif // normalization


			if (temp < min_dis) min_dis = temp, min_idx = j;
		}
		//		printf("%d %d %d %.10f\n", i, min_idx, slt_cddt(min_idx),min_dis);
		data.land_cor(i) = slt_cddt(min_idx);

	}
	data.shape.land_cor=data.land_cor;
	//std :: cout << "slt_cddt_cdnt\n" << slt_cddt_cdnt.block(0,0, slt_cddt_cdnt.rows(),2) << "\n";
	//std::cout << "out land\n" << data.land_2d.block(0, 0,15,2) << "\n";
	//std::cout << "land correlation\n" << data.land_cor.transpose() << "\n";
	//system("pause");
}

void update_slt_ddex_db(
	cv::Mat img,
	Eigen::MatrixXf &exp_r_t_all_matrix, std::vector<int> *slt_line,
	std::vector<std::pair<int, int> > *slt_point_rect, DataPoint &data) {
	////////////////////////////////project

	puts("updating silhouette...");
	//Eigen::Matrix3f R = data.shape.rot;
	Eigen::Matrix3f R = get_r_from_angle_zyx(data.shape.angle);
#ifdef perspective
	Eigen::Vector3f T = data.shape.tslt;
#endif // perspective
#ifdef normalization
	Eigen::Vector3f T = data.shape.tslt.transpose();
#endif // normalization


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
		int en = slt_line[i].size();
		if (data.shape.angle(1) < -0.1 && i < 34) en /= 3;
		if (data.shape.angle(1) < -0.1 && i >= 34 && i < 41) en /= 2;
		if (data.shape.angle(1) > 0.1 && i >= 49 && i < 84) en /= 3;
		if (data.shape.angle(1) > 0.1 && i >= 42 && i < 49) en /= 2;
#ifdef perspective
		for (int j = 0, sz = en; j < sz; j++) {
			//printf("j %d\n", j);
			int x = slt_line[i][j];
			//printf("x %d\n", x);
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++)
				point[0](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);

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
					point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].first, axis);
					point[2](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].second, axis);
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
		cdnt(0) = cdnt(0)*data.fcs / cdnt(2) + data.center(0);
		cdnt(1) = cdnt(1)*data.fcs / cdnt(2) + data.center(1);
		cv::circle(img, cv::Point2f(cdnt(0), img.rows-cdnt(1)),
			 0.1, cv::Scalar(255, 255, 255), 2);
		slt_cddt_cdnt.row(i) = cdnt.transpose();
#endif // perspective
#ifdef normalization
		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
			//			printf("j %d\n", j);
			int x = slt_line[i][j];
			//			printf("j %d x %d ",j, x);
			Eigen::Vector3f nor;
			nor.setZero();
			Eigen::Vector3f V[2], point[3];
			for (int axis = 0; axis < 3; axis++)
				point[0](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);
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
					point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].first, axis);
					point[2](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].second, axis);
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
		cdnt.block(0, 0, 2, 1) = data.s*cdnt + T.block(0, 0, 2, 1);
		slt_cddt_cdnt.row(i) = cdnt.transpose();
		//		printf("\n=-= + %d %d %.5f %.5f\n",i,min_idx,cdnt(0),cdnt(1));
		/*		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
					//printf("j %d\n", j);
					int x = slt_line[i][j];
					//printf("x %d\n", x);
					Eigen::Vector3f point, temp;
					for (int axis = 0; axis < 3; axis++)
						point(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix,data.shape.exp, x, axis);
					point = R * point;
					temp = point;
					point.normalize();
					if (fabs(point(2)) < min_v_n) min_v_n = fabs(point(2)), min_idx = x, cdnt = temp;// printf("%.6f %.6f %.6f \n", point[0](0), point[0](1), point[0](2));
				}
				slt_cddt(i) = min_idx;
				cdnt.block(0, 0, 2, 1) = data.s*cdnt + T.block(0, 0, 2, 1);
				/*cdnt(0) = cdnt(0)*ide[id_idx].s(exp_idx, 0) + T(0);
				cdnt(1) = cdnt(1)*ide[id_idx].s(exp_idx, 1) + T(1);
				slt_cddt_cdnt.row(i) = cdnt.transpose();*/
#endif

	}
	//fclose(fp);
	//puts("C");
	for (int i = 0; i < 15; i++) {
		//printf("C i %d\n", i);
		cv::circle(img, cv::Point2f(data.land_2d(i, 0),img.rows-data.land_2d(i, 1)),
					 1, cv::Scalar(255, 0, 0), 0.5);
		float min_dis = 10000;
		int min_idx = 0;
		int be = 0, en = G_land_num;
		if (i < 7) be = 41, en = 84;
		if (i > 7) be = 0, en = 42;
		for (int j = be; j < en; j++) {
#ifdef perspective
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - data.land_2d(i, 0) + data.shape.dis(i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - data.land_2d(i, 1)+ data.shape.dis(i, 1));
#endif // perspective
#ifdef normalization
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - data.land_2d(i, 0) + data.shape.dis(i, 0)) + //data.shape.dis(i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - data.land_2d(i, 1) + data.shape.dis(i, 1));// +data.shape.dis(i, 1));
#endif // normalization


			if (temp < min_dis) min_dis = temp, min_idx = j;
		}
		//		printf("%d %d %d %.10f\n", i, min_idx, slt_cddt(min_idx),min_dis);
		data.land_cor(i) = slt_cddt(min_idx);

	}
	data.shape.land_cor=data.land_cor;
	//std :: cout << "slt_cddt_cdnt\n" << slt_cddt_cdnt.block(0,0, slt_cddt_cdnt.rows(),2) << "\n";
	//std::cout << "out land\n" << data.land_2d.block(0, 0,15,2) << "\n";
	//std::cout << "land correlation\n" << data.land_cor.transpose() << "\n";
	//system("pause");
}

void update_slt_ddex_me(
	Eigen::MatrixXf &exp_r_t_all_matrix, std::vector<int> *slt_line,
	std::vector<std::pair<int, int> > *slt_point_rect, DataPoint &data) {
	////////////////////////////////project
	puts("updating silhouette...");
	Eigen::Matrix3f R = get_r_from_angle_zyx(data.shape.angle);
	Eigen::VectorXf angle = data.shape.angle;
#ifdef perspective
	Eigen::Vector3f T = data.shape.tslt;
	float f = data.fcs;
#endif // perspective
#ifdef normalization
	Eigen::Vector3f T = data.init_shape.tslt.transpose();
#endif // normalization

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
					point(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);
				
				point = R * point + T;
				point(0) = point(0)*f / point(2) + data.center( 0);
				point(1) = point(1)*f / point(2) + data.center( 1);
				//test_slt_2dpt.push_back(cv::Point2f(point(0), point(1)));
				for (int p = 0; p < 15; p++) {
					float temp = (point.block(0, 0, 2, 1).transpose() - data.land_2d.row(p) + data.shape.dis.row(p)).squaredNorm();
					if (temp < land_cor_mi(p)) {
						land_cor_mi(p) = temp;
						data.land_cor(p) = x;
					}
				}
			}

#endif // perspective

		}
		data.shape.land_cor = data.land_cor;
		std::cout << "land_cor_mi:\n" << land_cor_mi.transpose() << "\n";

		std::cout << "out land correlation\n" << data.land_cor.transpose() << "\n";
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
					point(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);
				point = R * point + T;
				point(0) = point(0)*f / point(2) + data.center( 0);
				point(1) = point(1)*f / point(2) + data.center( 1);
				test_slt_2dpt.push_back(cv::Point2f(point(0), point(1)));
				for (int p = 8; p < 15; p++) {
					float temp = (point.block(0, 0, 2, 1).transpose() - data.land_2d.row(p) + data.shape.dis.row(p)).squaredNorm();
					if (temp < land_cor_mi(p)) {
						land_cor_mi(p) = temp;
						data.land_cor(p) = x;
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
					point[0](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);
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
						point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].first, axis);;
						point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].second, axis);;
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

			cdnt(0) = cdnt(0)*f / cdnt(2) + data.center(0);
			cdnt(1) = cdnt(1)*f / cdnt(2) + data.center(1);
			for (int p = 0; p < 12; p++) {
				float temp = (cdnt.block(0, 0, 2, 1).transpose() - data.land_2d.row(p) + data.shape.dis.row(p)).squaredNorm();

				if (temp < land_cor_mi(p)) {
					land_cor_mi(p) = temp;
					data.land_cor(p) = min_idx;
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
					point(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);
				point = R * point + T;
				point(0) = point(0)*f / point(2) + data.center(0);
				point(1) = point(1)*f / point(2) + data.center(1);
				for (int p = 0; p < 7; p++) {
					float temp = (point.block(0, 0, 2, 1).transpose() - data.land_2d.row(p) + data.shape.dis.row(p)).squaredNorm();


					if (temp < land_cor_mi(p)) {
						land_cor_mi(p) = temp;
						data.land_cor(p) = x;
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
					point[0](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, x, axis);
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
						point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].first, axis);;
						point[1](axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp, slt_point_rect[x][k].second, axis);;
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

			cdnt(0) = cdnt(0)*f / cdnt(2) + data.center(0);
			cdnt(1) = cdnt(1)*f / cdnt(2) + data.center(1);
			for (int p = 4; p < 15; p++) {
				float temp = (cdnt.block(0, 0, 2, 1).transpose() - data.land_2d.row(p)+data.shape.dis.row(p)).squaredNorm();
				if (temp < land_cor_mi(p)) {
					land_cor_mi(p) = temp;
					data.land_cor(p) = min_idx;
				}
			}
#endif // perspective

		}
	}
	data.shape.land_cor = data.land_cor;
	//std::cout << "land_cor_mi:\n" << land_cor_mi.transpose() << "\n";

	//std::cout << "out land correlation\n" << out_land_cor.transpose() << "\n";
	//system("pause");
}

void get_init_shape_ed(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {
//[15~35)+[46,73)

	puts("calculating initial shape");
	float mi_land[G_dde_K]; int idx[G_dde_K];
	for (int i = 0; i < G_dde_K; i++) mi_land[i] = 100000000, idx[i] = -1;
	Eigen::MatrixXf da_d(20+27, 2);
	da_d.block(0, 0, 20, 2) = data.land_2d.block(15, 0, 20, 2);
	da_d.block(20,0,27,2)= data.land_2d.block(46, 0, 27, 2);	
	da_d.rowwise() -= da_d.colwise().mean();

	for (int i = 0; i < train_data.size(); i++) {	

		Eigen::MatrixXf tr_d(20+27, 2);
		tr_d.block(0, 0, 20, 2) = train_data[i].land_2d.block(15, 0, 20, 2);
		tr_d.block(20, 0, 27, 2) = train_data[i].land_2d.block(46, 0, 27, 2);
		tr_d.rowwise() -= tr_d.colwise().mean();
		
		float distance_land = (da_d - tr_d).squaredNorm();


		//for (int j=0;j<G_dde_K;j++)
		//	if (distance_land < mi_land[j]) {
		//		for (int k = G_dde_K - 1; k > j; k--) {
		//			idx[k] = idx[k - 1];
		//			mi_land[k] = mi_land[k - 1];
		//		}
		//		idx[j] = i;
		//		mi_land[j] = distance_land;
		//		for (int t = 0; t < G_dde_K; t++)
		//			printf("%d ", idx[t]);
		//		puts("");
		//		break;
		//	}
		for (int j = G_dde_K - 1; j >= 0; j--) {
			if (distance_land > mi_land[j]) {
				j++;
				if (j == G_dde_K) break;
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("+++");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("++++");
				break;
			}
			if (j == 0) {
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("");
				break;
			}
		}
	}
	for (int i = 0; i < G_dde_K; i++) {
		//ans[i] = data.shape;
		ans[i].exp = train_data[idx[i]].shape.exp;
		ans[i].dis = train_data[idx[i]].shape.dis;
		ans[i].land_cor = data.land_cor;
		ans[i].angle = data.shape.angle;
		ans[i].tslt = data.shape.tslt;

		printf("%d ", idx[i]);
	}
	puts("");
	//system("pause");
}

void get_init_shape_at(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {
//[0,15)+[35,46)+[64,64)

	puts("calculating initial shape");
	float mi_land[G_dde_K]; int idx[G_dde_K];
	for (int i = 0; i < G_dde_K; i++) mi_land[i] = 100000000, idx[i] = -1;
	Eigen::MatrixXf da_d(G_outer_land_num+11+1, 2);
	da_d.block(0, 0, 15, 2) = data.land_2d.block(0, 0, 15, 2);
	da_d.row(G_outer_land_num) = data.land_2d.row(64);
	da_d.block(G_outer_land_num+1, 0, 11, 2) = data.land_2d.block(35, 0, 11, 2);
	
	da_d.rowwise() -= da_d.colwise().mean();

	for (int i = 0; i < train_data.size(); i++) {

		Eigen::MatrixXf tr_d(G_outer_land_num + 11 + 1, 2);
		tr_d.block(0, 0, 15, 2) = train_data[i].land_2d.block(0, 0, 15, 2);
		tr_d.row(G_outer_land_num) = train_data[i].land_2d.row(64);
		tr_d.block(G_outer_land_num+1, 0, 11, 2) = train_data[i].land_2d.block(35, 0, 11, 2);
		tr_d.rowwise() -= tr_d.colwise().mean();

		float distance_land = (da_d - tr_d).squaredNorm();


		//for (int j=0;j<G_dde_K;j++)
		//	if (distance_land < mi_land[j]) {
		//		for (int k = G_dde_K - 1; k > j; k--) {
		//			idx[k] = idx[k - 1];
		//			mi_land[k] = mi_land[k - 1];
		//		}
		//		idx[j] = i;
		//		mi_land[j] = distance_land;
		//		for (int t = 0; t < G_dde_K; t++)
		//			printf("%d ", idx[t]);
		//		puts("");
		//		break;
		//	}
		for (int j = G_dde_K - 1; j >= 0; j--) {
			if (distance_land > mi_land[j]) {
				j++;
				if (j == G_dde_K) break;
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("+++");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("++++");
				break;
			}
			if (j == 0) {
				for (int k = G_dde_K - 1; k > j; k--) {
					idx[k] = idx[k - 1];
					mi_land[k] = mi_land[k - 1];
				}
				idx[j] = i;
				mi_land[j] = distance_land;
				for (int t = 0; t < G_dde_K; t++)
					printf("%d ", idx[t]);
				puts("");
				for (int t = 0; t < G_dde_K; t++)
					printf("%.5f ", mi_land[t]);
				puts("");
				break;
			}
		}
	}
	for (int i = 0; i < G_dde_K; i++) {
		//ans[i] = data.shape;
		ans[i].angle = train_data[idx[i]].shape.angle;
		ans[i].land_cor = train_data[idx[i]].shape.land_cor;

/*		ans[i].tslt = train_data[idx[i]].shape.tslt;
		ans[i].tslt.block(0, 0, 2, 1) -=
			((train_data[idx[i]].centeroid - data.centeroid) / train_data[idx[i]].fcs*ans[i].tslt(2)).transpose();
*/		

		printf("%d ", idx[i]);
	}
	puts("");
	//system("pause");
}

void get_init_shape_part(std::vector<Target_type> &ans, DataPoint &data, std::vector<DataPoint>&train_data) {
	get_init_shape_ed(ans, data, train_data);
//	get_init_shape_at(ans, data, train_data);
}

int debug_ite = 0;

void DDEX::dde(
	cv::Mat debug_init_img, DataPoint &data, Eigen::MatrixXf &bldshps,
	Eigen::MatrixX3i &tri_idx, std::vector<DataPoint> &train_data,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,Eigen::MatrixXf &exp_r_t_all_matrix)const {


	//std::cout << tri_idx.transpose() //<< "\n";
	std::cout << "land-cor: " << data.land_cor << "\n";
	std::cout << "shape land-cor: " << data.shape.land_cor << "\n";
//	exit(9);

	Target_type result;
	result.dis.resize(G_land_num, 2);
	result.dis.setZero();
	result.exp.resize(G_nShape);
	result.exp.setZero();
	result.tslt.setZero();
	//result.rot.setZero();
	result.angle.setZero();
	
//	show_image_0rect(data.image, data.landmarks);
#ifdef debug_init
	cv::Mat init_img = debug_init_img.clone();
	for (cv::Point2d landmark : data.landmarks)
	{
		cv::circle(init_img, landmark, 1, cv::Scalar(0, 250, 0), 2);
	}
#endif

	std::cout << "older angle:" << ((data.shape.angle)*180/pi) << "\n";
//	change_nearest(data,train_data);
//	change_nearest_slt(data,train_data);
	std::cout << "new angle:" << ((data.shape.angle) * 180 / pi) << "\n";
	//system("pause");
/*
	std::cout <<data.shape.dis << "\n";
	show_image_land_2d(data.image, data.land_2d);
	*/
//#ifdef update_slt_def
//	update_slt_ddex(exp_r_t_all_matrix, slt_line, slt_point_rect, jaw_land_corr, data);
//#endif // update_slt_def
	//show_image_0rect(data.image, data.landmarks);
	//std::cout << data.shape.dis << "\n";
	//print_datapoint(data);
	

	data.shape.exp(0) = 1;
	update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
#ifdef debug_init
	for (cv::Point2d landmark : data.landmarks)
	{
		cv::circle(init_img, landmark, 1, cv::Scalar(0, 0, 250), 2);
	}
//	cv::imshow("nearestdebug",init_img);
//	cv::waitKey(0);
#endif
	//debug_ite++;
	//for (cv::Point2d landmark : data.landmarks)
	//{
	//	cv::circle(G_debug_up_image, landmark, 0.1*debug_ite, cv::Scalar(0,0,debug_ite*20), 2);
	//}
/*	std::cout << data.landmarks << "\n";
	puts("look here A!");	
	show_image_0rect(data.image, data.landmarks);*/

	//find init
	long long start_time = cv::getTickCount();
	std::cout << "land-cor: " << data.land_cor << "\n";
	std::cout << "shape land-cor: " << data.shape.land_cor << "\n";
#ifdef no_nearest_me
	std::vector<Target_type> init_shape(G_dde_K);

	//get_init_shape_rand(init_shape, data, train_data);
	//get_init_shape_exp(init_shape, data, train_data);
	puts("A");
	for (int i=0;i<G_dde_K;i++) init_shape[i]=data.shape;
	puts("AB");
#else
	std::vector<Target_type> init_shape(G_dde_K);

	get_init_shape_part(init_shape, data, train_data);
//	get_init_shape(init_shape, data, train_data);
//	get_init_shape_inner(init_shape, data, train_data);
#endif // no_nearest_me

	

	FILE *fp;
	fp=fopen( "debug_ite_k_near_idx.txt", "a");
	fprintf(fp, "%d ", ++debug_ite);
	for (int i=0;i<G_dde_K;i++) fprintf(fp, " %d", init_shape_la[i].first);	
	fprintf(fp, "\n");
	//for (int i = 0; i < G_dde_K; i++) fprintf(fp, " %.3f", init_shape[i].angle(1));
	for (int i = 0; i < G_dde_K; i++) fprintf(fp, " %.3f", init_shape_la[i].second);
	fprintf(fp, "\n");
	fprintf(fp, "%.5f\n",data.shape.angle(1));

	fclose(fp);

	std::cout << "finding time: "
		<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << endl;
	for (int i = 0; i < init_shape.size(); ++i)
	{
		printf("%d init shape\n", i);
		//Transform t = Procrustes(initial_landmarks, test_init_shapes_[i]);
		//t.Apply(&init_shape);

		Target_type result_shape = init_shape[i];
		
		std::vector<cv::Point2d> land_temp;
		result_shape.exp(0) = 1;
//		std::cout << "result before tslt: "<< result_shape.tslt <<"\n";		
		cal_2d_land_i_ang_0ide(land_temp, result_shape, exp_r_t_all_matrix, data);
#ifdef debug_init
		for (cv::Point2d landmark : land_temp)
		{
			cv::circle(init_img, landmark, 0.1, cv::Scalar(250, 250, 220), 2);
		}
#endif // debug_init

/*		print_target(result_shape);
		puts("look here B!");	
		show_image_0rect(data.image, land_temp);*/

		long long start_time = cv::getTickCount();
#ifdef intersection
		for (int j = 0; j < stage_regressors_dde_.size(); ++j)
		{
			//printf("outer regressor %d:\n", j);
			//Transform t = Procrustes(init_shape, mean_shape_);
			result_shape.exp(0) = 1;


			Target_type offset =
				stage_regressors_dde_[j].Apply_ta(result_shape,tri_idx,data,bldshps, exp_r_t_all_matrix);
			//t.Apply(&offset, false);

			result_shape = shape_adjustment(result_shape, offset);
			result_shape.exp(0) = 1;
			offset =
				stage_regressors_dde_[j].Apply_expdis(result_shape, tri_idx, data, bldshps, exp_r_t_all_matrix);
			//t.Apply(&offset, false);
			result_shape = shape_adjustment(result_shape, offset);
			//printf("outer regressor == %d:\n", j);
		}
#else
		for (int j = 0; j < stage_regressors_dde_.size(); ++j)
		{
			//printf("outer regressor %d:\n", j);
			//Transform t = Procrustes(init_shape, mean_shape_);
			result_shape.exp(0) = 1;


			Target_type offset =
				stage_regressors_dde_[j].Apply_ta(result_shape,tri_idx,data,bldshps, exp_r_t_all_matrix);
			//t.Apply(&offset, false);

			result_shape = shape_adjustment(result_shape, offset);
			//printf("outer regressor == %d:\n", j);
//			std::cout << "offset tslt: "<< offset.tslt <<"\n";
		}
//		std::cout << "result after tslt: "<< result_shape.tslt <<"\n";		
//		exit(9);
		for (int j = 0; j < stage_regressors_dde_.size(); ++j)
		{
			//printf("outer regressor %d:\n", j);
			//Transform t = Procrustes(init_shape, mean_shape_);
			result_shape.exp(0) = 1;
			Target_type offset =
				stage_regressors_dde_[j].Apply_expdis(result_shape, tri_idx, data, bldshps, exp_r_t_all_matrix);
			//t.Apply(&offset, false);
			result_shape = shape_adjustment(result_shape, offset);
			//printf("outer regressor == %d:\n", j);
		}
#endif
		std::cout << "Alignment time: "
			<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
			<< "s" << endl;
		//std::vector<cv::Point2d> land_temp;
//-----------------------------------------------------------------------------------------------------------

		result_shape.exp(0) = 1;
/*		cal_2d_land_i_ang_0ide(land_temp, result_shape, exp_r_t_all_matrix,data);
		print_target(result_shape);
		puts("look here C!");	
		show_image_0rect(data.image, land_temp);*/

		result.dis.array() += result_shape.dis.array();
		result.exp.array() += result_shape.exp.array();
		//result.rot.array() += result_shape.rot.array();
		result.angle.array() += result_shape.angle.array();
		result.tslt.array() += result_shape.tslt.array();

	}
#ifdef debug_init
	debug_init_output_video << init_img;
#endif // debug_init
	result.dis.array() /= init_shape.size();
	result.exp.array() /= init_shape.size();
	//result.rot.array() /= G_dde_K;
	result.angle.array() /= init_shape.size();
	result.tslt.array() /= init_shape.size();
	data.shape = result;
	data.shape.exp(0) = 1;
	data.shape.land_cor = data.land_cor;
	
#ifdef update_slt_def
	update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
/*	cv::Mat db_slt_img = debug_init_img.clone();
	for (cv::Point2d landmark : data.landmarks)
	{
		cv::circle(db_slt_img, landmark, 1, cv::Scalar(0, 250,0), 2);
	}*/
	//update_slt_ddex_db(db_slt_img,exp_r_t_all_matrix, slt_line, slt_point_rect, data);
	update_slt_ddex_me(exp_r_t_all_matrix, slt_line, slt_point_rect, data);
#endif // update_slt_def
	update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
/*	for (cv::Point2d landmark : data.landmarks)
	{
		cv::circle(db_slt_img, landmark, 1, cv::Scalar(0,0, 250), 2);
	}*/
//	cv::imshow("db_slt_img",db_slt_img);
//	cv::waitKey(0);
}

void DDEX::dde_onlyexpdis(
	cv::Mat debug_init_img, DataPoint &data, Eigen::MatrixXf &bldshps,
	Eigen::MatrixX3i &tri_idx, std::vector<DataPoint> &train_data,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect, Eigen::MatrixXf &exp_r_t_all_matrix)const {


	Target_type result_shape = data.shape;

	std::vector<cv::Point2d> land_temp;

	long long start_time = cv::getTickCount();

	for (int j = 0; j < stage_regressors_dde_.size(); ++j)
	{
		//printf("outer regressor %d:\n", j);
		//Transform t = Procrustes(init_shape, mean_shape_);
		result_shape.exp(0) = 1;
		Target_type offset =
			stage_regressors_dde_[j].Apply_expdis(result_shape, tri_idx, data, bldshps, exp_r_t_all_matrix);
		//t.Apply(&offset, false);
		result_shape = shape_adjustment(result_shape, offset);
		//printf("outer regressor == %d:\n", j);
	}
	std::cout << "Alignment time: "
		<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << endl;
	//std::vector<cv::Point2d> land_temp;
//-----------------------------------------------------------------------------------------------------------

		//result_shape.exp(0) = 1;
		//cal_2d_land_i_ang_0ide(land_temp, result_shape, exp_r_t_all_matrix,data);
		//print_target(result_shape);
		//show_image_0rect(data.image, land_temp);
	data.shape = result_shape;

#ifdef update_slt_def
	update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
	update_slt_ddex(exp_r_t_all_matrix, slt_line, slt_point_rect, data);
#endif // update_slt_def
	update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
}



void DDEX::visualize_feature_cddt(cv::Mat rbg_image, Eigen::MatrixX3i &tri_idx, std::vector<cv::Point2d> &landmarks)const {

	puts("dde... visualizing feature index");
	for (int j = 0; j < stage_regressors_dde_.size(); ++j) {
		stage_regressors_dde_[j].visualize_feature_cddt_ta(rbg_image, tri_idx,landmarks);
		stage_regressors_dde_[j].visualize_feature_cddt_expdis(rbg_image, tri_idx, landmarks);
	}
}

