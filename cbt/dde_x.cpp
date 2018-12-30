#include "dde_x.hpp"

#include <algorithm>
#include <stdexcept>
#define update_slt_def
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
	for (int i = 0; i < train_data.size(); i++) {
		//float distance = (data.center - train_data[i].center).norm();
		float distance_land = ((data.land_2d.rowwise()-data.center) - (train_data[i].land_2d.rowwise()-train_data[i].center)).norm();
		if (distance_land < mi_land) {
			idx = i;
			mi_land = distance_land;
			printf("idx %d dis%.10f center :%.10f %.10f train_center :%.10f %.10f\n",
				idx,mi_land,data.center(0), data.center(1), train_data[i].center(0), train_data[i].center(1));
		}
	}
	printf("nearest vertex: %d train dataset size: %d\n", idx,train_data.size());
	//data.shape.rot = train_data[idx].shape.rot;
	data.shape.angle = train_data[idx].shape.angle;
	FILE *fp;
	fopen_s(&fp, "debug_ite_nearest_idx.txt", "a");
	fprintf(fp, "%d %.10f\n", idx,mi_land);
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
			((data.land_2d.rowwise() - data.center) -
			(train_data[init_shape_la[i].first].land_2d.rowwise() - train_data[init_shape_la[i].first].center)).norm();
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
		float distance_land = ((data.land_2d - train_data[i].land_2d).rowwise()-(data.center-train_data[i].center)).norm();
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

//align the center by tslt
		ans[i].tslt.block(0, 0, 1, 2) -= train_data[idx[i]].center;
		ans[i].tslt.block(0, 0, 1, 2) += data.center;


		init_shape_la[i].first = idx[i];
		printf("%d ",idx[i]);
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
			((data.land_2d.block(15,0,G_inner_land_num,2).rowwise() - data.center) -
			(train_data[init_shape_la[i].first].land_2d.block(15, 0, G_inner_land_num, 2).rowwise() - train_data[init_shape_la[i].first].center)).norm();
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
			- (data.center - train_data[i].center)).norm();
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

		//align the center by tslt
		ans[i].tslt.block(0, 0, 1, 2) -= train_data[idx[i]].center;
		ans[i].tslt.block(0, 0, 1, 2) += data.center;


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

void update_slt(
	Eigen::MatrixXf &exp_r_t_all_matrix,std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,
	Eigen::VectorXi &jaw_land_corr, DataPoint &data) {
	////////////////////////////////project

	puts("updating silhouette...");
	//Eigen::Matrix3f R = data.shape.rot;
	Eigen::Matrix3f R = get_r_from_angle_zyx(data.shape.angle);
	Eigen::Vector3f T = data.shape.tslt.transpose();

	//puts("A");
	Eigen::VectorXi slt_cddt(G_line_num + G_jaw_land_num);
	Eigen::MatrixX3f slt_cddt_cdnt(G_line_num + G_jaw_land_num, 3);
	//puts("B");
	//FILE *fp;
	//fopen_s(&fp, "test_slt.txt", "w");
	for (int i = 0; i < G_line_num; i++) {
		//printf("i %d\n", i);
		float min_v_n = 10000;
		int min_idx = 0;
		Eigen::Vector3f cdnt;

#ifdef normalization
		for (int j = 0, sz = slt_line[i].size(); j < sz; j++) {
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
		cdnt(1) = cdnt(1)*ide[id_idx].s(exp_idx, 1) + T(1);*/
		slt_cddt_cdnt.row(i) = cdnt.transpose();
#endif

	}
	//fclose(fp);
	//puts("C");
	for (int i_jaw = 0; i_jaw < G_jaw_land_num; i_jaw++) {
		//printf("B i %d\n", i_jaw);
		Eigen::Vector3f point;
		for (int axis = 0; axis < 3; axis++)
			point(axis) = cal_3d_vtx_0ide(exp_r_t_all_matrix, data.shape.exp,jaw_land_corr(i_jaw), axis);
#ifdef normalization
		point.block(0, 0, 2, 1) = data.s*R * point;
		//point(0) *= ide[id_idx].s(exp_idx, 0), point(1) *= ide[id_idx].s(exp_idx, 1);
		point = point + T;
#endif // normalization		
		slt_cddt(i_jaw + G_line_num) = jaw_land_corr(i_jaw);
		slt_cddt_cdnt.row(i_jaw + G_line_num) = point.transpose();
	}
	for (int i = 0; i < 15; i++) {
		//printf("C i %d\n", i);
		float min_dis = 10000;
		int min_idx = 0;;
		for (int j = 0; j < G_line_num + G_jaw_land_num; j++) {
#ifdef posit
			float temp =
				fabs(slt_cddt_cdnt(j, 0) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 0)) +
				fabs(slt_cddt_cdnt(j, 1) - ide[id_idx].land_2d(G_land_num*exp_idx + i, 1));
#endif // posit
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
	//std :: cout << "slt_cddt_cdnt\n" << slt_cddt_cdnt.block(0,0, slt_cddt_cdnt.rows(),2) << "\n";
	//std::cout << "out land\n" << data.land_2d.block(0, 0,15,2) << "\n";
	//std::cout << "land correlation\n" << data.land_cor.transpose() << "\n";
	//system("pause");
}


int debug_ite = 0;

void DDEX::dde(
	cv::Mat debug_init_img, DataPoint &data, Eigen::MatrixXf &bldshps,
	Eigen::MatrixX3i &tri_idx, std::vector<DataPoint> &train_data, Eigen::VectorXi &jaw_land_corr,
	std::vector<int> *slt_line, std::vector<std::pair<int, int> > *slt_point_rect,Eigen::MatrixXf &exp_r_t_all_matrix)const {


	//std::cout << tri_idx.transpose() << "\n";

	Target_type result;
	result.dis.resize(G_land_num, 2);
	result.dis.setZero();
	result.exp.resize(G_nShape);
	result.exp.setZero();
	result.tslt.setZero();
	//result.rot.setZero();
	result.angle.setZero();
	
	//show_image_0rect(data.image, data.landmarks);

	std::cout << "older angle:" << ((data.shape.angle)*180/pi) << "\n";
	change_nearest(data,train_data);
	std::cout << "new angle:" << ((data.shape.angle) * 180 / pi) << "\n";
	//system("pause");

	//std::cout <<data.shape.dis << "\n";
	//show_image_land_2d(data.image, data.land_2d);
//#ifdef update_slt_def
//	update_slt(exp_r_t_all_matrix, slt_line, slt_point_rect, jaw_land_corr, data);
//#endif // update_slt_def
	//show_image_0rect(data.image, data.landmarks);
	//std::cout << data.shape.dis << "\n";
	//print_datapoint(data);
	

	data.shape.exp(0) = 1;
	update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
	//debug_ite++;
	//for (cv::Point2d landmark : data.landmarks)
	//{
	//	cv::circle(G_debug_up_image, landmark, 0.1*debug_ite, cv::Scalar(0,0,debug_ite*20), 2);
	//}
	//std::cout << data.landmarks << "\n";
	
	//show_image_0rect(data.image, data.landmarks);

	//find init
	long long start_time = cv::getTickCount();
#ifdef no_nearest_me
	std::vector<Target_type> init_shape(G_dde_K);

	//get_init_shape_rand(init_shape, data, train_data);
	get_init_shape_exp(init_shape, data, train_data);
#else
	std::vector<Target_type> init_shape(G_dde_K);

	get_init_shape(init_shape, data, train_data);
#endif // no_nearest_me

	

	FILE *fp;
	fopen_s(&fp, "debug_ite_k_near_idx.txt", "a");
	fprintf(fp, "%d ", debug_ite);
	for (int i=0;i<G_dde_K;i++) fprintf(fp, " %d", init_shape_la[i].first);
	fprintf(fp, "\n");
	fclose(fp);

	std::cout << "finding time: "
		<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << endl;
#ifdef debug_init
	cv::Mat init_img = data.image.clone();
#endif
	for (int i = 0; i < init_shape.size(); ++i)
	{
		printf("%d init shape\n", i);
		//Transform t = Procrustes(initial_landmarks, test_init_shapes_[i]);
		//t.Apply(&init_shape);

		Target_type result_shape = init_shape[i];
		
		std::vector<cv::Point2d> land_temp;
		result_shape.exp(0) = 1;
		cal_2d_land_i_ang_0ide(land_temp, result_shape, exp_r_t_all_matrix, data);
#ifdef debug_init
		for (cv::Point2d landmark : land_temp)
		{
			cv::circle(debug_init_img, landmark, 0.1, cv::Scalar(250, 250, 220), 2);
		}
#endif // debug_init

		//print_target(result_shape);
		//show_image_0rect(data.image, land_temp);

		long long start_time = cv::getTickCount();

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
		}
		
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

		result.dis.array() += result_shape.dis.array();
		result.exp.array() += result_shape.exp.array();
		//result.rot.array() += result_shape.rot.array();
		result.angle.array() += result_shape.angle.array();
		result.tslt.array() += result_shape.tslt.array();

	}
#ifdef debug_init
	debug_init_output_video << debug_init_img;
#endif // debug_init
	result.dis.array() /= init_shape.size();
	result.exp.array() /= init_shape.size();
	//result.rot.array() /= G_dde_K;
	result.angle.array() /= init_shape.size();
	result.tslt.array() /= init_shape.size();
	data.shape = result;


	
#ifdef update_slt_def
	update_2d_land_ang_0ide(data, exp_r_t_all_matrix);
	update_slt(exp_r_t_all_matrix, slt_line, slt_point_rect, jaw_land_corr, data);
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